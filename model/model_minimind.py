# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig): #继承自 Hugging Face PretrainedConfig，负责设置模型的大小、功能和超参数
    model_type = "minimind" #用于识别模型的类型

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768, #最大序列长度
            num_attention_heads: int = 8, #Q的头数
            num_hidden_layers: int = 8,   #minimindBlock块数
            num_key_value_heads: int = 2, #kv的头数
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0, #rope的基数
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):  #继承torch.nn.Module 所有的模型（无论是层、块还是整个网络）都必须继承 torch.nn.Module 能用 PyTorch 框架的功能
    def __init__(self, dim: int, eps: float = 1e-5):  #初始化     eps是防止除零
        super().__init__()   # 调用 nn.Module 的初始化函数
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))  #nn.Parameter 让这个张量成为模型训练的一部分 ones全1向量

    def _norm(self, x): 
        #pow(2) 对每个元素平方 mean():对-1最后一个维度求平均 keepdim=True 保留维度为 1，方便广播计算
        #rsqrt先开方 再求倒数
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  

    def forward(self, x): 
        # 避免数值不稳定（尤其是 fp16 的溢出 和 精度损失）
        # x.float() 会把张量 转换成 float32（FP32）精度 然后参与_norm()计算
        # type_as(x) 把张量的 dtype 转成和 x 即原来的一模一样的 dtype  python会区分不同精度的浮点数
        return self.weight * self._norm(x.float()).type_as(x)   


#不是网络中的一层 所以不需要用类
#预计算 RoPE 所需的 cos/sin 旋转矩阵（包含 YaRN 长度扩展）
def precompute_freqs_cis_old(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, #end表示序列长
                         rope_scaling: Optional[dict] = None):   # Optional[int] 表示的是rope_scaling的类型 可以接受 dict字典 或 None
    
    #计算频率 torch.arange(0, dim, 2) 产生一个从 0 到 dim-1 的 偶数索引序列（步长为 2）
    # [: (dim // 2)] 取前 dim//2 个元素   .float() 是为了防止前面的张量是整数类型（int），那 / dim 就是整数除法，结果会全部变成 0
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) 
    
    if rope_scaling is not None:   #yarn的缩放处理
        # 元组 从 rope_scaling 这个字典里取参数  字典的 .get(key, default) 方法 如果有这个 key 去key 没有 就默认default
        orig_max, factor, beta_fast, beta_slow = (  
            rope_scaling.get("original_max_position_embeddings", 2048), 
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), 
            rope_scaling.get("beta_slow", 1.0)
        )
        if end / orig_max > 1.0:    #你推理时使用的最大序列长度（end）是否大于模型训练时用的最大序列长度（orig_max)?
            #corr_dim 分界点
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            #把维度索引 i 映射到 [0, 1] 范围，用作线性插值权重   max里传的参数1 是为了 防止除零，保证分母至少为 1。
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power #张量
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,   #生成 [0, 1, 2, ..., dim//2 - 1]，对应每一对 RoPE 维度
                (beta * factor - beta + 1) / (beta * factor), #分界点 corr_dim 之前用公式 (beta * factor - beta + 1)/(beta * factor) 替代
                1.0 / factor #分界点之后统一用 1.0 / factor 替代
            ) 
            freqs = freqs * scale #逐元素相乘

    t = torch.arange(end, device=freqs.device)  #生成一个序列位置的索引
    freqs = torch.outer(t, freqs).float()  #shape [end,dim//2]
    
    #如果说把dim分成这样[q0,q1,q2,q3,q4,q5,] 现在这样是让q0和q3一组旋转 q2和q4一组...  对半配对旋转
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) #shape [end,dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) #shape [end,dim]
    
    # 下面是标准llama做法 相邻的dim一组
    # freqs_cos = torch.cos(freqs).repeat_interleave(2,dim=-1) # 【seq_len,dim】
    # freqs_sin = torch.sin(freqs).repeat_interleave(2,dim=-1) # 【seq_len,dim】
    return freqs_cos, freqs_sin


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


#把 “旋转” 应用到 Q/K 上
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        #原向量 [x, y] → rotate_half → [-y, x]
        #最终 q*cos + rotate_half(q)*sin → [x*cos - y*sin, x*sin + y*cos]
        # -x[..., x.shape[-1] // 2:] 表示取最后一半的元素，再变成负的。
        # x[..., : x.shape[-1] // 2] 取前一半 
        # [-y,x]
        # 按照上面的例子 [-q3,-q4,-q5,q0,q1,q2]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1) #拼接
    
    # q*cos + rotate_half(q)*sin → [x*cos - y*sin, x*sin + y*cos]
    # q\k 的shape [bsz, seq_len, n_heads, head_dim]
    # cos.unsqueeze(1): [seq_len, 1, dim] 方便广播
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

#重复使用kv 这个函数将 KV向量的每个头复制n_rep次
# 将 Key 和 Value 的头数量扩展到 Query 的头数量，以进行注意力计算
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor: #n_rep重复次数
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape  #bs:batchsize  slen:seq_len 
    if n_rep == 1:
        return x
    return (
        # x[:, :, :, None, :] 插入新维度 shape变成[bs, slen, num_key_value_heads, 1, head_dim]
        # expend() 把每个 KV 头(head_dim)重复 n_rep 次，shape变成[bs, slen, num_key_value_heads, n_rep, head_dim] expand 只是逻辑扩展，不增加显存
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

#GQA
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # num_attention_heads 一般指的是Q的头
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0 #断言 是整数倍 不然报错
        self.n_local_heads = args.num_attention_heads #指Q的头数
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads #n_rep个Q头共用一个KV
        self.head_dim = args.hidden_size // args.num_attention_heads  #hidden_size 感觉就是embedding的维度
        
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False) #linear初始化的参数：输入维度 输出维度 是否偏执
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False) #output 对所有注意力头的结果进行加权组合和信息提炼
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout) #残差
        self.dropout = args.dropout
        #flash是一个bool类型的变量 hasattr()检查一个对象是否有指定的属性
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin cos/sin的shape [seq_len,dim]
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, #过去的kv缓存 很好奇 推理的时候past_key_value长什么样
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None): # attention_mask [bsz,seq_len]
        #计算 qkv
        bsz, seq_len, _ = x.shape 
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #拆分成多头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        #对qk使用rope
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len]) #cos[:seq_len] 把cos的维度[seq,dim]切成[seq_len,dim]

        # kv_cache实现 用于推理 可以看出推理阶段的x仅仅是最后一个token
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1) # [past_key_value[0]代表K的缓存 dim=1是在seq维度上拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2), #交换维度 变为[bsz,n_local_head,seq_len,head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  #由于Q的头数和kv的头数不一样 所以进行了repeat 再转置为[bsz,n_local_head（Q的头数）,seq_len,head_dim]
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        
        #进行attention计算 q@k^T/sqrt(d)
        
        # 检查是否满足启用 F.scaled_dot_product_attention 的所有条件 
        # seq_len > 1 代表不在推理阶段
        # attention_mask is None or torch.all(attention_mask == 1) 在pytorch里面有效token是1 （表示没有 padding） 
        # Flash Attention 不支持复杂 padding mask
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 调用函数进行falsh计算 dropout_p=self.dropout if self.training else 0.0 推理的时候不能用dropout self.training是torch.nn.Module的属性
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 自己实现
            # [bsz,n_local_head,seq_len,seq_len]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # @是在两个张量的相应位置的最后两维上进行矩阵相乘 sqrt开方
            
            # scores+causal_mask
            # torch.full((seq_len, seq_len), float("-inf"), device=scores.device) 创建一个形状[seq,seq]元素都是负无穷的张量
            #.triu(..., diagonal=1) 构建上三角 1表示主对角线的右边一位
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  #shape[1,1,seq,seq]广播计算加法

            if attention_mask is not None: #padding_mask
                #attention_mask [bsz,seq_len]一般是标记K序列的token是不是padding  里面1是有效token 0是padding
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #[bsz,1,1,seq_len]
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9 #反转矩阵后再乘极小值
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq) #又做了精度转换 float()精度为32float 模型一般不会用这么高的
            scores = self.attn_dropout(scores)
            output = scores @ xv # [bsz,n_local_head,seq_len,head_dim]

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1) #之前分头的逆运算 shape变成 [bsz,seq,hidden_size]
        output = self.resid_dropout(self.o_proj(output)) #o_proj是一个线性层 融合多头的信息 再做一次dropout
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:  #intermediate_size 是升维后的维度
            intermediate_size = int(config.hidden_size * 8 / 3) # 2.66倍
            # 对齐优化 向上取整到最近的 64 的倍数 
            # GPU 做矩阵乘法（尤其是 transformer 的 Linear 层和注意力计算）时，通常会把数据按块（block）处理，比如 32、64、128 这样的尺寸。
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False) #门控
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False) #降维
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False) #升维
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act] #激活函数 SiLU ACT2FN 是一个字典 ACT2FN["silu"] 

    def forward(self, x):
        # 1. act_fn() 对gate_proj激活 充当一个动态门，它的值（通常在 0-1 之间）决定了up_proj()中的哪些特征应该被保留或放大，哪些应该被抑制或丢弃
        # 2. 激活后 与up_proj()逐元素相乘
        # 3. self.down_proj() 降维 输出变回原来的形状
        # 4. 做一个dropout
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

#路由
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok # 每个 Token 选择多少个专家
        self.n_routed_experts = config.n_routed_experts # 总共有多少个专家可供路由选择。

        self.scoring_func = config.scoring_func # 设置计算专家评分时使用的函数 softmax
        self.alpha = config.aux_loss_alpha # 设置负载均衡辅助损失（Auxiliary Loss）的权重因子 alpha
        self.seq_aux = config.seq_aux # 设置一个布尔值，用于决定辅助损失是否需要考虑序列维度来进行计算

        self.norm_topk_prob = config.norm_topk_prob # 设置一个布尔值，决定是否需要对选出的top_k个专家的权重进行二次归一化（确保它们的和为1）
        self.gating_dim = config.hidden_size # 设置门控线性层的输入维度
        # 可训练参数 torch.empty() 创建一个未初始化的张量 shape[n_routed_experts,gating_dim] 用于计算 “这个 token 应该去哪个专家”
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters() #初始化权重

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) #kaiming初始化 根号五是分布的方差

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h) #[bsz*seq_len, h]
        # 这个不是nn.Linear 但是差不多也是实现一个全连接层 会让hidden_states @ weight^T
        logits = F.linear(hidden_states, self.weight, None) # [(bsz * seq_len), n_routed_experts]    (bsz * seq_len)总token数
        if self.scoring_func == 'softmax': 
            scores = logits.softmax(dim=-1) #最后一维计算概率值
        else:
            # 抛出异常 raise
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # torch.topk() 用于找出张量中沿指定维度最大的k个元素及其索引
        # topk_weight 被选中的top_k个专家的 Softmax 概率 形状为[bsz*seq_len, top_k]
        # topk_idx 被选中的top_K个专家的原始索引 形状为[bsz*seq_len, top_k]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 选择了多个专家 且 模型配置明确要求对 Top-K 权重（概率值）进行二次归一化
        if self.top_k > 1 and self.norm_topk_prob: 
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20 # ：沿着最后一维求和 保持shape[bsz*seq_len, 1]
            topk_weight = topk_weight / denominator #  这一操作强制将每个 Token的top_k个权重之和调整为1
 
        # 在训练时计算 MoE 负载均衡辅助损失：统计每个专家的实际使用频率与 gate 概率分布，使专家使用更均匀
        if self.training and self.alpha > 0.0:  # L = main_loss + alpha * aux_loss alpha表示的是负载均衡辅助损失的惩罚力度 只要训练的时候算损失 因为推理的时候不需要反向传播了
            scores_for_aux = scores # [(bsz * seq_len), n_routed_experts]
            aux_topk = self.top_k 
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) #[bsz, seq_len*aux_topk] 
            # 判断配置是否要求使用序列级(样本级)辅助损失
            # 对整个序列（seq_len 个 token） 来统计每个专家的使用情况，然后计算辅助损失
            if self.seq_aux: 
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [bsz, seq_len, n_routed_experts]
                # 初始化 ce：统计每个 sample 中每个 expert 的使用次数 初始化为0
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device) #[bsz,n_routed_experts]
                # scatter_add_ (累加统计) 统计每个样本(batch)中每个专家的使用次数 原地修改ce张量
                # .div_(x) 元素都除以x
                ce.scatter_add_(
                    1,       #表示累加发生在第一个维度即专家维度
                    topk_idx_for_aux_loss, # index 指定加到ce的哪个列
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device) #要累加的值为1
                ).div_(
                    seq_len * aux_topk / self.n_routed_experts # 归一化：除以理想均匀使用次数
                ) # 表达 实际负载是理想负载的多少倍
                
                # scores_for_seq_aux.mean(dim=1) 得到每个专家的平均被选择的概率(意图）shape[bsz,n_routed_experts]
                # ce 实际专家利用频率（负载）
                # ce与scores_for_seq_aux.mean(dim=1)  逐元素相乘是为了惩罚那些同时具有高意图和高负载的专家 形状还是[bsz,n_routed_experts]
                # .sum(dim=1) 计算每个样本的总不均衡度
                #.mean() 对所有元素求平均（默认是对所有维度求平均） 这样就得到一个值而不是一个张量了
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 整个批次所有 Token 的总和/平均 上计算负载(实际选择)
                # F.one_hot(..., num_classes=n_routed_experts) → 变成 [bsz * seq_len * top_k, n_routed_experts] 的 被选中的专家的one-hot 张量
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 形状变为[n_routed_experts] 每个专家被选中的平均比例
                ce = mask_ce.float().mean(0) 
                # fi这个指标能够反映专家i的实际负载相对于理想负载的偏离程度
                # 理想负载是1/n_routed_experts
                # 这样相乘之后 表示专家i的负载是理想平均负载的多少倍 如果负载完美 fi会趋向于1 
                fi = ce * self.n_routed_experts #[n_routed_experts]
                
                # pi [n_routed_experts] 所有 Token 对每个专家i的 平均路由概率
                Pi = scores_for_aux.mean(0) # 希望选择每个专家的平均概率
                
                aux_loss = (Pi * fi).sum() * self.alpha #逐元素相乘 再求和 
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建 n_routed_experts 个 FeedForward 网络，并把它们存进 ModuleList 里
        # ModuleList：PyTorch 会自动把它们加入模型参数，能正常训练、保存、加载
        self.experts = nn.ModuleList([ 
            FeedForward(config)
            for _ in range(config.n_routed_experts) 
        ])
        # 路由
        self.gate = MoEGate(config) 
        # 创建一组共享专家 每个token都要过共享专家 不包括在topk选择中
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
            

    def forward(self, x):
        identity = x # 保留原始输入x的副本
        orig_shape = x.shape # 存储原始输入形状
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1]) # [bsz*seq_len, hidden_size] 即[token总数，hidden_size]
        flat_topk_idx = topk_idx.view(-1) #[bsz*seq_len*top_k]
        # 训练时
        if self.training:
            # 把每个 token 复制 top-k 次，以便送给它要路由的 top-k 个专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) #在第一维度token维度 复制top_k次 [bsz*seq_len*top_k, hiddensize]
            # 创建未初始化的y，用来存放每个 token 通过专家计算得到的输出 形状和x一样
            y = torch.empty_like(x, dtype=torch.float16) 
            # flat_topk_idx == i 布尔高级索引 变为一个有true false 矩阵
            # x[flat_topk_idx == i] 筛选并提取所有被路由到专家i的 Token 副本 并送入expert进行前向传播
            # y[flat_topk_idx == i] 将专家计算的输出，通过相同的布尔掩码，精确地写回输出y中对应的位置
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致 [bsz*seq_len*top_k, hiddensize]
            # y.view(*topk_weight.shape, -1) [bsz*seq_len, top_k, hidden_size] 星号将元组 (T, k) 解包成两个独立的参数
            # topk_weight.unsqueeze(-1) [bsz*seq_len, top_k, 1]
            # .sum(dim=1) 沿着专家维度进行聚合 [bsz*seq_len, hidden_size]
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape) #[bsz,seq_len,hidden_size]
        else: #推理时
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)#转为[bsz,seq_len,hidden_size]
        # 让每个 Token 经过这些共享专家 并联计算后累加
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss #负载均衡辅助损失
        return y #[bsz,seq_len,hidden_size]

    # x [bsz*seq_len, hidden_size]
    # flat_expert_indices 扁平化后的专家index [bsz*seq_len*top_k]
    # flat_expert_weights 扁平化后的专家权重 [bsz*seq_len*top_k, 1]
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) # [bsz*seq_len, hidden_size]
        # 获取一个索引序列，每个元素是flat_expert_indices列表的下标，这个序列能按照专家 ID 的大小顺序进行重新排列
        idxs = flat_expert_indices.argsort() # [bsz*seq_len*top_k]
        # 确定每个专家负载边界
        # flat_expert_indices.bincount() 统计输入张量中每个非负整数值（即每个专家 ID）出现的次数 输出 [n_routed_experts]
        # .cpu().numpy() 将 PyTorch 张量转换到 CPU，并转换为 NumPy 数组
        # .cumsum(0) 计算前缀和 用来确定每个专家组在排序后的 Token 列表中的起始和终止边界
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # [n_routed_experts]
        # 排序后的每个 token 副本，来自原序列中的哪个 token 因为idxs里面的每个元素 其实是flat_expert_indices列表的下标 所以要除以专家总数
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert): # i代表专家 end_idx是token_per_expert的元素 表示结束边界
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1] #因为下标从0开始 所以 tokens_per_expert[i - 1] 表示下一个的开始index
            if start_idx == end_idx: #跳过没被选择的专家
                continue
            expert = self.experts[i]
            # 切出当前专家对应的所有token的下标 左闭右开
            exp_token_idx = token_idxs[start_idx:end_idx]
            # #高级索引切出当前专家所需的 Token 输入
            expert_tokens = x[exp_token_idx] 
            # 进入专家进行前向传播 输出[专家i处理的所有token,hidden_size]
            expert_out = expert(expert_tokens).to(expert_cache.dtype) 
            # idxs[start_idx:end_idx] 切出bsz*seq_len*top_k这个范围内的下标
            # flat_expert_weights[idxs[start_idx:end_idx]] 得到权重 [专家i处理的所有token,1]
            # .mul_ 执行逐元素乘法
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 把当前 expert 产生的输出，加回到它处理的 token 的原位置上（累加），恢复成原来的 token 序列顺序。
            expert_cache.scatter_add_(
                0, # 按第 0 维（即 token 维）进行累加
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), # index  把形状变为[专家i处理的所有token,hidden_size] scatter_add_的要求
                expert_out
            )

        return expert_cache # [bsz*seq_len, hidden_size]

# 把GQA和ffn拼接在一起
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)#初始化attention

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config) #多层感知机 提供非线性、扩展表达能力

    # 传入的参数position_embeddings 是cos/sin对 形状是[seq, dim]  attention_mask是padding mask[bsz,seq]
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states # 残差 
        # Attention 之前先做一次 RMSNorm
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, 
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual # 做 Attention 的残差连接
        # 对输入在做一次RMSNorm 然后又做了残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value #这里我也有一个小疑问 这个block要重复k次 推理的时候怎么办 这里的present_key_value按理说只用算第一个block? 疑问解答了 每一个block都需要一个kvcache 且不相同


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # num_hidden_layers 代表的是minimindBlock的块数
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 将输入的token id 转化为embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 创建一个包含 num_hidden_layers 个  minimindBlock 的列表
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 预计算Rope参数 计算cos sin频率 形状为[max_position_embeddings, head_dim]
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads, #每个注意力头的维度 head_dim
            end=config.max_position_embeddings, #模型支持的最大序列长度（上下文长度）
            rope_base=config.rope_theta, #RoPE 的 “频率衰减基数” 1000000
            rope_scaling=config.rope_scaling #用于yarn的参数
        ) 
        # 注册为缓存
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, #模型的输入 token_id [bsz, seq_length]
                attention_mask: Optional[torch.Tensor] = None, #padding mask [bsz, seq_length]
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, #列表 缓存每层 minimindBlock 的历史 (key,value)元组
                use_cache: bool = False,
                **kwargs):  # **kwargs 可选额外参数
        batch_size, seq_length = input_ids.shape
        
        # 判断对象是否有名为 'layers' 的属性 如果past_key_values是某种封装对象而不是一个列表，且包含layers属性，置为None  为了兼容 Hugging Face 的格式
        if hasattr(past_key_values, 'layers'): past_key_values = None
        
        # 如果 past_key_values 已经有值（非 None），就沿用原来的值  Python 中，or 不一定返回 True 或 False，它返回 第一个“真值”
        # 如果 past_key_values 是 None，就创建一个长度等于 self.layers 的 [None, None, ...] 
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # past_key_values[0][0] 是第 0 层的 key 张量，形状 [bsz, seq_len, n_local_heads, head_dim]
        # .shape[1] 取的是 seq_len，也就是已经缓存的序列长度
        # 如果没有缓存（训练或第一次推理），就用 0
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 把 token id 转成向量表示，并对向量做 dropout 形状为[bsz, seq_len, hidden_size]
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 切片操作选取当前输入序列对应的 RoPE 参数
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length], #[seq_length, head_dim] 用于单步推理 也兼容训练
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = [] #记录每一层的kv元组
        # zip(self.layers, past_key_values) 每一层minimindBlock和它对应的 past KV 被打包成一个二元组
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states, #输入[bsz, seq_len, hidden_size]
                position_embeddings, # cos/sin
                past_key_value=past_key_value, #每一层对应的past kv
                use_cache=use_cache,
                attention_mask=attention_mask #padding mask [bsz, seq_len]
            )
            presents.append(present)# 把当前层产生的 KV cache（present）收集起来，存进列表 presents 中，供下一次推理使用  每一层产生的kvcache都是不一样的 输入不一样 权重Wk Wv也不一样

        # 最后经过一个RMSNorm
        hidden_states = self.norm(hidden_states)

        #把每一个moe层的负载均衡辅助损失加起来
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward) #判断layer.mlp 这个对象的类型是不是 MOEFeedForward 类
        )

        return hidden_states, presents, aux_loss

# 它继承 PreTrainedModel 和 GenerationMixin是为了让模型具备 HuggingFace 标准语言模型的全部能力
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin): 
    config_class = MiniMindConfig # 告诉 PreTrainedModel 模型用哪个 Config

    def __init__(self, config: MiniMindConfig = None):
        
        # 如果用户提供了 config 就用它，否则创建一个默认的 MiniMindConfig()
        self.config = config or MiniMindConfig()
        # 调用父类 PreTrainedModel 的构造函数，把配置传给父类 这两句不能写反
        super().__init__(self.config)
        
        self.model = MiniMindModel(self.config)
        # 初始化语言模型的输出头
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 通过赋值把 lm_head.weight 和 embed_tokens.weight 指向同一块内存
        self.model.embed_tokens.weight = self.lm_head.weight # 共享词向量权重 训练时，这两个权重会同步更新，节省参数
        # # CausalLMOutputWithPast 是一个 封装了模型全部关键输出的标准化数据结构
        # self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None, #模型的输入 token_id [bsz, seq_length]
                attention_mask: Optional[torch.Tensor] = None, #padding mask [bsz, seq_length]
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, #列表 缓存每层 minimindBlock 的历史 (key,value)元组
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,  #union的意思是 类型可以是int 也可以是tensor  默认值为0
                **args):
        # 把输入数据送入模型进行一次前向计算，然后得到三个输出
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # if isinstance(logits_to_keep, int) 检查 logits_to_keep 是否是整数 如果是整数，就意味着用户希望“取最后 logits_to_keep 个 token”。
        # 如果是整数 执行 slice(-logits_to_keep, None) 是切片语法 -logits_to_keep 表示从倒数第logits_to_keep个token开始  None：切到最后一个 token
        # 如果不是整数 是张量 直接使用它
        # 这里如果logits_to_keep是0 就会保留所有的seq
        # 这个切片兼容单步推理（只输入最后一个token） 也兼容全量推理（首次输入 Prompt 或不使用 KV Cache 进行长序列推理时） 还可以设置为0用于训练
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        # hidden_states 的形状是 [batch_size, seq_length, hidden_size] 这里是对seq_length维度切片 
        # 然后经过lm_head
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 创建一个标准化的输出对象 这是 HuggingFace 的标准输出结构
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
