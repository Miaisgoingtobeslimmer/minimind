import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed
warnings.filterwarnings('ignore')

def init_model(args):
    # 当 args.load_from 的值是 'model' 时，AutoTokenizer.from_pretrained('model') 会去项目根目录下的 model/ 文件夹 中寻找分词所需的所有组件
    # 当 args.load_from 的值是一个 HuggingFace 模型名 会从hf上下载
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from: # 用 MiniMind 自己写的模型类
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling # 推理的时候 才可能会开启? bool
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth' #/out文件夹下
        model.load_state_dict( #把权重加载到模型
            torch.load(ckp, map_location=args.device), # 是从ckp文件里面加载权重字典到device上
            strict=True # 要求state_dict 和 model 的参数集合必须完全一致
        )
        if args.lora_weight != 'None':
            apply_lora(model) # 改模型参数 来自model.model_lora.py
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth') # 只加载 LoRA 参数
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True) #加载HF上一个已经打包好的因果语言模型
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)') # p.numel()计算这个参数张量里面的元素总数
    return model.eval().to(args.device), tokenizer

def main():
    # 创建一个命令行参数解析器（parser）
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    # 给训练脚本添加所有可配置参数 会自动检查类型 如果命令行里没有输入某个参数 → 就使用 default 值
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    # 如果在命令行里写了 --inference_rope_scaling，它就会被自动设为 True
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    # 从命令行中解析用户输入的参数，并把结果存到 args 里 如果用户没有输入某个参数，就会使用默认值
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n')) # 让用户在终端选择推理的模式
    # TextStreamer 模型每生成一个 token，就立刻打印出来，而不是等全部生成完
    # skip_prompt=True 生成时跳过用户输入的prompts, skip_special_tokens=True跳过特殊token
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '') # iter(函数, 终止值) 不断调用这个函数，把返回值作为迭代元素
    for prompt in prompt_iter:
        setup_seed(2026) # or setup_seed(random.randint(0, 2048)) 设置种子后，在参数（温度、Top-p）不变的情况下，模型的回复将永远保持一致 方便做benchmark
        if input_mode == 0: print(f'👶: {prompt}')# 自动测试模式 要打印prompt
        conversation = conversation[-args.historys:] if args.historys else [] #对话历史裁剪器 只会切下最后args.historys个元素
        conversation.append({"role": "user", "content": prompt}) # 存入这一次的对话 字典的格式 角色是user

        # 字典 为tokenizer.apply_chat_template 准备参数  "tokenize": False 告诉分词器，暂时不要把它转成数字（Token IDs）
        # "add_generation_prompt": True 是否添加“诱导模型生成”的后缀 模板会在字符串最后强行加上一段代表 AI 开始说话的记号
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True} 
        if args.weight == 'reason': templates["enable_thinking"] = True # 仅Reason模型使用
        
        # 构建输入文本 非 pretrain情况 使用 apply_chat_template。这会将对话历史包装成带有 <|user|> 和 <|assistant|> 等标记的特殊格式的字符串
        # pretrain 模型 输入是起始符+提示词的字符串 预训练模型没有学过对话格式
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        # 分词与张量化 inputs 得到的输出是字典 inputs["input_ids"]、inputs["attention_mask"] 张量的维度[batchsize = 1, seq_len]
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖️: ', end='') # end='' 指打印完之后，不换行，不追加任何字符
        # 因为模型继承了PreTrainedModel, GenerationMixin 而且它的返回值是CausalLMOutputWithPast() 所以可以用generate方法
        # generate() 是一个“自动循环解码器”：它反复调用模型的 forward()，一步一步生成 token，直到满足停止条件
        # generated_ids [batch_size, prompt_len + generated_len]
        generated_ids = model.generate(
            inputs=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], #padding mask
            max_new_tokens=args.max_new_tokens, # 最大新生成长度
            do_sample=True, # 随机采样的总开关
            streamer=streamer, # 边生成边输出 负责打印生成的字符串
            pad_token_id=tokenizer.pad_token_id, # 用于batch 推理
            eos_token_id=tokenizer.eos_token_id, # 一旦生成这个 token，就立刻停止生成
            top_p=args.top_p,  # 只在“累计概率 ≥ top_p 的最小 token 集合”中采样
            temperature=args.temperature, # 调节概率分布的“软硬程度”
            repetition_penalty=1.0 # 对已经生成过的 token 进行惩罚
        )
        # 把模型“新生成的 token”解码成字符串，作为 assistant 的回答文本
        # tokenizer.decode(...) = 把「token ID 序列」还原成「人类可读的字符串」并且去除特殊符号
        # generated_ids[0][len(inputs["input_ids"][0]):] 因为batch=1 所以这里是在取出“新生成的 token”
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print('\n\n')

if __name__ == "__main__":
    main()