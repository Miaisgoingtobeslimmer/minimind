"""
训练工具函数集合
"""
import gc
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))

# 初始化 PyTorch 的分布式数据并行（DDP）：每张 GPU 拿到不同的小批次数据，计算梯度后进行同步
def init_distributed_mode():
    # 从环境变量里获取 "RANK" 如果获取不到（说明不是 torchrun 启动），就返回默认值 -1
    # 在分布式训练中，Rank = 当前进程的编号
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    # 每个进程都是独立运行同一份 train.py 文件 但每个进程看到的 LOCAL_RANK 不一样 代码里面不需要做循环
    dist.init_process_group(backend="nccl") # 让所有进程参与一个“通信组”
    local_rank = int(os.environ["LOCAL_RANK"]) # 读出当前进程属于本地机器上的哪个 GPU local_rank是本地机器上的排名 rank是所有进程的排名
    torch.cuda.set_device(local_rank) #设置当前进程默认使用的 GPU 设备
    return local_rank


def setup_seed(seed: int): # 种子（seed）只是随机数生成器的“起点”
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # 设置当前默认 CUDA 设备的随机种子
    torch.cuda.manual_seed_all(seed) # 设置该进程中所有 CUDA 设备的随机种子
    torch.backends.cudnn.deterministic = True # 告诉 CuDNN：所有算子必须使用「可复现算法
    torch.backends.cudnn.benchmark = False # 不要自动寻找最快的算法，使用固定算法（保证可复现）


# 这个函数通过判断是否传入了 model 对象来切换工作模式：
# 如果 model 不为 None: 执行 保存模式（保存当前训练状态）。
# 如果 model 为 None: 执行 加载模式（查找并返回上次的训练状态）。
# 参数weight用于保存文件名
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True) # 创建一个保存模型的文件夹
    moe_path = '_moe' if lm_config.use_moe else '' #根据模型是否启用 MoE 来决定文件名中要不要加 "_moe"
    # 构造“模型权重文件”的路径 路径下文件仅包含模型参数
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth' 
    # 构造“训练恢复用的 checkpoint 的路径”  包含模型 + 优化器 + 训练进度   可以从断点继续训练
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    # 保存模型和训练状态，生成两个文件
    if model is not None: 
        from torch.nn.parallel import DistributedDataParallel
        # 如果模型是 DDP 包装过的，需要取 model.module 才是真正模型
        # state_dict() 返回 模型权重字典  字典的Key是模型中各层和子模块的参数名称，value是张量
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        # 把模型的每个权重张量转换为半精度（fp16）并移动到 CPU
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        # 先把模型权重保存到临时文件 .tmp，保存完成后再原子替换成正式文件 .pth，保证写入过程中不会损坏原文件。
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        # 用于在保存训练检查点时，尝试获取当前 WandB运行的唯一ID (wandb_id)，以便将来进行断点续训时，可以将新的训练日志 继续追加到同一个WandB实验记录上
        wandb_id = None
        if wandb:
            # 判断 wandb 对象是否有 get_run() 这个方法
            if hasattr(wandb, 'get_run'): 
                run = wandb.get_run() # 调用 get_run()，返回当前训练 run 对象
                # 如果 run 存在，就尝试取 run.id，这是这个训练 run 的唯一标识
                # 如果 run 是 None 或没有 id，就把 wandb_id 设置为 None
                wandb_id = getattr(run, 'id', None) if run else None 
            else:   
                # else: 处理老版本或者没有 get_run() 方法的 wandb 对象
                # 直接从 wandb 对象获取 run 的 id，如果没有就设为 None
                wandb_id = getattr(wandb, 'id', None) 

        resume_data = {
            'model': state_dict, #模型权重
            'optimizer': optimizer.state_dict(), # 保存优化器状态（如动量、学习率等）
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1, # 记录检查点保存时使用的 GPU 数量（总进程数）。
            'wandb_id': wandb_id  # 当前 WandB运行的唯一ID
        }
        
        # 将额外传入的模块或对象也保存到 resume_data 中  kwargs: 这是一个 Python 字典
        for key, value in kwargs.items():  
            if value is not None:
                # 判断这个对象是否有 state_dict() 方法
                if hasattr(value, 'state_dict'):
                    # 如果对象是 DDP 包装的模型，需要取 module.state_dict()
                    if isinstance(value, DistributedDataParallel): 
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict() # 否则直接用 value.state_dict()
                else: 
                    # 如果对象没有 state_dict()方法，直接把对象本身保存
                    resume_data[key] = value
                    
        # 先把训练状态保存到临时文件 并原子替换为正式文件 resume_path
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        
        del state_dict, resume_data # 删除 Python 中的对象引用 释放 CPU 内存
        gc.collect() # 强制垃圾回收 只回收 CPU 内存
        torch.cuda.empty_cache() # 清理 GPU 上的缓存显存
        
    else:  # 加载模式 未传入 model 参数
        # 检查续训文件是否存在
        if os.path.exists(resume_path): 
            # 用 torch.load() 读取 .pth 文件 把所有张量加载到 CPU
            ckp_data = torch.load(resume_path, map_location='cpu')
            # 从加载的ckp_data中获取保存时所用的总进程数/总GPU数量 没有默认为1
            saved_ws = ckp_data.get('world_size', 1)
            # 代表本次恢复训练时，正在使用多少个 GPU 进程
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            # 如果当前 GPU 数量和保存时不同，需要 按比例调整 step 会影响学习率lr
            # 因为分布式训练中每个 step 处理的总数据量和 GPU 数量有关，如果 GPU 数量变了，每 step 处理的总数据不同，为了保证继续训练的总训练量和之前一致
            # 如果 GPU 数量增加一倍，全局 Batch Size 也增加一倍。这意味着：达到相同的训练进度（即处理相同的样本总量）所需的 Step 数 只有原来的一半。
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        # 这个返回值在主脚本中会被检查：如果 ckp_data 是一个字典，则执行续训；如果是 None，则从头开始训练
        return None

#from_weight 基于哪个权重训练，比如基于预训练模型进行微调，为'none'则从头开始
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    # 从 tokenizer_path 路径读取 tokenizer 文件
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 初始化模型结构 权重是随机的
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        # 构建出 只包含模型权重的文件路径
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        # 直接把权重加载到指定设备（GPU/CPU）
        weights = torch.load(weight_path, map_location=device)
        # 把权重 load 到模型里 strict=False 如果加载的权重字典中的键与模型中的参数键 不完全匹配也不抛出错误
        model.load_state_dict(weights, strict=False)

    # p.numel(): 返回张量 p 中元素的总数量 p.requires_grad=True的参数是可训练的
    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model.to(device), tokenizer # 模型被移动到指定设备上


class SkipBatchSampler(Sampler): #sampler 是索引生成器
    def __init__(self, sampler, batch_size, skip_batches=0):
        # 这里的sampler是数据集的 index 流 传入参数的时候 要分为分布式的和不是分布式的情况
        self.sampler = sampler 
        self.batch_size = batch_size
        self.skip_batches = skip_batches # 需要丢弃掉的 batch 数量 其实也是续训开始的batch

    # 把 sampler 产生的 index 流，切成 batch，并在 batch 级别跳过前 skip_batches 个
    # DataLoader 迭代时 自动调用SkipBatchSampler.__iter__ 每次拿到一个batch的index列表 用这些index去dataset取数据
    def __iter__(self):
        batch = []
        skipped = 0 # 已经丢弃了多少个 完整 batch
        # 从 sampler 拿 index 每次只来一个样本 index
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size: # 如果刚好拼成了一个完整 batch
                if skipped < self.skip_batches: # 如果没有丢弃够
                    skipped += 1
                    batch = []
                    continue # 直接进入下一次for
                
                # 把当前这个 batch 作为一次“迭代结果”交给外层（DataLoader），然后函数在这里“暂停”，下次再从这里继续往下跑。
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch # 处理最后一个不满 batch_size 的残余数据

    # 这个 sampler 实际上还能产出多少个 batch
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size # 加 self.batch_size - 1 是为了向上取整
        # 真实能 yield 的 batch 数 = 总 batch 数 − 跳过的 batch 数
        return max(0, total_batches - self.skip_batches) 

