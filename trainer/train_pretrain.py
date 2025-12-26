import os
import sys

__package__ = "trainer" # 当前文件属于 trainer 这个包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 把“项目根目录”加入 Python 的模块搜索路径 sys.path

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None): #iters 指 DataLoader 可以迭代多少次 我觉得这里有点问题 默认start_start应该是-1
    # 包含了 Softmax 运算 reduction='none' 表示 交叉熵是“按 token 算的 
    loss_fct = nn.CrossEntropyLoss(reduction='none') 
    # 记录“当前时刻”的时间戳
    start_time = time.time()
    
    # 从 DataLoader 中一次拿一个 batch，给它编号为 step，从start开始, 然后把 batch 拆成 X、Y、loss_mask。
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 把 batch 里的 tensor 从 CPU 复制到 GPU
        X = X.to(args.device) # [batch_size, seq_len]
        Y = Y.to(args.device) # [batch_size, seq_len]
        loss_mask = loss_mask.to(args.device) # [batch_size, seq_len]
        # 保证 lr 随训练进度动态变化
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器不同参数组的学习率 本项目是单参数组
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 在这个 with 块内部 会启用自动混合精度上下文管理器
        with autocast_ctx:
            res = model(X) #前向传播 把输入 X 送进模型，得到输出 res 我觉得这里也有问题 应该把padding mask传进来
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)), # 输出的logits [batch_size*seq_len, vocab_size]
                Y.view(-1) # tag [batch_size*seq_len]
            ).view(Y.size()) # 输出 shape [batch_size*seq_len]，每个 token 一个 loss 最后reshape 回 [batch_size, seq_len]

            loss = (loss * loss_mask).sum() / loss_mask.sum() # (loss * loss_mask) 逐元素相乘 按loss_mask屏蔽padding 最终得到平均损失(常数)
            loss += res.aux_loss #加上辅助损失
            # 梯度积累 缩放 loss
            loss = loss / args.accumulation_steps
        
        # scaler梯度缩放器 当FP16时 .scale(loss)会放大loss 防止梯度太小下溢
        scaler.scale(loss).backward() # .backward() → 计算梯度，累加到模型参数的 .grad  这是梯度积累

        # 判断是否更新模型参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer) # 取消梯度缩放 将梯度恢复到其真实的数值范围
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪 args.grad_clip是梯度范数上限

            scaler.step(optimizer) # scaler.step() 检查梯度是否有溢出 如果没有 就会更新模型参数 如果溢出 就不更新防止破坏模型
            scaler.update() # 作用是调整下次梯度缩放时 把损失扩大的倍数
            optimizer.zero_grad(set_to_none=True) # 清空梯度 只有更新参数 才清空  set_to_none=True直接把 .grad 设置为 None
            torch.cuda.empty_cache() # 会清空缓存池中未使用的显存，释放给操作系统

        # 每隔log_interval 个step打一次日志 或者 当前step是该epoch的最后一个batch
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time # 计算 从 epoch 开始到当前 step 已经消耗的时间（秒）
            current_loss = loss.item() * args.accumulation_steps # 还原梯度积累前的实际loss
            current_lr = optimizer.param_groups[-1]['lr'] # 获取当前使用的学习率 param_groups[-1]是因为本项目的optimizer只有一个参数组 取最后一个
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60 # 估算当前epoch剩余时间
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}) # 如果使用wandb 记录训练指标

        # 训练过程中保存模型检查点 会覆盖原来的文件
        # 每隔save_interval step保存一次模型 或者当前step是该epoch的最后一步 且 只有主进程保存模型
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval() # 切换模型到评估模式
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth' # 构造模型保存路径 这里是保存在../out文件夹下
            if isinstance(model, torch.nn.parallel.DistributedDataParallel): #如果是分布式训练 
                state_dict = model.module.state_dict() # 模型参数
            else:
                state_dict = model.state_dict() 
            
            # 把模型参数字典 模型参数转换成半精度 放到cpu上
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp) # 将处理后的权重字典保存到/out文件夹下
            
            # 保存检查点文件 保存在../checkpoints文件夹下
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train() #转换回训练模型
            del state_dict

        del X, Y, loss_mask, res, loss # del是删除对象引用 此时会将这些大块内存区域标记为空闲，并将它们归入其内部的缓存池中


if __name__ == "__main__":
    # 创建一个命令行参数解析器（parser）
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 给训练脚本添加所有可配置参数 会自动检查类型 如果命令行里没有输入某个参数 → 就使用 default 值
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量") #这个是minimindBlock块数
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始") #字符串 是文件名字
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # 这是一个布尔（Boolean）标记。 如果用户在命令行中输入 --use_wandb，则 args.use_wandb 的值为 True；如果用户不输入，则其默认值为 False。
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    # 从命令行中解析用户输入的参数，并把结果存到 args 里 如果用户没有输入某个参数，就会使用默认值
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    
    # 判断是不是分布式训练，初始化分布式通信，并返回该进程应使用的 GPU 编号
    local_rank = init_distributed_mode() 
    # 如果当前是分布式环境（DDP） → 就把训练设备 args.device 设成对应的 local_rank GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0)) # 给每个 GPU（每个 rank）分配一个不同但可复现的随机种子
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    
    # 创建一个用于保存模型的目录（save_dir），不存在就创建，存在就忽略
    os.makedirs(args.save_dir, exist_ok=True) 
    # 配置模型架构参数
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 这行代码根据是否断点续训，加载之前的训练状态ckp_data，否则置为 None，从头训练
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 根据命令行参数 args.dtype 来确定用于混合精度计算的低精度类型
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # GPU → 用混合精度  autocast 会自动决定哪些算子适合半精度，哪些必须保持 FP32
    # CPU → 不用混合精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    
    wandb = None
    # 在用户允许 并且是主进程时rank=0 才会使用 WandB
    if args.use_wandb and is_main_process():
        # 用国产的swanlab 命名为wanb
        import swanlab as wandb
        # ckp_data (之前加载的检查点数据) 存在，就从中尝试获取上次保存的 实验 Run ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 如果找到了上次的 wandb_id，则设置 resume 参数为 'must' 强制从这个 wandb 任务恢复
        resume = 'must' if wandb_id else None
        # 定义 wanb run 名字
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        # 启动一次新的 W&B 实验（run），或恢复之前中断的实验
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device) # 加载模型和 tokenizer
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len) # 数据集初始化
    # 只有在分布式模式下才使用 DistributedSampler 来正确划分数据；单卡训练则不用
    # 在分布式训练时，把“全量数据集”逻辑上切分成 world_size 份，每个进程（rank）只“看到并遍历”其中一份, 每个 rank 拿到数据集的不同的 index 子序列。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 当你用 FP16 训练时，就启用梯度缩放（GradScaler）来防止梯度下溢（数值精度不够导致梯度变成 0）；否则不启用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 创建了一个 AdamW 优化器 这的lr是初始的学习率
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) 
    
    # ========== 6. 从ckp恢复状态 ==========
    
    # 只有在用户指定续训时候执行 根据之前加载的检查点数据 (ckp_data)，恢复模型、优化器、梯度缩放器以及训练进度
    start_epoch, start_step = 0, 0
    if ckp_data: #ckp_data是一个字典
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer']) # 恢复优化器状态
        scaler.load_state_dict(ckp_data['scaler']) # 保存续训文件的时候 似乎没有保存这个scaler
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    
    if dist.is_initialized(): #分布式训练
        # 在构建要广播/打包/同步的 parameters/buffers 列表时，把名为 freqs_cos 和 freqs_sin 的属性排除掉 因为太大了
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 把模型用 PyTorch 的 DistributedDataParallel（DDP）进行包装，使其能在多 GPU 的分布式训练环境中同步梯度
        model = DistributedDataParallel(model, device_ids=[local_rank]) # local_rank 当前节点（机器）上的 GPU 索引
    
    # ========== 8. 开始训练 ==========
    
    # 按epoch训练 支持从中间 epoch 继续训练
    for epoch in range(start_epoch, args.epochs):
        # 如果是分布式训练 让每个 epoch 的数据打乱顺序都不一样，但所有进程保持一致的打乱规则,然后再结合 每个进程的rank 做数据集的切分
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0: # 断点续训逻辑 (跳过已完成的 Step) 在恢复训练的第一个epoch且存在检查点
            # 跳过已经训练过的 batch，只从中断的 step 继续训练
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), 
                args.batch_size, 
                start_step + 1 #要跳过的batch数 我觉得 这里指的是step从0开始?
            )
            # Loader 在迭代时，返回的是“已经打包好的一个 batch”
            loader = DataLoader(
                train_ds, #dataset
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers,  # 开启多进程数据加载
                pin_memory=True # 当设置为True且在GPU上训练时，DataLoader会将数据加载到特殊的固定内存区域 从固定内存到GPU显存的数据传输速度比从普通CPU内存传输要快得多
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else: # 默认从头开始
            loader = DataLoader(
                train_ds, #dataset
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), #当不是分布式训练的时候 在每次epoch打乱数据集
                sampler=train_sampler, #当是分布式训练 开启
                num_workers=args.num_workers, 
                pin_memory=True
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
