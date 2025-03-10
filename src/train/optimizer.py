# src/training/optimizer.py
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.utils.utils import get_timestamp

def create_optimizer_and_scheduler(model, args, num_training_steps):
    """옵티마이저 및 학습률 스케줄러 생성"""
    # 가중치 감쇠를 적용하지 않을 파라미터 지정
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # AdamW 옵티마이저 설정
    print(f"[{get_timestamp()}] AdamW 옵티마이저 생성 (lr={args.learning_rate})")
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6,
        correct_bias=False  # ALBERT에서는 bias correction 사용 안함
    )
    
    # 선형 스케줄러 설정
    print(f"[{get_timestamp()}] 학습률 스케줄러 생성 (warmup_steps={args.warmup_steps})")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler

def setup_fp16_and_distributed(model, optimizer, args):
    """혼합 정밀도 및 분산 학습 설정"""
    # FP16 혼합 정밀도 설정
    if args.fp16:
        try:
            from apex import amp
            print(f"[{get_timestamp()}] FP16 혼합 정밀도 학습 설정 중...")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        except ImportError:
            print(f"[{get_timestamp()}] 경고: apex가 설치되어 있지 않습니다. 일반 정밀도로 학습합니다.")
    
    # 분산 학습 설정
    if args.local_rank != -1:
        import torch.distributed as dist
        print(f"[{get_timestamp()}] 분산 학습 설정 중 (local_rank={args.local_rank})...")
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank, 
            find_unused_parameters=True
        )
    
    return model, optimizer
