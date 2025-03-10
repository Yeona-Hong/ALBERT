# src/data/dataset.py
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from transformers import DataCollatorForLanguageModeling
from src.utils.utils import get_timestamp

def tokenize_function(examples, tokenizer, max_seq_length):
    """텍스트 데이터를 토큰화하는 함수"""
    return tokenizer(
        examples["text"],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_special_tokens_mask=True,
    )

def prepare_datasets(tokenizer, args):
    """데이터셋 로드 및 전처리"""
    # 데이터셋 로드
    print(f"[{get_timestamp()}] 데이터셋 '{args.dataset_name}' 로드 중...")
    datasets = load_dataset(args.dataset_name)
    
    # 토큰화 함수 래핑
    def tokenize_wrapper(examples):
        return tokenize_function(examples, tokenizer, args.max_seq_length)
    
    # 데이터셋 토큰화
    print(f"[{get_timestamp()}] 데이터셋 토큰화 시작...")
    tokenized_datasets = datasets.map(
        tokenize_wrapper,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=["text"],
        load_from_cache_file=not args.overwrite_cache,
        desc="데이터셋 토큰화 중",
    )
    
    print(f"[{get_timestamp()}] 데이터셋 토큰화 완료. 크기: {len(tokenized_datasets['train'])} 샘플")
    return tokenized_datasets

def create_dataloaders(tokenized_datasets, tokenizer, args):
    """데이터로더 생성"""
    # 데이터 콜레이터 설정 - MLM 작업
    print(f"[{get_timestamp()}] 데이터 콜레이터 설정 (MLM probability=0.15)")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # 학습 데이터로더 설정
    train_dataset = tokenized_datasets["train"]
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        print(f"[{get_timestamp()}] RandomSampler 설정")
    else:
        train_sampler = DistributedSampler(train_dataset)
        print(f"[{get_timestamp()}] DistributedSampler 설정")
    
    print(f"[{get_timestamp()}] 학습 데이터로더 생성 (배치 크기={args.per_device_train_batch_size})")
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    
    # 평가 데이터로더 설정 (있는 경우)
    eval_dataloader = None
    if 'validation' in tokenized_datasets:
        print(f"[{get_timestamp()}] 검증 데이터셋 발견. 평가 데이터로더 생성...")
        eval_dataset = tokenized_datasets["validation"]
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )
    
    return train_dataloader, eval_dataloader
