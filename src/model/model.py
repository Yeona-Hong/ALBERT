# src/model/modeling.py
from transformers import AlbertConfig, AlbertForPreTraining, AlbertTokenizerFast
from src.utils.utils import get_timestamp

def create_model_and_tokenizer(args):
    """모델 및 토크나이저 생성"""
    # 토크나이저 로드
    print(f"[{get_timestamp()}] 토크나이저 로드 중: {args.tokenizer_path}")
    tokenizer = AlbertTokenizerFast.from_pretrained(args.tokenizer_path)
    
    # vocab_size가 지정되지 않았으면 토크나이저에서 가져옴
    if args.vocab_size is None:
        args.vocab_size = tokenizer.vocab_size
        print(f"[{get_timestamp()}] 토크나이저에서 vocab_size={args.vocab_size}를 자동으로 가져왔습니다.")
    
    # 모델 구성 설정
    config = AlbertConfig(
        vocab_size=args.vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=args.max_seq_length,
        type_vocab_size=2,
    )
    
    # 모델 초기화
    print(f"[{get_timestamp()}] ALBERT 모델 초기화 중...")
    model = AlbertForPreTraining(config)
    model.to(args.device)
    
    print(f"[{get_timestamp()}] 모델 생성 완료. 파라미터 수: {sum(p.numel() for p in model.parameters())}")
    return model, tokenizer, config

def save_checkpoint(model, tokenizer, optimizer, scheduler, config, args, global_step, checkpoint_dir=None):
    """모델 체크포인트 저장"""
    import os
    from src.utils.utils import save_config
    
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 모델 저장
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # 학습 상태 저장
    import torch
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
        }, 
        os.path.join(checkpoint_dir, "optimizer.pt")
    )
    
    # 현재 설정 저장
    config_to_save = {
        'dataset': {k: v for k, v in vars(args).items() if k in ['dataset_name', 'preprocessing_num_workers', 'overwrite_cache', 'max_seq_length']},
        'model': {k: v for k, v in vars(args).items() if k in ['tokenizer_path', 'embedding_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'vocab_size']},
        'training': {k: v for k, v in vars(args).items() if k in ['output_dir', 'per_device_train_batch_size', 'per_device_eval_batch_size', 'learning_rate', 'weight_decay', 'max_steps', 'num_train_epochs', 'warmup_steps', 'save_steps', 'logging_steps', 'eval_steps', 'gradient_accumulation_steps', 'save_total_limit', 'seed', 'fp16', 'local_rank']},
    }
    save_config(config_to_save, checkpoint_dir)
    
    print(f"[{get_timestamp()}] 모델 체크포인트 저장: {checkpoint_dir}")
