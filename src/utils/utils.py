# src/utils/utils.py
import os
import yaml
import json
import datetime
import torch
import numpy as np
import shutil
from pathlib import Path

def get_timestamp():
    """현재 시간 포맷팅"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, output_dir):
    """설정 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정을 YAML 형식으로 저장
    with open(output_dir / "config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # JSON 형식으로도 저장 (호환성 위해)
    with open(output_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"[{get_timestamp()}] 설정이 {output_dir}에 저장되었습니다.")

def config_to_args(config):
    """설정을 argparse 형식으로 변환"""
    from argparse import Namespace
    args = Namespace()
    
    # 데이터셋 설정
    args.dataset_name = config['dataset']['name']
    args.preprocessing_num_workers = config['dataset']['preprocessing_num_workers']
    args.overwrite_cache = config['dataset']['overwrite_cache']
    args.max_seq_length = config['dataset']['max_seq_length']
    
    # 모델 설정
    args.tokenizer_path = config['model']['tokenizer_path']
    args.embedding_size = config['model']['embedding_size']
    args.hidden_size = config['model']['hidden_size']
    args.num_hidden_layers = config['model']['num_hidden_layers']
    args.num_attention_heads = config['model']['num_attention_heads']
    args.intermediate_size = config['model']['intermediate_size']
    args.vocab_size = config['model'].get('vocab_size', None)  # 선택적 (자동 계산 가능)
    
    # 학습 설정
    args.output_dir = config['training']['output_dir']
    args.per_device_train_batch_size = config['training']['per_device_train_batch_size']
    args.per_device_eval_batch_size = config['training']['per_device_eval_batch_size']
    args.learning_rate = config['training']['learning_rate']
    args.weight_decay = config['training']['weight_decay']
    args.max_steps = config['training']['max_steps']
    args.num_train_epochs = config['training']['num_train_epochs']
    args.warmup_steps = config['training']['warmup_steps']
    args.save_steps = config['training']['save_steps']
    args.logging_steps = config['training']['logging_steps']
    args.eval_steps = config['training']['eval_steps']
    args.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    args.save_total_limit = config['training']['save_total_limit']
    args.seed = config['training']['seed']
    args.fp16 = config['training']['fp16']
    args.local_rank = config['training']['local_rank']
    
    return args

def set_seed(seed):
    """재현성을 위한 랜덤 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def setup_distributed_training(args):
    """분산 학습 환경 설정"""
    import torch.distributed as dist
    
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
        args.n_gpu = 1
    
    args.device = device
    return args

def cleanup_checkpoints(args):
    """오래된 체크포인트 정리"""
    if args.save_total_limit <= 0:
        return
    
    checkpoints = sorted(
        [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[1])
    )
    
    if len(checkpoints) > args.save_total_limit:
        checkpoint_to_delete = os.path.join(args.output_dir, checkpoints[0])
        print(f"[{get_timestamp()}] 오래된 체크포인트 삭제: {checkpoint_to_delete}")
        shutil.rmtree(checkpoint_to_delete)
