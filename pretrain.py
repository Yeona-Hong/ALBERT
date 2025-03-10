# pretrain.py
import os
import argparse
import torch
from src.utils.utils import (
    get_timestamp, 
    load_config, 
    config_to_args, 
    set_seed, 
    setup_distributed_training,
    save_config
)
from src.model.model import create_model_and_tokenizer
from src.data.dataset import prepare_datasets, create_dataloaders
from src.training.optimizer import create_optimizer_and_scheduler, setup_fp16_and_distributed
from src.training.train import train, evaluate

def main():
    parser = argparse.ArgumentParser(description="ALBERT 모델 사전학습")
    parser.add_argument("--config", type=str, required=True, help="설정 파일 경로")
    parser.add_argument("--local_rank", type=int, default=-1, help="분산 학습을 위한 로컬 랭크 (필요시 명령줄에서 덮어쓰기)")
    cmd_args = parser.parse_args()
    
    # 설정 파일 로드
    print(f"[{get_timestamp()}] 설정 파일 '{cmd_args.config}' 로드 중...")
    config = load_config(cmd_args.config)
    
    # 명령줄 인자에서 local_rank가 지정되었으면 설정 파일의 값을 덮어씀
    if cmd_args.local_rank != -1:
        config["training"]["local_rank"] = cmd_args.local_rank
    
    # 설정을 args 형식으로 변환
    args = config_to_args(config)
    
    # 랜덤 시드 설정
    print(f"[{get_timestamp()}] 랜덤 시드 설정: {args.seed}")
    set_seed(args.seed)
    
    # 분산 학습 설정
    args = setup_distributed_training(args)
    print(f"[{get_timestamp()}] 장치: {args.device}, GPU 수: {args.n_gpu}")
    
    # 출력 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"[{get_timestamp()}] 출력 디렉토리 생성: {args.output_dir}")
    
    # 설정 저장
    if args.local_rank in [-1, 0]:
        save_config(config, args.output_dir)
    
    # 모델 및 토크나이저 생성
    model, tokenizer, model_config = create_model_and_tokenizer(args)
    
    # 데이터셋 준비
    tokenized_datasets = prepare_datasets(tokenizer, args)
    
    # 데이터로더 생성
    train_dataloader, eval_dataloader = create_dataloaders(tokenized_datasets, tokenizer, args)
    
    # 전체 학습 스텝 수 계산
    if args.max_steps > 0:
        t_total = args.max_steps
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    print(f"[{get_timestamp()}] 전체 학습 스텝 수: {t_total}")
    
    # 옵티마이저 및 스케줄러 설정
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, t_total)
    
    # 혼합 정밀도 및 분산 학습 설정
    model, optimizer = setup_fp16_and_distributed(model, optimizer, args)
    
    # 학습 실행
    global_step, avg_loss = train(
        model, 
        train_dataloader, 
        optimizer, 
        scheduler, 
        args, 
        tokenizer, 
        config, 
        eval_dataloader, 
        tokenized_datasets
    )
    
    # 최종 평가 (검증 데이터셋이 있는 경우)
    if eval_dataloader is not None and args.local_rank in [-1, 0]:
        print(f"[{get_timestamp()}] ***** 최종 평가 실행 *****")
        evaluate(model, eval_dataloader, args)
    
    print(f"[{get_timestamp()}] 학습 완료! 평균 손실: {avg_loss:.4f}")
    
    return global_step, avg_loss

if __name__ == "__main__":
    main()
