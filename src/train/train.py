# src/training/trainer.py
import torch
from tqdm import tqdm
from src.utils.utils import get_timestamp, cleanup_checkpoints
from src.model.model import save_checkpoint

def train(model, train_dataloader, optimizer, scheduler, args, tokenizer, config, eval_dataloader=None, tokenized_datasets=None):
    """ALBERT 모델 학습 함수"""
    # 전체 학습 스텝 수 계산
    if args.max_steps > 0:
        t_total = args.max_steps
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # 학습 로그 초기화
    print("\n" + "="*50)
    print(f"[{get_timestamp()}] ***** 학습 시작 *****")
    print(f"[{get_timestamp()}] 모델 타입 = ALBERT")
    print(f"[{get_timestamp()}] 임베딩 크기 = {args.embedding_size}")
    print(f"[{get_timestamp()}] 은닉층 크기 = {args.hidden_size}")
    print(f"[{get_timestamp()}] 은닉층 수 = {args.num_hidden_layers}")
    print(f"[{get_timestamp()}] 배치 크기 = {args.per_device_train_batch_size}")
    print(f"[{get_timestamp()}] 그래디언트 누적 스텝 = {args.gradient_accumulation_steps}")
    print(f"[{get_timestamp()}] 총 최적화 스텝 = {t_total}")
    print("="*50 + "\n")
    
    # 학습 변수 초기화
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    
    # 메인 학습 루프
    progress_bar = tqdm(range(t_total), disable=args.local_rank not in [-1, 0])
    
    while global_step < t_total:
        epoch_iterator = iter(train_dataloader)
        
        while True:
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                # 데이터셋을 모두 읽었으면 다시 시작
                if args.local_rank != -1:
                    # 분산 학습 시 에폭마다 샘플러 셔플
                    train_dataloader.sampler.set_epoch(global_step // len(train_dataloader))
                epoch_iterator = iter(train_dataloader)
                batch = next(epoch_iterator)
            
            model.train()
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            # 순전파
            outputs = model(**batch)
            loss = outputs.loss
            
            # 손실 스케일링 (그래디언트 누적)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # 역전파
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            tr_loss += loss.item()
            
            # 그래디언트 누적 완료 시 최적화 단계 실행
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=1.0, 
                    norm_type=2
                )
                
                # 옵티마이저 스텝
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                progress_bar.update(1)
                global_step += 1
                
                # 로깅
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 학습 손실 로깅
                    print(f"[{get_timestamp()}] Step = {global_step} / {t_total}, Loss = {(tr_loss - logging_loss) / args.logging_steps:.5f}")
                    logging_loss = tr_loss
                
                # 모델 저장
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # 체크포인트 저장
                    save_checkpoint(model, tokenizer, optimizer, scheduler, config, args, global_step)
                    
                    # 오래된 체크포인트 정리
                    cleanup_checkpoints(args)
                
                # 평가 (선택적)
                if args.eval_steps > 0 and global_step % args.eval_steps == 0 and eval_dataloader is not None:
                    evaluate(model, eval_dataloader, args)
            
            if global_step >= t_total:
                break
    
    # 학습 완료 후 최종 모델 저장
    if args.local_rank in [-1, 0]:
        print(f"[{get_timestamp()}] 전체 학습 완료. 총 스텝: {global_step}")
        
        final_checkpoint_dir = os.path.join(args.output_dir, "final-model")
        save_checkpoint(model, tokenizer, optimizer, scheduler, config, args, global_step, final_checkpoint_dir)
    
    return global_step, tr_loss / global_step

def evaluate(model, eval_dataloader, args):
    """평가 데이터셋에서 모델 성능 평가"""
    print(f"[{get_timestamp()}] ***** 모델 평가 *****")
    
    # 평가 변수 초기화
    eval_loss = 0.0
    nb_eval_steps = 0
    
    # 평가 루프
    model.eval()
    for batch in tqdm(eval_dataloader, desc="평가 중", disable=args.local_rank not in [-1, 0]):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
        
        eval_loss += loss.item()
        nb_eval_steps += 1
    
    # 평균 손실 계산
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    
    result = {
        "perplexity": perplexity,
        "eval_loss": eval_loss
    }
    
    print(f"[{get_timestamp()}] ***** 평가 결과 *****")
    for key, value in result.items():
        print(f"[{get_timestamp()}] {key} = {value}")
    
    return result
