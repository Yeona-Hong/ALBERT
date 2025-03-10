# src/tokenizer/tokenizer.py
import os
import shutil
from pathlib import Path
import tempfile
from tqdm import tqdm
from transformers import AlbertTokenizerFast
from src.utils.utils import get_timestamp

def train_tokenizer(input_files, vocab_size=32000, model_prefix="albert-ko-spm", 
                   model_type="bpe", character_coverage=0.9995, 
                   user_defined_symbols=None):
    """
    SentencePiece 모델을 학습시키는 함수
    
    Args:
        input_files: 학습에 사용할 텍스트 파일 리스트
        vocab_size: 어휘 크기
        model_prefix: 모델 저장 시 사용할 접두사
        model_type: 토크나이저 타입 (bpe, unigram, char, word)
        character_coverage: 문자 커버리지 비율
        user_defined_symbols: 사용자 정의 심볼 리스트
    """
    import sentencepiece as spm
    
    # 임시 파일 생성하여 모든 텍스트 데이터 병합
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp_file:
        temp_file_name = temp_file.name
        for input_file in tqdm(input_files, desc="데이터 파일 처리 중"):
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        temp_file.write(line + '\n')
    
    print(f"[{get_timestamp()}] 임시 파일 생성 완료: {temp_file_name}")
    
    # SentencePiece 학습 옵션 설정
    train_args = [
        f"--input={temp_file_name}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        f"--pad_id=0",
        f"--unk_id=1",
        f"--bos_id=2",
        f"--eos_id=3",
        f"--mask_id=4",
        "--input_sentence_size=10000000",  # 처리할 최대 문장 수
        "--shuffle_input_sentence=true",
        "--normalization_rule_name=nmt_nfkc_cf"  # 정규화 규칙
    ]
    
    # 사용자 정의 심볼 추가
    if user_defined_symbols:
        train_args.append(f"--user_defined_symbols={','.join(user_defined_symbols)}")
    
    # SentencePiece 학습 실행
    print(f"[{get_timestamp()}] SentencePiece 학습 시작...")
    spm.SentencePieceTrainer.train(" ".join(train_args))
    print(f"[{get_timestamp()}] SentencePiece 모델이 {model_prefix}.model과 {model_prefix}.vocab에 저장되었습니다.")
    
    # 임시 파일 삭제
    os.unlink(temp_file_name)
    
    return f"{model_prefix}.model", f"{model_prefix}.vocab"

def convert_to_albert_tokenizer(spm_model_path, vocab_path, save_directory):
    """
    SentencePiece 모델을 ALBERT 토크나이저로 변환
    
    Args:
        spm_model_path: SentencePiece 모델 경로
        vocab_path: SentencePiece 어휘 파일 경로
        save_directory: 저장할 디렉토리
    """
    # 디렉토리 생성
    os.makedirs(save_directory, exist_ok=True)
    
    # SentencePiece 모델 파일 복사
    shutil.copy(spm_model_path, os.path.join(save_directory, "spiece.model"))
    
    # ALBERT 특수 토큰 설정
    special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    }
    
    # AlbertTokenizerFast 생성
    tokenizer = AlbertTokenizerFast(
        vocab_file=os.path.join(save_directory, "spiece.model"),
        **special_tokens
    )
    
    # 토크나이저 저장
    tokenizer.save_pretrained(save_directory)
    print(f"[{get_timestamp()}] ALBERT 토크나이저가 {save_directory}에 저장되었습니다.")
    
    return tokenizer

def test_tokenizer(tokenizer_path):
    """
    학습된 토크나이저로 테스트 텍스트를 토큰화하고 결과 출력
    
    Args:
        tokenizer_path: 토크나이저 디렉토리 경로
    """
    tokenizer = AlbertTokenizerFast.from_pretrained(tokenizer_path)
    
    # 다양한 테스트 텍스트 준비
    test_texts = [
        "안녕하세요! 이것은 한국어 ALBERT 모델을 위한 서브워드 토크나이저 테스트입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 인공지능의 한 분야입니다.",
        "서울은 대한민국의 수도이며, 인구가 약 1000만 명입니다.",
        "프로그래밍 언어에는 파이썬, 자바, C++ 등이 있습니다.",
        "인공지능(AI)과 머신러닝은 4차 산업혁명의 핵심 기술입니다.",
        "대한민국의 역사는 반만년이 넘습니다. 고조선부터 현대까지 다양한 왕조와 정부가 있었습니다."
    ]
    
    print("===== 토크나이저 테스트 결과 =====")
    for i, text in enumerate(test_texts):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        
        print(f"\n[테스트 {i+1}]")
        print(f"원문: {text}")
        print(f"토큰화 결과: {tokens}")
        print(f"토큰 수: {len(tokens)}")
        print(f"IDs: {token_ids}")
        
        # 토큰별 ID 대응 확인
        token_id_pairs = []
        for j, token in enumerate(tokens):
            # [CLS]와 [SEP] 토큰 제외하고 실제 텍스트 토큰만 표시
            if j < len(token_ids) - 2:
                token_id_pairs.append(f"{token}: {token_ids[j+1]}")
        print(f"토큰-ID 쌍: {token_id_pairs}")
