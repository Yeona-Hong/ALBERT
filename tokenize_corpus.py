# tokenize_corpus.py
import argparse
import os
from pathlib import Path
import glob
from src.tokenizer.tokenizer import train_tokenizer, convert_to_albert_tokenizer, test_tokenizer
from src.utils.utils import get_timestamp

def main():
    parser = argparse.ArgumentParser(description="한국어 코퍼스를 위한 ALBERT 토크나이저 생성")
    
    # 입력 데이터 관련 인자
    parser.add_argument("--input_dir", type=str, required=True, help="텍스트 파일이 포함된 디렉토리 경로")
    parser.add_argument("--file_pattern", type=str, default="*.txt", help="처리할 파일 패턴 (예: '*.txt')")
    
    # 토크나이저 설정 관련 인자
    parser.add_argument("--vocab_size", type=int, default=32000, help="어휘 크기")
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram", "char", "word"],
                       help="SentencePiece 모델 타입")
    parser.add_argument("--character_coverage", type=float, default=0.9995, help="문자 커버리지")
    parser.add_argument("--model_prefix", type=str, default="albert-ko-spm", help="SentencePiece 모델 파일 접두사")
    
    # 출력 관련 인자
    parser.add_argument("--output_dir", type=str, default="albert-ko-tokenizer", help="토크나이저 저장 디렉토리")
    parser.add_argument("--test", action="store_true", help="토크나이저 테스트 실행 여부")
    
    args = parser.parse_args()
    
    # 입력 디렉토리 체크
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
    
    # 텍스트 파일 목록 가져오기
    input_files = glob.glob(str(input_dir / args.file_pattern))
    if not input_files:
        raise ValueError(f"입력 디렉토리에서 '{args.file_pattern}' 패턴과 일치하는 파일을 찾을 수 없습니다.")
    
    print(f"[{get_timestamp()}] 총 {len(input_files)}개의 파일을 처리합니다.")
    
    # ALBERT 특수 토큰 정의
    special_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    
    # SentencePiece 모델 학습
    print(f"[{get_timestamp()}] SentencePiece 모델 학습 시작...")
    spm_model_path, vocab_path = train_tokenizer(
        input_files, 
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        user_defined_symbols=special_symbols
    )
    
    # ALBERT 토크나이저로 변환
    print(f"[{get_timestamp()}] SentencePiece 모델을 ALBERT 토크나이저로 변환 중...")
    tokenizer = convert_to_albert_tokenizer(spm_model_path, vocab_path, args.output_dir)
    
    print(f"[{get_timestamp()}] 토크나이저가 {args.output_dir}에 성공적으로 저장되었습니다.")
    
    # 토크나이저 테스트 (선택 사항)
    if args.test:
        print(f"[{get_timestamp()}] 토크나이저 테스트 시작...")
        test_tokenizer(args.output_dir)
    
    print(f"[{get_timestamp()}] 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
