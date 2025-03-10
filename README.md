# ALBERT

albert-ko/

│

├── config/

│   ├── albert_base.yaml   # ALBERT base 모델 설정

│   └── albert_large.yaml  # ALBERT large 모델 설정 (선택적)

│

├── scripts/

│   ├── train.sh           # 학습 실행 스크립트

│   └── distributed_train.sh # 분산 학습 스크립트

│

├── src/

│   ├── __init__.py

│   ├── data/

│   │   ├── __init__.py

│   │   └── dataset.py     # 데이터셋 관련 함수들

│   │

│   ├── model/

│   │   ├── __init__.py

│   │   └── modeling.py    # 모델 관련 함수들

│   │

│   ├── tokenizer/

│   │   ├── __init__.py

│   │   └── tokenizer.py   # 토크나이저 관련 함수들

│   │

│   ├── training/

│   │   ├── __init__.py

│   │   ├── optimizer.py   # 옵티마이저 관련 함수들

│   │   └── trainer.py     # 학습 관련 함수들

│   │

│   └── utils/

│       ├── __init__.py

│       └── utils.py       # 유틸리티 함수들

│

├── tokenize_corpus.py     # 토크나이저 생성 스크립트

├── pretrain.py            # 사전학습 메인 스크립트

└── requirements.txt       # 필요한 패키지 목록
