#!/bin/bash
# 단일 GPU에서 ALBERT 모델 학습 스크립트

MODEL_SIZE="base"  # 'base' 또는 'large'

# 사용할 설정 파일 결정
if [ "$MODEL_SIZE" == "base" ]; then
    CONFIG_FILE="config/albert_base.yaml"
else
    CONFIG_FILE="config/albert_large.yaml"
fi

# 학습 실행
CUDA_VISIBLE_DEVICES=0 python pretrain.py --config $CONFIG_FILE

echo "학습 완료!"
