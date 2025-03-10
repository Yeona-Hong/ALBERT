#!/bin/bash
# 다중 GPU에서 분산 학습 스크립트

MODEL_SIZE="base"  # 'base' 또는 'large'
NUM_GPUS=4  # 사용할 GPU 수

# 사용할 설정 파일 결정
if [ "$MODEL_SIZE" == "base" ]; then
    CONFIG_FILE="config/albert_base.yaml"
else
    CONFIG_FILE="config/albert_large.yaml"
fi

# 분산 학습 실행
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS pretrain.py --config $CONFIG_FILE

echo "분산 학습 완료!"
