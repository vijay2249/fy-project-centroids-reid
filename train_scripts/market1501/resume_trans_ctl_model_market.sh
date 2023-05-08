python3 train_ctl_model.py \
--config_file="configs/transformers.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR './data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 32 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/transformer_new' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False \
SOLVER.DISTANCE_FUNC 'manhattan' \
REPRODUCIBLE True \
MODEL.PRETRAIN_PATH 'logs/market1501/transformer_new/train_ctl_model/version_53/auto_checkpoints/checkpoint_94.pth' \
MODEL.RESUME_TRAINING True
