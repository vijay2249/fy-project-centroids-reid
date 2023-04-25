python3 train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR './data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/mars/256_resnet50' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False \
SOLVER.DISTANCE_FUNC 'manhattan' \
REPRODUCIBLE True \
MODEL.PRETRAIN_PATH './logs/market1501/256_resnet50/train_ctl_model/version_50/auto_checkpoints/checkpoint_38.pth' \
MODEL.RESUME_TRAINING True
