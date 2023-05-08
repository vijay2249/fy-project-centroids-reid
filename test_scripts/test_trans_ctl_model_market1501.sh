python3 train_ctl_model.py \
--config_file="configs/transformers.yml" \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR './data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 32 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/transformer' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False \
SOLVER.DISTANCE_FUNC 'cosine' \
SOLVER.PCA_K 1024 \
REPRODUCIBLE True \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.USE_CENTROIDS True \
MODEL.PRETRAIN_PATH "/home/system7-user3/project/centroids-reid/logs/market1501/transformer_new/train_ctl_model/version_80/checkpoints/epoch=89.ckpt"
