python3 train_ctl_model.py \
--config_file="configs/transformers.yml" \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR './data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 32 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/transformer_new' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False \
SOLVER.DISTANCE_FUNC 'euclidean' \
SOLVER.PCA_K 1024 \
REPRODUCIBLE True \
SOLVER.EVAL_PERIOD 10
