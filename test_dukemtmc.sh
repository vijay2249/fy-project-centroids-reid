python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'dukemtmcreid' \
DATASETS.ROOT_DIR '/home/cse04_197121/project/centroids-reid/data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/dukemtmcreid/256_resnet50/' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/home/cse04_197121/project/centroids-reid/logs/dukemtmcreid/256_resnet50/train_ctl_model/version_0/checkpoints/epoch=119.ckpt"
