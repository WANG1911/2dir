import torch

class Config:

    TRAIN_IMG_FOLDER = "data/image/train/image"
    TRAIN_CSV_FOLDER = "data/image/train/contact"
    TEST_IMG_FOLDER = "data/image/test/image"
    TEST_CSV_FOLDER = "data/image/test/contact"
    OUTPUT_FOLDER = "data/out"


    OUTPUT_SIZE = (7, 7)
    MODEL_SAVE_PATH = "tnt_s_patch16_224.pth.tar"
    BEST_MODEL_SAVE_PATH = "best_model.pth"
    FINAL_MODEL_SAVE_PATH = "final_model.pth"

    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE_BACKBONE = 1e-4
    LEARNING_RATE_UP = 1e-3
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    TENSORBOARD_LOG_DIR = "runs/experiment_1"

    SEED = 42