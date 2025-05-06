import torch
import os

class Config:
    # 資料和路徑設定
    DATA_DIR = "iclevr"
    OUTPUT_DIR = "output"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    
    # 資料集文件
    TRAIN_JSON = "train.json"
    TEST_JSON = "test.json"
    NEW_TEST_JSON = "new_test.json"
    OBJECTS_JSON = "objects.json"
    IMAGE_SIZE = 64
    
    # 模型架構參數
    LATENT_CHANNELS = 4  # 潛在空間的通道數
    # ENCODER_SCALE_FACTOR = 4  # 編碼器下採樣因子
    CONDITION_DIM = 256  # 條件嵌入維度
    # TIME_EMBEDDING_DIM = 256  # 時間嵌入維度
    # UNET_DIM = 128  # UNet基礎維度
    NUM_CLASSES = 24  # 物件類別數
    # USE_ATTENTION = True  # 使用交叉注意力
    
    # 擴散過程參數
    NUM_TRAIN_TIMESTEPS = 1000
    NUM_INFERENCE_STEPS = 100  # DDIM採樣步數
    # BETA_SCHEDULE = "linear"  # 可選: linear, cosine
    # PREDICTION_TYPE = "epsilon"  # 預測噪聲
    
    # 訓練參數
    BATCH_SIZE = 64  # RTX 3090可以用更大的批次如128
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 1e-4
    FP16 = False  # 使用混合精度訓練
    GRAD_CLIP = 0.5  # 梯度裁剪
    SEED = 42
    
    # 硬體設定
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8
    
    # 採樣參數
    GUIDANCE_SCALE = 3.0  # Classifier-free guidance強度
    CLASSIFIER_GUIDANCE_SCALE = 0.3  # 分類器引導強度
    
    # 創建必要的目錄
    @staticmethod
    def create_directories():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.IMAGES_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.IMAGES_DIR, "test"), exist_ok=True)
        os.makedirs(os.path.join(Config.IMAGES_DIR, "new_test"), exist_ok=True)