import os
import argparse
import torch
from config import Config
from train import train
from sample import generate_test_images, load_model
from evaluator import evaluation_model
import logging

def set_seed(seed):
    """設置所有隨機種子以確保可重現性"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='Conditional DDPM for i-CLEVR dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'],
                      help='運行模式: 訓練模型或生成樣本')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='用於採樣或繼續訓練的特定檢查點路徑')
    parser.add_argument('--resume', action='store_true',
                      help='從檢查點恢復訓練')
    parser.add_argument('--no_eval', action='store_true',
                      help='採樣時不進行評估')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='批次大小，覆蓋配置中的設置')
    parser.add_argument('--epochs', type=int, default=None,
                      help='訓練的輪數，覆蓋配置中的設置')
    parser.add_argument('--seed', type=int, default=None,
                      help='隨機種子，覆蓋配置中的設置')
    parser.add_argument('--device', type=str, default=None,
                      help='使用的設備，例如 cuda:0 或 cpu')
    
    args = parser.parse_args()
    
    # 加載配置
    config = Config()
    
    # 更新配置（如果提供了命令行參數）
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.seed is not None:
        config.SEED = args.seed
    if args.device is not None:
        config.DEVICE = torch.device(args.device)
    
    # 創建必要的目錄
    config.create_directories()
    
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, "main.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("main")
    
    # 設置隨機種子
    set_seed(config.SEED)
    
    # 確定檢查點路徑
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.resume:
        # 如果沒有指定檢查點但要恢復訓練，使用最後一個檢查點或最佳模型
        checkpoint_dir = config.CHECKPOINT_DIR
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        
        if 'model_best.pth' in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pth')
        elif checkpoint_files:
            # 按創建時間排序檢查點文件
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        else:
            checkpoint_path = None
            logger.warning("No checkpoints found, starting from scratch")
    else:
        checkpoint_path = None
    
    if args.mode == 'train':
        if args.resume and checkpoint_path:
            logger.info(f"恢復訓練，從檢查點: {checkpoint_path}")
            train(config, resume_checkpoint=checkpoint_path)
        else:
            logger.info("開始新的訓練...")
            train(config)
        
    elif args.mode == 'sample':
        logger.info("採樣生成圖像...")
        # 檢查點路徑
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pth")
        
        # 加載評估器（如果需要）
        evaluator = None if args.no_eval else evaluation_model()
        
        # 加載模型
        model, epoch = load_model(checkpoint_path, config)
        logger.info(f"已加載來自 epoch {epoch} 的模型")
        
        # 生成圖像
        generate_test_images(model, config, evaluator)
        
        logger.info("生成完成！圖像已保存")

if __name__ == "__main__":
    main()