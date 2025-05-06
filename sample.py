import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import argparse
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# from models.model import ConditionalDiffusionModel
from models.model_sd import ConditionalDiffusionModel
from evaluator import evaluation_model
from data.dataset import get_dataloader
from config import Config

def setup_logger(log_dir):
    """設置日誌記錄器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 設置基本日誌記錄
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'sample.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("ConditionalDDPM_Sampling")

def load_model(checkpoint_path, config):
    """加載訓練好的模型"""
    device = config.DEVICE
    
    # 創建模型
    model = ConditionalDiffusionModel(
        num_classes=config.NUM_CLASSES,
        unet_in_channels=config.LATENT_CHANNELS,
        unet_sample_size=config.IMAGE_SIZE // 4,  # 潛在空間大小
        condition_embedding_dim=config.CONDITION_DIM,
        device=device,
    )
    
    # 加載檢查點
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['epoch']

def generate_test_images(model, config, evaluator=None):
    """生成測試圖像並評估"""
    logger = setup_logger(config.OUTPUT_DIR)
    logger.info("Generating test images...")
    
    # 加載測試條件
    test_labels, new_test_labels, obj2idx = get_dataloader(config, train=False)
    
    # 將條件移至設備
    test_labels = test_labels.to(config.DEVICE)
    new_test_labels = new_test_labels.to(config.DEVICE)
    
    # 獲取指定的標籤 ["red sphere", "cyan cylinder", "cyan cube"]
    special_label = torch.zeros(1, 24)
    special_label[0, obj2idx["red sphere"]] = 1.0
    special_label[0, obj2idx["cyan cylinder"]] = 1.0
    special_label[0, obj2idx["cyan cube"]] = 1.0
    special_label = special_label.to(config.DEVICE)
    
    # 設置隨機生成器以確保可重現性
    generator = torch.Generator(device=config.DEVICE).manual_seed(config.SEED)
    
    # 生成test圖像
    logger.info("Generating images for test.json...")
    test_images = model.sample(
        test_labels,
        evaluator=evaluator,
        guidance_scale=config.GUIDANCE_SCALE,
        classifier_scale=config.CLASSIFIER_GUIDANCE_SCALE,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        generator=generator,
    )
    
    # 生成new_test圖像
    logger.info("Generating images for new_test.json...")
    new_test_images = model.sample(
        new_test_labels,
        evaluator=evaluator,
        guidance_scale=config.GUIDANCE_SCALE,
        classifier_scale=config.CLASSIFIER_GUIDANCE_SCALE,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        generator=generator,
    )
    
    # 生成去噪過程可視化
    logger.info("Generating denoising visualization...")
    process_images = model.visualize_denoising(
        special_label,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        num_images=8,
        generator=generator,
    )
    
    # 評估生成的圖像
    if evaluator is not None:
        logger.info("Evaluating generated images...")
        test_norm = (test_images - 0.5) / 0.5  # [0,1] -> [-1,1]
        new_test_norm = (new_test_images - 0.5) / 0.5
        
        with torch.no_grad():
            test_acc = evaluator.eval(test_norm.to(config.DEVICE), test_labels)
            new_test_acc = evaluator.eval(new_test_norm.to(config.DEVICE), new_test_labels)
            
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"New Test Accuracy: {new_test_acc:.4f}")
        logger.info(f"Average Accuracy: {(test_acc + new_test_acc) / 2:.4f}")
        
        # 保存評估結果
        with open(os.path.join(config.OUTPUT_DIR, "final_accuracy.json"), 'w') as f:
            json.dump({
                'test_accuracy': float(test_acc),
                'new_test_accuracy': float(new_test_acc),
                'avg_accuracy': float((test_acc + new_test_acc) / 2)
            }, f, indent=4)
    
    # 保存圖像網格
    logger.info("Saving image grids...")
    save_dir = os.path.join(config.OUTPUT_DIR, "final_results")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存test網格
    test_grid = make_grid(test_images, nrow=8)
    save_image(test_grid, os.path.join(save_dir, "test_grid.png"))
    
    # 保存new_test網格
    new_test_grid = make_grid(new_test_images, nrow=8)
    save_image(new_test_grid, os.path.join(save_dir, "new_test_grid.png"))
    
    # 保存去噪過程
    process_grid = make_grid(process_images.reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), nrow=process_images.shape[0])
    save_image(process_grid, os.path.join(save_dir, "denoising_process.png"))
    
    # 保存每張單獨的圖像
    logger.info("Saving individual images...")
    
    # 保存test圖像
    test_dir = os.path.join(config.IMAGES_DIR, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    for i in range(test_images.shape[0]):
        save_image(test_images[i], os.path.join(test_dir, f"{i}.png"))
    
    # 保存new_test圖像
    new_test_dir = os.path.join(config.IMAGES_DIR, "new_test")
    os.makedirs(new_test_dir, exist_ok=True)
    
    for i in range(new_test_images.shape[0]):
        save_image(new_test_images[i], os.path.join(new_test_dir, f"{i}.png"))
    
    # 保存去噪過程圖像
    process_dir = os.path.join(save_dir, "denoising_steps")
    os.makedirs(process_dir, exist_ok=True)
    
    for i in range(process_images.shape[0]):
        save_image(process_images[i], os.path.join(process_dir, f"step_{i}.png"))
    
    logger.info("All images generated and saved successfully!")
    
    # 創建可視化圖像用於報告
    visualize_for_report(test_images, new_test_images, process_images, save_dir)
    
    return test_images, new_test_images, process_images

def visualize_for_report(test_images, new_test_images, process_images, save_dir):
    """創建報告用可視化圖像"""
    # 創建資料夾
    report_dir = os.path.join(save_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # 為報告創建更好的網格
    plt.figure(figsize=(16, 16))
    for i in range(min(32, test_images.size(0))):
        plt.subplot(4, 8, i+1)
        plt.imshow(test_images[i].cpu().permute(1, 2, 0))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "test_grid_report.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(16, 16))
    for i in range(min(32, new_test_images.size(0))):
        plt.subplot(4, 8, i+1)
        plt.imshow(new_test_images[i].cpu().permute(1, 2, 0))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "new_test_grid_report.png"), dpi=300)
    plt.close()
    
    # 去噪過程可視化
    plt.figure(figsize=(20, 4))
    for i in range(process_images.size(0)):
        plt.subplot(1, process_images.size(0), i+1)
        plt.imshow(process_images[i][0].cpu().permute(1, 2, 0))
        plt.title(f'Step {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "denoising_process_report.png"), dpi=300)
    plt.close()

def main(args):
    """主函數"""
    # 加載配置
    config = Config()
    config.create_directories()
    
    # 如果提供了特定的檢查點路徑，使用它
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(config.CHECKPOINT_DIR, "model_best.pth")
    
    # 加載評估器
    evaluator = evaluation_model()
    
    # 加載模型
    model, epoch = load_model(checkpoint_path, config)
    print(f"Loaded model from epoch {epoch}")
    
    # 生成圖像並評估
    generate_test_images(model, config, evaluator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Conditional DDPM")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint to use")
    args = parser.parse_args()
    
    main(args)