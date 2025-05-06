import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import logging
import time
from tqdm import tqdm
import json
from accelerate import Accelerator

from models.model import ConditionalDiffusionModel
# from models.model_sd import ConditionalDiffusionModel
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
            logging.FileHandler(os.path.join(log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("ConditionalDDPM")

def train(config, resume_checkpoint=None):
    """
    訓練條件式擴散模型
    
    Args:
        config: 配置對象
        resume_checkpoint: 恢復訓練的檢查點路徑
    """
    logger = setup_logger(config.OUTPUT_DIR)
    logger.info(f"Starting training with config: {vars(config)}")
    
    # 創建必要的目錄
    config.create_directories()
    
    # 設置設備
    device = config.DEVICE
    logger.info(f"Using device: {device}")
    
    # 設置隨機種子
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if config.FP16 else "no",
        gradient_accumulation_steps=1,
    )
    
    # 加載資料
    train_loader, obj2idx = get_dataloader(config, train=True)
    logger.info(f"DataLoader created with {len(train_loader)} batches")
    
    # 加載評估器
    evaluator = evaluation_model()
    
    # 創建模型
    model = ConditionalDiffusionModel(
        num_classes=config.NUM_CLASSES,
        unet_in_channels=config.LATENT_CHANNELS,
        unet_sample_size=config.IMAGE_SIZE // 4,  # 潛在空間大小
        condition_embedding_dim=config.CONDITION_DIM,
        device=device,
    )
    
    # 設置優化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 設置學習率調度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=1e-5
    )
    
    # 恢復訓練邏輯
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        logger.info(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # 加載模型參數
        model.load_state_dict(checkpoint['model'])
        
        # 加載優化器狀態
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加載調度器狀態（如果存在）
        if checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        # 恢復訓練狀態
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        
        logger.info(f"Resuming from epoch {start_epoch}, global step {global_step}, best loss {best_loss:.6f}")
    else:
        if resume_checkpoint is not None:
            logger.warning(f"Checkpoint {resume_checkpoint} not found, starting from scratch")
        else:
            logger.info("Starting new training")
    
    # 使用 Accelerator 準備模型、優化器、資料加載器
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # 訓練循環
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        # 進度條
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # 計算損失
                loss = model(images, labels)
                
                # 如果使用 DataParallel，需要取平均
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                
                # 反向傳播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if config.GRAD_CLIP > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    
                # 優化器步驟
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新進度條
            epoch_loss += loss.item()
            
            # 顯示當前損失
            progress_bar.set_postfix({
                'loss': loss.item(),
            })
            
            global_step += 1
            
            # 定期保存樣本
            if global_step % 500 == 0:
                save_sample(model, labels[:4], epoch, global_step, config)
        
        # 更新學習率
        scheduler.step()
        
        # 計算平均損失
        avg_loss = epoch_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Avg Loss: {avg_loss:.6f}")
        
        # 保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(accelerator.unwrap_model(model), optimizer, scheduler, epoch, global_step, best_loss, config, is_best=True)
            logger.info(f"Saved best model with loss {best_loss:.6f}")
        
        # 每5個epoch保存一個檢查點
        if (epoch + 1) % 5 == 0:
            save_checkpoint(accelerator.unwrap_model(model), optimizer, scheduler, epoch, global_step, avg_loss, config)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
            
            # 評估當前模型
            try:
                # 使用 try-except 包裝評估過程，防止因評估錯誤中斷訓練
                evaluate(accelerator.unwrap_model(model), evaluator, config, epoch)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                logger.info("Continuing training despite evaluation error")
    
    logger.info("Training completed!")
    
    # 最終評估
    try:
        evaluate(accelerator.unwrap_model(model), evaluator, config, config.NUM_EPOCHS)
    except Exception as e:
        logger.error(f"Error during final evaluation: {e}")

def save_sample(model, labels, epoch, step, config):
    """
    儲存生成的樣本圖片
    """
    # 如果model是DataParallel或被accelerator包裝，使用unwrap_model
    if hasattr(model, "module"):  # DataParallel
        sample_model = model.module
    else:
        sample_model = model
            
    # 生成圖片
    sample_model.eval()
    with torch.no_grad():
        try:
            samples = sample_model.sample(
                labels.to(config.DEVICE),
                guidance_scale=config.GUIDANCE_SCALE,
                num_inference_steps=config.NUM_INFERENCE_STEPS // 2,  # 使用更少步數加快採樣
            )
            
            # 保存圖片
            save_dir = os.path.join(config.OUTPUT_DIR, "samples")
            os.makedirs(save_dir, exist_ok=True)
            save_image(samples, os.path.join(save_dir, f"sample_e{epoch+1}_s{step}.png"), nrow=2)
        except Exception as e:
            print(f"Error generating samples: {e}")
    
    sample_model.train()

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, config, is_best=False):
    """
    保存模型檢查點
    """
    checkpoint = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'loss': loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'config': vars(config)
    }
    
    # 保存檢查點
    if is_best:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pth")
    else:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}.pth")
    
    torch.save(checkpoint, checkpoint_path)

def evaluate(model, evaluator, config, epoch):
    """
    評估模型
    """
    # 加載測試條件
    test_labels, new_test_labels, obj2idx = get_dataloader(config, train=False)
    
    # 將model設置為評估模式
    model.eval()
    
    # 生成測試圖像
    try:
        # 分離評估過程，防止一個評估失敗影響另一個
        try:
            test_images = model.sample(
                test_labels.to(config.DEVICE),
                evaluator=None,  # 先不使用分類器引導，避免錯誤
                guidance_scale=config.GUIDANCE_SCALE,
                classifier_scale=0.0,  # 關閉分類器引導
                num_inference_steps=config.NUM_INFERENCE_STEPS,
            )
        except Exception as e:
            print(f"Error generating test images: {e}")
            test_images = None
            
        try:
            new_test_images = model.sample(
                new_test_labels.to(config.DEVICE),
                evaluator=None,  # 先不使用分類器引導，避免錯誤
                guidance_scale=config.GUIDANCE_SCALE,
                classifier_scale=0.0,  # 關閉分類器引導
                num_inference_steps=config.NUM_INFERENCE_STEPS,
            )
        except Exception as e:
            print(f"Error generating new test images: {e}")
            new_test_images = None
        
        # 評估準確率
        test_acc = 0.0
        new_test_acc = 0.0
        
        if test_images is not None:
            test_norm = (test_images - 0.5) / 0.5  # [0,1] -> [-1,1]
            with torch.no_grad():
                test_acc = evaluator.eval(test_norm.to(config.DEVICE), test_labels.to(config.DEVICE))
                
        if new_test_images is not None:
            new_test_norm = (new_test_images - 0.5) / 0.5
            with torch.no_grad():
                new_test_acc = evaluator.eval(new_test_norm.to(config.DEVICE), new_test_labels.to(config.DEVICE))
        
        # 保存評估結果
        eval_dir = os.path.join(config.OUTPUT_DIR, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 保存準確率
        with open(os.path.join(eval_dir, f"accuracy_epoch{epoch}.json"), 'w') as f:
            json.dump({
                'test_accuracy': float(test_acc),
                'new_test_accuracy': float(new_test_acc),
                'avg_accuracy': float((test_acc + new_test_acc) / 2)
            }, f, indent=4)
        
        # 保存測試圖像
        if test_images is not None and new_test_images is not None:
            test_grid = torch.cat([test_images, new_test_images], dim=0)
            save_image(test_grid, os.path.join(eval_dir, f"test_samples_epoch{epoch}.png"), nrow=8)
        elif test_images is not None:
            save_image(test_images, os.path.join(eval_dir, f"test_samples_epoch{epoch}.png"), nrow=8)
        elif new_test_images is not None:
            save_image(new_test_images, os.path.join(eval_dir, f"new_test_samples_epoch{epoch}.png"), nrow=8)
        
        # 記錄結果
        logging.info(f"Evaluation at epoch {epoch}:")
        logging.info(f"Test Accuracy: {test_acc:.4f}")
        logging.info(f"New Test Accuracy: {new_test_acc:.4f}")
        logging.info(f"Avg Accuracy: {(test_acc + new_test_acc) / 2:.4f}")
        
        # 如果是最終評估，保存每個樣本
        if epoch == config.NUM_EPOCHS and (test_images is not None or new_test_images is not None):
            # 保存test圖像
            if test_images is not None:
                test_dir = os.path.join(config.IMAGES_DIR, "test")
                os.makedirs(test_dir, exist_ok=True)
                
                for i in range(test_images.shape[0]):
                    save_image(test_images[i], os.path.join(test_dir, f"{i}.png"))
            
            # 保存new_test圖像
            if new_test_images is not None:
                new_test_dir = os.path.join(config.IMAGES_DIR, "new_test")
                os.makedirs(new_test_dir, exist_ok=True)
                
                for i in range(new_test_images.shape[0]):
                    save_image(new_test_images[i], os.path.join(new_test_dir, f"{i}.png"))
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
    
    # 將model重新設置為訓練模式
    model.train()
    
    return test_acc, new_test_acc