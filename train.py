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

from models.model import ConditionalLDM
from evaluator import evaluation_model
from data.dataset import get_dataloader

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

def train(config):
    """
    訓練條件式擴散模型
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
    
    # 加載資料
    train_loader, obj2idx = get_dataloader(config, train=True)
    logger.info(f"DataLoader created with {len(train_loader)} batches")
    
    # 加載評估器
    evaluator = evaluation_model()
    
    # 創建模型
    model = ConditionalLDM(
        unet_dim=config.UNET_DIM,
        condition_dim=config.CONDITION_DIM,
        time_embedding_dim=config.TIME_EMBEDDING_DIM,
        num_classes=config.NUM_CLASSES,
        use_attention=config.USE_ATTENTION,
        image_size=config.IMAGE_SIZE,
        channels=3,
        latent_channels=config.LATENT_CHANNELS,
        variational=True,
        training=True
    )
    model = model.to(device)
    
    # 如果有多個GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
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
    
    # 混合精度訓練
    scaler = torch.cuda.amp.GradScaler() if config.FP16 else None
    
    # 訓練循環
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_kl_loss = 0
        
        # 進度條
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # 混合精度訓練
            if config.FP16:
                with torch.cuda.amp.autocast():
                    loss, mse_loss, kl_loss = model(images, labels)
                    
                    # 如果使用DataParallel，需要取平均
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                        mse_loss = mse_loss.mean()
                        kl_loss = kl_loss.mean()
                
                # 反向傳播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if config.GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 標準訓練
                loss, mse_loss, kl_loss = model(images, labels)
                
                # 如果使用DataParallel，需要取平均
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                    mse_loss = mse_loss.mean()
                    kl_loss = kl_loss.mean()
                
                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if config.GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    
                optimizer.step()
            
            # 更新進度條
            epoch_loss += loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # 顯示當前損失
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mse_loss': mse_loss.item(),
                'kl_loss': kl_loss.item()
            })
            
            global_step += 1
            
            # 定期保存樣本
            if global_step % 500 == 0:
                save_sample(model, labels[:4], epoch, global_step, config)
        
        # 更新學習率
        scheduler.step()
        
        # 計算平均損失
        avg_loss = epoch_loss / len(train_loader)
        avg_mse_loss = epoch_mse_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Avg Loss: {avg_loss:.6f}, MSE: {avg_mse_loss:.6f}, KL: {avg_kl_loss:.6f}")
        
        # 保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_loss, config, is_best=True)
            logger.info(f"Saved best model with loss {best_loss:.6f}")
        
        # 每5個epoch保存一個檢查點
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, avg_loss, config)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
            
            # 評估當前模型
            evaluate(model, evaluator, config, epoch)
    
    logger.info("Training completed!")
    
    # 最終評估
    evaluate(model, evaluator, config, config.NUM_EPOCHS)

def save_sample(model, labels, epoch, step, config):
    """
    儲存生成的樣本圖片
    """
    model.eval()
    with torch.no_grad():
        # 如果model是DataParallel，使用model.module
        if isinstance(model, nn.DataParallel):
            sample_model = model.module
        else:
            sample_model = model
            
        # 生成圖片
        samples = sample_model.sample(
            labels.to(config.DEVICE),
            guidance_scale=config.GUIDANCE_SCALE,
            num_inference_steps=config.NUM_INFERENCE_STEPS // 2,  # 使用更少步數加快採樣
            device=config.DEVICE
        )
        
        # 保存圖片
        save_dir = os.path.join(config.OUTPUT_DIR, "samples")
        os.makedirs(save_dir, exist_ok=True)
        save_image(samples, os.path.join(save_dir, f"sample_e{epoch+1}_s{step}.png"), nrow=2)
    
    model.train()

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, config, is_best=False):
    """
    保存模型檢查點
    """
    # 如果model是DataParallel，保存model.module
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'loss': loss,
        'model': model_state,
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
    
    # 如果model是DataParallel，使用model.module
    if isinstance(model, nn.DataParallel):
        sample_model = model.module
    else:
        sample_model = model
    
    # 生成測試圖像
    test_images = sample_model.sample(
        test_labels.to(config.DEVICE),
        evaluator=evaluator,
        guidance_scale=config.GUIDANCE_SCALE,
        classifier_scale=config.CLASSIFIER_GUIDANCE_SCALE,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        device=config.DEVICE
    )
    
    new_test_images = sample_model.sample(
        new_test_labels.to(config.DEVICE),
        evaluator=evaluator,
        guidance_scale=config.GUIDANCE_SCALE,
        classifier_scale=config.CLASSIFIER_GUIDANCE_SCALE,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        device=config.DEVICE
    )
    
    # 評估準確率
    test_norm = (test_images - 0.5) / 0.5  # [0,1] -> [-1,1]
    new_test_norm = (new_test_images - 0.5) / 0.5
    
    with torch.no_grad():
        test_acc = evaluator.eval(test_norm.to(config.DEVICE), test_labels.to(config.DEVICE))
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
    test_grid = torch.cat([test_images, new_test_images], dim=0)
    save_image(test_grid, os.path.join(eval_dir, f"test_samples_epoch{epoch}.png"), nrow=8)
    
    # 記錄結果
    logging.info(f"Evaluation at epoch {epoch}:")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"New Test Accuracy: {new_test_acc:.4f}")
    logging.info(f"Avg Accuracy: {(test_acc + new_test_acc) / 2:.4f}")
    
    # 如果是最終評估，保存每個樣本
    if epoch == config.NUM_EPOCHS:
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
    
    # 將model重新設置為訓練模式
    model.train()
    
    return test_acc, new_test_acc