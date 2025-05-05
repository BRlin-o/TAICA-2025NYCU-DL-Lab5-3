import os
import torch
import numpy as np
import random
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def set_seed(seed):
    """
    設置所有隨機種子以確保可重現性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_images_for_evaluation(images, save_dir, indices=None):
    """
    保存圖像用於評估
    
    Args:
        images (torch.Tensor): 形狀為 [B, C, H, W] 的圖像張量，值範圍為 [0, 1]
        save_dir (str): 保存目錄
        indices (list, optional): 圖像索引列表，如果為None，則使用範圍 0 到 B-1
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if indices is None:
        indices = list(range(images.shape[0]))
    
    for i, idx in enumerate(indices):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f"{idx}.png"))

def visualize_batch(images, nrow=8, title=None, save_path=None):
    """
    視覺化一批圖像
    
    Args:
        images (torch.Tensor): 形狀為 [B, C, H, W] 的圖像張量，值範圍為 [0, 1]
        nrow (int): 每行的圖像數量
        title (str, optional): 圖像標題
        save_path (str, optional): 保存路徑，如果為None，則顯示圖像
    """
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid = grid.cpu().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    if title:
        plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_loss_curve(losses, save_path=None):
    """
    繪製損失曲線
    
    Args:
        losses (list): 損失值列表
        save_path (str, optional): 保存路徑，如果為None，則顯示圖像
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def format_time(seconds):
    """
    將秒數格式化為時:分:秒的形式
    
    Args:
        seconds (float): 秒數
        
    Returns:
        str: 格式化的時間字符串
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"