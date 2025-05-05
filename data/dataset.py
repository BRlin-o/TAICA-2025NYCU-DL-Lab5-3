import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

class ICLEVRDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, obj2idx=None):
        """
        i-CLEVR 資料集加載器
        
        Args:
            json_file (str): 標籤JSON檔案路徑
            img_dir (str): 圖片目錄路徑
            transform (callable, optional): 圖像轉換函數
            obj2idx (dict): 物件名稱到索引的映射
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.obj2idx = obj2idx
        
        # 將字典轉換為列表 (訓練資料)
        if isinstance(self.data, dict):
            self.img_files = list(self.data.keys())
        else:
            # 測試資料已經是列表格式
            self.img_files = [f"{i}.png" for i in range(len(self.data))]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        
        # 獲取標籤
        if isinstance(self.data, dict):
            # 訓練資料格式
            labels = self.data[img_name]
            img_path = os.path.join(self.img_dir, img_name)
        else:
            # 測試資料格式
            labels = self.data[idx]
            img_path = os.path.join(self.img_dir, img_name)
        
        # 轉換標籤為 one-hot 向量
        label_tensor = torch.zeros(24)
        for obj in labels:
            label_tensor[self.obj2idx[obj]] = 1.0
        
        # 讀取圖片 (如果存在)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label_tensor
        else:
            # 對於測試資料，可能還沒有圖片
            return torch.zeros((3, 64, 64)), label_tensor

def get_transforms(train=True, image_size=64):
    """
    獲取訓練和測試的圖像轉換函數
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return transform

def get_dataloader(config, train=True):
    """
    創建訓練或測試的資料加載器
    """
    # 加載物件映射
    with open(config.OBJECTS_JSON, 'r') as f:
        obj2idx = json.load(f)
    
    transform = get_transforms(train, config.IMAGE_SIZE)
    
    if train:
        dataset = ICLEVRDataset(
            json_file=config.TRAIN_JSON, 
            img_dir=os.path.join(config.DATA_DIR, "images"), 
            transform=transform,
            obj2idx=obj2idx
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        return dataloader, obj2idx
    else:
        # 對於測試資料
        with open(config.TEST_JSON, 'r') as f:
            test_conditions = json.load(f)
        with open(config.NEW_TEST_JSON, 'r') as f:
            new_test_conditions = json.load(f)
        
        # 轉換測試條件為張量
        test_labels = []
        for cond in test_conditions:
            label = torch.zeros(24)
            for obj in cond:
                label[obj2idx[obj]] = 1.0
            test_labels.append(label)
        test_labels = torch.stack(test_labels)
        
        new_test_labels = []
        for cond in new_test_conditions:
            label = torch.zeros(24)
            for obj in cond:
                label[obj2idx[obj]] = 1.0
            new_test_labels.append(label)
        new_test_labels = torch.stack(new_test_labels)
        
        return test_labels, new_test_labels, obj2idx