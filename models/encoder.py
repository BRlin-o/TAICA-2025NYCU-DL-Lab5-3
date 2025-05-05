import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """雙層卷積區塊，用於編碼器中"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下採樣區塊，用於編碼器中"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Encoder(nn.Module):
    """
    VAE編碼器，將圖像轉換為潛在表示
    """
    def __init__(self, in_channels=3, z_channels=4, channels_mult=(1, 2, 4, 8)):
        super().__init__()
        
        # 初始層
        self.inc = DoubleConv(in_channels, 64)
        
        # 下採樣路徑
        self.down_layers = nn.ModuleList()
        channels = 64
        for mult in channels_mult:
            out_channels = 64 * mult
            self.down_layers.append(Down(channels, out_channels))
            channels = out_channels
        
        # 最終投影層
        self.final_conv = nn.Conv2d(channels, z_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.inc(x)
        
        # 逐層下採樣
        for layer in self.down_layers:
            x = layer(x)
        
        # 投影到潛在空間
        z = self.final_conv(x)
        
        return z

class EncoderWithVariational(nn.Module):
    """
    變分自編碼器(VAE)編碼器，增加了變分特性
    將圖像編碼為均值和方差參數
    """
    def __init__(self, in_channels=3, z_channels=4, channels_mult=(1, 2, 4, 8)):
        super().__init__()
        
        # 基本編碼器架構
        self.base_encoder = Encoder(in_channels, z_channels * 2, channels_mult)
        
    def reparameterize(self, mu, logvar):
        """重參數化技巧，使梯度可以通過隨機採樣操作"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, sample=True):
        # 得到均值和log方差
        h = self.base_encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        
        # 訓練時使用重參數化，評估時直接使用均值
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        return z, mu, logvar