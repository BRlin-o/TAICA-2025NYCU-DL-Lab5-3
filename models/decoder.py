import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """雙層卷積區塊，用於解碼器中"""
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

class Up(nn.Module):
    """上採樣區塊，用於解碼器中"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 上採樣方法選擇：雙線性插值或轉置卷積
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class OutConv(nn.Module):
    """輸出卷積層，用於最終輸出通道調整"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    """
    VAE解碼器，將潛在表示轉換回圖像
    """
    def __init__(self, z_channels=4, out_channels=3, channels_mult=(8, 4, 2, 1)):
        super().__init__()
        
        # 初始層
        self.inc = DoubleConv(z_channels, 64 * channels_mult[0])
        
        # 上採樣路徑
        self.up_layers = nn.ModuleList()
        channels = 64 * channels_mult[0]
        for mult in channels_mult[1:]:
            out_channels_layer = 64 * mult
            self.up_layers.append(Up(channels, out_channels_layer, bilinear=False))
            channels = out_channels_layer
        
        # 最終輸出層
        self.outc = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 將輸出標準化至 [-1, 1] 區間
        )
        
    def forward(self, z):
        x = self.inc(z)
        
        # 逐層上採樣
        for layer in self.up_layers:
            x = layer(x)
        
        # 投影回圖像空間
        out = self.outc(x)
        
        return out