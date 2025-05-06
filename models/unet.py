import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    使用正弦曲線作為時間步的位置嵌入
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # 處理維度為奇數的情況
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:,:1])], dim=-1)
            
        return embeddings

class ResBlock(nn.Module):
    """殘差區塊，支援時間條件"""
    def __init__(self, in_ch, out_ch, time_emb_dim=None, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb.reshape(time_emb.shape[0], -1, 1, 1)
            
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """注意力區塊"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 標準化
        x_norm = self.norm(x)
        
        # 計算Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 重塑為多頭格式
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)
        
        # 計算注意力
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # 應用注意力得到輸出
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        
        # 重塑回原始形狀
        out = out.reshape(B, C, H, W)
        
        # 投影並應用殘差連接
        return x + self.proj(out)

class CrossAttentionBlock(nn.Module):
    """交叉注意力區塊，用於注入條件信息"""
    def __init__(self, channels, cond_dim, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Linear(cond_dim, channels)
        self.v = nn.Linear(cond_dim, channels)
        
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, cond):
        B, C, H, W = x.shape
        
        # 標準化
        x_norm = self.norm(x)
        
        # 計算Q, K, V
        q = self.q(x_norm).reshape(B, self.num_heads, self.head_dim, H * W)
        k = self.k(cond).reshape(B, self.num_heads, self.head_dim, -1)
        v = self.v(cond).reshape(B, self.num_heads, self.head_dim, -1)
        
        # 計算注意力
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # 應用注意力得到輸出
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        
        # 重塑回原始形狀
        out = out.reshape(B, C, H, W)
        
        # 投影並應用殘差連接
        return x + self.proj(out)

class Downsample(nn.Module):
    """下採樣層"""
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    """上採樣層"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        # 使用近鄰插值而非雙線性插值，以避免尺寸不匹配問題
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class UNet(nn.Module):
    """
    簡化版UNet，專為穩定擴散設計
    """
    def __init__(
        self,
        in_channels=4,             # 輸入通道數
        model_channels=128,        # 基礎通道數
        out_channels=4,            # 輸出通道數
        num_res_blocks=2,          # 每個解析度的殘差塊數量
        attention_resolutions=(8, 4),  # 使用注意力的解析度
        dropout=0.0,              # Dropout比率
        channel_mult=(1, 2, 4, 8),  # 各層級通道乘數
        time_embedding_dim=256,    # 時間嵌入維度
        condition_dim=256         # 條件維度
    ):
        super().__init__()
        
        time_embed_dim = time_embedding_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.has_condition = condition_dim > 0
        
        # 輸入卷積
        self.input_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        
        # 下採樣部分
        input_block_chans = [model_channels]
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, model_channels * mult, time_embed_dim, dropout)]
                ch = model_channels * mult
                
                # 在特定解析度添加注意力層
                if mult * model_channels in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                    if self.has_condition:
                        layers.append(CrossAttentionBlock(ch, condition_dim))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            
            # 除了最後一層，都添加下採樣
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Sequential(Downsample(ch)))
                input_block_chans.append(ch)
        
        # 中間部分
        self.middle_block = nn.Sequential(
            ResBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch),
            *([] if not self.has_condition else [CrossAttentionBlock(ch, condition_dim)]),
            ResBlock(ch, ch, time_embed_dim, dropout)
        )
        
        # 上採樣部分
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                
                # 在特定解析度添加注意力層
                if mult * model_channels in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                    if self.has_condition:
                        layers.append(CrossAttentionBlock(ch, condition_dim))
                
                # 除了最後一層，都添加上採樣
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # 最終輸出層
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps, condition=None):
        """
        前向傳播函數
        
        Args:
            x (torch.Tensor): 輸入張量，形狀為 [B, C, H, W]
            timesteps (torch.Tensor): 時間步，形狀為 [B]
            condition (torch.Tensor, optional): 條件嵌入，形狀為 [B, cond_dim]
            
        Returns:
            torch.Tensor: 預測的噪聲或原始樣本
        """
        # 時間嵌入
        emb = self.time_embed(timesteps)
        
        # 初始特徵
        h = self.input_blocks[0](x)
        hs = [h]
        
        # 下採樣路徑
        for module in self.input_blocks[1:]:
            h = module(h) if not isinstance(module[0], ResBlock) else module[0](h, emb)
            for layer in module[1:]:
                if isinstance(layer, CrossAttentionBlock) and condition is not None:
                    h = layer(h, condition)
                else:
                    h = layer(h)
            hs.append(h)
        
        # 中間部分
        h = self.middle_block[0](h, emb)
        for layer in self.middle_block[1:]:
            if isinstance(layer, CrossAttentionBlock) and condition is not None:
                h = layer(h, condition)
            else:
                h = layer(h)
        
        # 上採樣路徑
        for module in self.output_blocks:
            # 連接跳躍連接
            h = torch.cat([h, hs.pop()], dim=1)
            
            h = module[0](h, emb)
            for layer in module[1:]:
                if isinstance(layer, CrossAttentionBlock) and condition is not None:
                    h = layer(h, condition)
                else:
                    h = layer(h)
        
        # 最終輸出
        return self.out(h)