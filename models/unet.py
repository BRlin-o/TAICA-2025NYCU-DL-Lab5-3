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

class ResidualBlock(nn.Module):
    """
    殘差塊，用於UNet的下採樣和上採樣路徑
    """
    def __init__(self, in_channels, out_channels, time_dim=None, *, kernel_size=3, stride=1, padding=1, groups=8):
        super().__init__()
        self.time_dim = time_dim
        
        # 時間嵌入的處理
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_dim, out_channels)
            )
        
        # 主要卷積路徑
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.GELU()
        
        # 殘差連接
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb=None):
        # 主路徑
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # 添加時間嵌入
        if self.time_dim is not None and time_emb is not None:
            time_token = self.time_mlp(time_emb)
            h = h + time_token.reshape(time_token.shape[0], -1, 1, 1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        # 殘差連接
        return h + self.residual(x)

class CrossAttention(nn.Module):
    """
    交叉注意力模組，用於UNet中注入條件信息
    """
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(self, x, context=None):
        batch_size, c, h, w = x.shape
        
        # 重塑張量為注意力操作格式：[batch, h*w, channels]
        q_input = x.permute(0, 2, 3, 1).reshape(batch_size, h * w, c)
        q_input = self.norm(q_input)
        
        # 如果沒有提供context，使用自注意力
        context = context if context is not None else q_input
        
        # 產生查詢、鍵、值
        q = self.to_q(q_input)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 重塑為多頭格式
        q = q.reshape(batch_size, -1, self.heads, q.shape[-1] // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.heads, k.shape[-1] // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.heads, v.shape[-1] // self.heads).permute(0, 2, 1, 3)
        
        # 注意力操作
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # 重塑回原始格式
        out = out.permute(0, 2, 1, 3).reshape(batch_size, h * w, -1)
        out = self.to_out(out)
        
        # 添加殘差連接並重塑回空間格式 [batch, channel, height, width]
        out = out + q_input
        out = out.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        
        return out

class UNetWithCrossAttention(nn.Module):
    """
    帶有交叉注意力的UNet模型，用於條件式擴散模型
    """
    def __init__(
        self,
        in_channels=4,
        model_channels=128,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(8, 4),
        channel_mults=(1, 2, 4, 8),
        dropout=0.0,
        time_embedding_dim=256,
        condition_dim=256,
        use_checkpoint=False,
        use_attention=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.use_checkpoint = use_checkpoint
        self.use_attention = use_attention
        self.time_embedding_dim = time_embedding_dim
        self.condition_dim = condition_dim
        
        # 時間嵌入
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embedding_dim),
            nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        
        # 條件嵌入
        self.condition_embedding = nn.Sequential(
            nn.Linear(24, condition_dim),  # 假設輸入為24維的one-hot向量
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        
        # 初始卷積層
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 下採樣路徑
        self.down_blocks = nn.ModuleList()
        
        # 記錄每個解析度的通道數
        channels = [model_channels]
        
        # 當前通道數
        current_channels = model_channels
        
        # 下採樣塊
        for i, mult in enumerate(channel_mults):
            out_channels = model_channels * mult
            
            # 每個解析度有num_res_blocks個殘差塊
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(
                        current_channels, 
                        out_channels,
                        time_dim=time_embedding_dim
                    )
                )
                current_channels = out_channels
                channels.append(current_channels)
                
                # 在指定的解析度添加注意力機制
                if self.use_attention and i in attention_resolutions:
                    self.down_blocks.append(
                        CrossAttention(
                            current_channels, 
                            condition_dim,
                            heads=8,
                            dim_head=64,
                            dropout=dropout
                        )
                    )
                
            # 如果不是最後一個下採樣塊，添加下採樣層
            if i != len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv2d(current_channels, current_channels, 4, 2, 1))
        
        # 中間塊 - 底部的塊
        self.middle_block1 = ResidualBlock(current_channels, current_channels, time_embedding_dim)
        self.middle_attn = CrossAttention(current_channels, condition_dim) if use_attention else nn.Identity()
        self.middle_block2 = ResidualBlock(current_channels, current_channels, time_embedding_dim)
        
        # 上採樣路徑
        self.up_blocks = nn.ModuleList()
        
        # 上採樣塊，注意跳躍連接
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = model_channels * mult
            
            # 每個解析度有num_res_blocks個殘差塊，再加上跳躍連接
            for j in range(num_res_blocks + 1):
                # 使用跳躍連接
                skip_channels = channels.pop() if j > 0 else 0
                self.up_blocks.append(
                    ResidualBlock(
                        current_channels + skip_channels,
                        out_channels,
                        time_dim=time_embedding_dim
                    )
                )
                current_channels = out_channels
                
                # 在指定的解析度添加注意力機制
                if self.use_attention and i in attention_resolutions:
                    self.up_blocks.append(
                        CrossAttention(
                            current_channels, 
                            condition_dim,
                            heads=8,
                            dim_head=64,
                            dropout=dropout
                        )
                    )
                
            # 如果不是最後一個上採樣塊，添加上採樣層
            if i > 0:
                self.up_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(current_channels, current_channels, 3, padding=1)
                    )
                )
        
        # 最終輸出層
        self.out = nn.Sequential(
            nn.GroupNorm(8, current_channels),
            nn.GELU(),
            nn.Conv2d(current_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, time, condition):
        # 時間嵌入
        time_emb = self.time_embedding(time)
        
        # 條件嵌入
        if condition.shape[-1] != self.condition_dim:
            cond_emb = self.condition_embedding(condition)
        else:
            cond_emb = condition
        
        # 初始特徵提取
        h = self.input_conv(x)
        
        # 儲存中間特徵用於跳躍連接
        hs = [h]
        
        # 下採樣路徑
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb)
            elif isinstance(module, CrossAttention):
                h = module(h, cond_emb)
            else:
                # 下採樣層
                h = module(h)
            hs.append(h)
        
        # 中間塊
        h = self.middle_block1(h, time_emb)
        h = self.middle_attn(h, cond_emb)
        h = self.middle_block2(h, time_emb)
        
        # 上採樣路徑
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                # 使用跳躍連接
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)
            elif isinstance(module, CrossAttention):
                h = module(h, cond_emb)
            else:
                # 上採樣層
                h = module(h)
        
        # 最終輸出
        return self.out(h)

    def enable_gradient_checkpointing(self):
        """啟用梯度檢查點以節省顯存"""
        self.use_checkpoint = True