import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .encoder import Encoder, EncoderWithVariational
from .decoder import Decoder
from .unet import UNetWithCrossAttention
from .diffusion import DDPMScheduler, DDIMScheduler

class ConditionalLDM(nn.Module):
    """
    條件式潛在擴散模型 (Conditional Latent Diffusion Model)
    結合了VAE編碼器/解碼器和UNet去噪模型
    """
    def __init__(
        self,
        unet_dim=128,
        condition_dim=256,
        time_embedding_dim=256,
        num_classes=24,
        use_attention=True,
        image_size=64,
        channels=3,
        latent_channels=4,
        variational=True,
        training=True
    ):
        super().__init__()
        
        # 設置是否使用變分特性
        self.variational = variational
        
        # 編碼器-解碼器（用於潛在空間壓縮）
        if variational:
            self.encoder = EncoderWithVariational(
                in_channels=channels, 
                z_channels=latent_channels,
                channels_mult=(1, 2, 4, 8)
            )
        else:
            self.encoder = Encoder(
                in_channels=channels, 
                z_channels=latent_channels,
                channels_mult=(1, 2, 4, 8)
            )
            
        self.decoder = Decoder(
            z_channels=latent_channels, 
            out_channels=channels,
            channels_mult=(8, 4, 2, 1)
        )
        
        # 條件嵌入（將one-hot轉換為嵌入）
        self.condition_embedding = nn.Sequential(
            nn.Linear(num_classes, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        
        # 時間嵌入（正弦曲線 + MLP）
        self.time_embedding_dim = time_embedding_dim
        
        # UNet骨幹
        self.unet = UNetWithCrossAttention(
            in_channels=latent_channels,
            model_channels=unet_dim,
            out_channels=latent_channels,
            num_res_blocks=2,
            attention_resolutions=(8, 4),
            channel_mults=(1, 2, 4, 8),
            time_embedding_dim=time_embedding_dim,
            condition_dim=condition_dim,
            use_attention=use_attention
        )
        
        # 噪聲排程器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        # VAE KL散度的權重
        self.kl_weight = 1e-6
        
        # 是否處於訓練模式
        self.training = training
    
    def encode(self, x):
        """將圖像編碼到潛在空間"""
        if self.variational:
            z, mu, logvar = self.encoder(x)
            return z, mu, logvar
        else:
            z = self.encoder(x)
            return z
    
    def decode(self, z):
        """將潛在表示解碼回圖像空間"""
        return self.decoder(z)
    
    def forward(self, images, labels, return_loss=True):
        """
        前向傳播函數，訓練時使用
        """
        batch_size = images.shape[0]
        
        # 編碼到潛在空間
        if self.variational:
            latents, mu, logvar = self.encode(images)
            kl_loss = self._kl_loss(mu, logvar)
        else:
            latents = self.encode(images)
            kl_loss = 0.0
        
        # 採樣隨機噪聲和時間步
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=images.device).long()
        
        # 根據噪聲排程向潛在表示添加噪聲
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 條件嵌入
        condition = self.condition_embedding(labels)
        
        # 噪聲預測
        noise_pred = self.unet(noisy_latents, timesteps, condition)
        
        # 計算損失
        if return_loss:
            mse_loss = F.mse_loss(noise_pred, noise)
            loss = mse_loss + self.kl_weight * kl_loss
            return loss, mse_loss, kl_loss
        else:
            return noise_pred
    
    def _kl_loss(self, mu, logvar):
        """計算VAE的KL散度損失"""
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss
    
    def sample(
        self,
        conditions,
        evaluator=None,
        guidance_scale=3.0,
        classifier_scale=0.5,
        num_inference_steps=50,
        seed=42,
        device=None,
        return_latents=False
    ):
        """
        條件式圖像生成，使用DDIM採樣
        """
        # 設置隨機種子以確保可重現性
        torch.manual_seed(seed)
        
        # 如果未指定設備，使用模型所在的設備
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # 準備條件
        batch_size = conditions.shape[0]
        condition_emb = self.condition_embedding(conditions)
        
        # 初始化潛在表示
        latents = torch.randn(batch_size, 4, 16, 16).to(device)  # 4倍壓縮的潛在空間
        
        # DDIM採樣
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            eta=0.0  # eta=0.0 對應於 DDIM，更快的採樣
        )
        scheduler.set_timesteps(num_inference_steps)
        
        # 迭代時間步
        for t in tqdm(scheduler.timesteps):
            # 複製潛在表示用於分類器引導
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # 準備時間步嵌入
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 準備條件嵌入
            if guidance_scale > 1.0:
                # 準備無條件嵌入（零條件）
                uncond_emb = torch.zeros_like(condition_emb)
                # 連接條件和無條件嵌入
                stacked_condition = torch.cat([uncond_emb, condition_emb])
            else:
                stacked_condition = condition_emb
            
            # 預測噪聲
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, timestep.repeat(2) if guidance_scale > 1.0 else timestep, stacked_condition)
            
            # 進行引導
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 獲取前一個時間步
            prev_timestep = t - scheduler.num_train_timesteps // scheduler.timesteps.shape[0]
            
            # DDIM步驟
            latents = scheduler.step(noise_pred, t, latents, prev_timestep=prev_timestep)["prev_sample"]
            
            # 分類器引導（使用評估器提供額外引導）
            if evaluator is not None and t < 500 and t % 5 == 0:  # 在後半段每5步應用一次
                with torch.no_grad():
                    # 解碼到圖像空間
                    images = self.decode(latents)
                    # 標準化用於評估器
                    norm_images = (images + 1) / 2  # [-1, 1] -> [0, 1]
                    norm_images = (norm_images - 0.5) / 0.5  # [0, 1] -> [-1, 1] 用於評估器
                    
                    # 使用evaluator計算梯度
                    images.requires_grad = True
                    logits = evaluator.resnet18(images)
                    
                    # 計算分類器損失
                    cls_loss = -torch.sum(logits * conditions, dim=1).mean()
                    
                    # 計算梯度
                    grad = torch.autograd.grad(cls_loss, images)[0]
                    
                    # 縮放梯度並應用到潛在表示
                    grad_scale = classifier_scale * (1000 - t) / 1000  # 隨著t減小而縮小
                    
                    # 獲取編碼梯度
                    if self.variational:
                        latent_grad, _, _ = self.encoder(images + grad_scale * grad.detach())
                        latent_orig, _, _ = self.encoder(images)
                    else:
                        latent_grad = self.encoder(images + grad_scale * grad.detach())
                        latent_orig = self.encoder(images)
                    
                    # 應用到潛在表示
                    latents = latents - 0.1 * (latent_grad - latent_orig)
        
        # 解碼最終的潛在表示為圖像
        with torch.no_grad():
            images = self.decode(latents)
        
        # 標準化到 [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        
        if return_latents:
            return images, latents
        else:
            return images
    
    def visualize_denoising(
        self,
        conditions,
        num_inference_steps=50,
        seed=42,
        device=None,
        num_images=8
    ):
        """
        視覺化去噪過程
        """
        # 設置隨機種子以確保可重現性
        torch.manual_seed(seed)
        
        # 如果未指定設備，使用模型所在的設備
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # 準備條件
        batch_size = conditions.shape[0]
        condition_emb = self.condition_embedding(conditions)
        
        # 初始化潛在表示
        latents = torch.randn(batch_size, 4, 16, 16).to(device)
        
        # DDIM採樣
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            eta=0.0
        )
        scheduler.set_timesteps(num_inference_steps)
        
        # 收集去噪過程中的圖像
        process_images = []
        
        # 選擇均勻分佈的時間步用於視覺化
        vis_steps = torch.linspace(0, len(scheduler.timesteps)-1, num_images).long()
        
        # 迭代時間步
        for i, t in enumerate(tqdm(scheduler.timesteps)):
            # 準備時間步嵌入
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 預測噪聲
            with torch.no_grad():
                noise_pred = self.unet(latents, timestep, condition_emb)
            
            # 獲取前一個時間步
            prev_timestep = t - scheduler.num_train_timesteps // scheduler.timesteps.shape[0]
            
            # DDIM步驟
            latents = scheduler.step(noise_pred, t, latents, prev_timestep=prev_timestep)["prev_sample"]
            
            # 在選定的時間步收集圖像
            if i in vis_steps:
                # 解碼到圖像空間
                with torch.no_grad():
                    images = self.decode(latents)
                
                # 標準化到 [0, 1]
                images = (images + 1) / 2
                images = images.clamp(0, 1)
                
                process_images.append(images)
        
        # 將圖像沿時間維度堆疊
        process_images = torch.stack(process_images)
        
        return process_images