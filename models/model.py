import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler, AutoencoderKL

def download_vae_model(model_id="stabilityai/sd-vae-ft-mse", subfolder="vae", save_path="./pretrained_models"):
    """下載 VAE 模型到本地並返回保存路徑"""
    
    full_save_path = os.path.join(save_path, model_id.split("/")[-1])
    os.makedirs(full_save_path, exist_ok=True)
    
    print(f"Downloading VAE model from {model_id}/...")
    
    try:
        # 下載模型
        vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        
        # 保存到本地
        vae.save_pretrained(full_save_path)
        print(f"Model successfully downloaded and saved to {full_save_path}")
        
        return full_save_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

class ConditionalDiffusionModel(nn.Module):
    """
    條件式擴散模型，使用 diffusers 庫和 SD 預訓練 VAE
    """
    def __init__(
        self,
        num_classes=24,
        unet_in_channels=4,
        unet_sample_size=16,  # 默認潛在空間大小，64/4=16
        condition_embedding_dim=768,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_pretrain_vae=False,
        vae_model_id="stabilityai/sd-vae-ft-mse",
        vae_local_path="./pretrained_models",
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.condition_embedding_dim = condition_embedding_dim
        
        # 條件嵌入層 - 將 one-hot 轉為 embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(num_classes, condition_embedding_dim // 2),
            nn.LayerNorm(condition_embedding_dim // 2),
            nn.GELU(),
            nn.Linear(condition_embedding_dim // 2, condition_embedding_dim),
            nn.LayerNorm(condition_embedding_dim),
        ).to(device)

        vae_loaded = False
        if use_pretrain_vae:
            # VAE加載邏輯：優先嘗試從本地加載
            print(f"嘗試加載 VAE 模型到設備: {device}")
            
            # 構建可能的本地模型路徑
            local_model_path = os.path.join(vae_local_path, vae_model_id.split("/")[-1])
            
            # 1. 首先嘗試從指定的本地路徑加載
            if os.path.exists(local_model_path):
                try:
                    print(f"從本地路徑加載 VAE: {local_model_path}")
                    self.vae = AutoencoderKL.from_pretrained(
                        local_model_path,
                        torch_dtype=torch.float32,
                    ).to(device)
                    vae_loaded = True
                    print("成功從本地加載 VAE 模型")
                except Exception as e:
                    print(f"從本地加載 VAE 失敗: {e}")
            
            # 2. 如果本地加載失敗，嘗試從網絡加載
            if not vae_loaded:
                try:
                    print(f"從網絡加載 VAE: {vae_model_id}")
                    self.vae = AutoencoderKL.from_pretrained(
                        vae_model_id,
                        torch_dtype=torch.float32,
                    ).to(device)
                    vae_loaded = True
                    print("成功從網絡加載 VAE 模型")
                    
                    # 順便保存到本地以備未來使用
                    try:
                        os.makedirs(local_model_path, exist_ok=True)
                        self.vae.save_pretrained(local_model_path)
                        print(f"已將 VAE 模型保存到: {local_model_path}")
                    except Exception as e:
                        print(f"保存 VAE 模型到本地時出錯: {e}")
                except Exception as e:
                    print(f"從網絡加載 VAE 失敗: {e}")
        
        # 3. 如果還是加載失敗，使用簡單的 VAE 初始化
        if not vae_loaded or use_pretrain_vae is False:
            print("使用隨機初始化的 VAE")
            self.vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
                up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
                block_out_channels=(128, 256, 512, 512),
                # down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),  # 減少一層
                # up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),  # 減少一層
                # block_out_channels=(64, 128, 256),  # 降低通道數
                layers_per_block=2,
                act_fn="silu",
                latent_channels=unet_in_channels,
                sample_size=unet_sample_size * 2,
            ).to(device)
            self.vae_scale_factor = 0.5  # VAE 的縮放因子
            for param in self.vae.parameters():
                param.requires_grad = True
        else:
            self.vae_scale_factor = 0.18215  # SD VAE 的縮放因子
            for param in self.vae.parameters():
                if param.device != device:
                    print(f"警告: VAE 參數在錯誤設備上: {param.device}")
                    param.data = param.data.to(device)
                param.requires_grad = False # 預訓練VAE保持凍結
        
        # 設置UNet - 條件式去噪模型的核心
        self.unet = UNet2DConditionModel(
            sample_size=unet_sample_size,
            in_channels=unet_in_channels,
            out_channels=unet_in_channels,
            layers_per_block=2,
            # block_out_channels=(128, 256, 512, 512),
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            # block_out_channels=(64, 128, 256),
            # down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            # up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=condition_embedding_dim,
            attention_head_dim=16,  # 增加注意力頭維度
            dropout=0.1,  # 添加適量的dropout
        ).to(device)
        
        # 噪聲排程器 - 用於訓練
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            prediction_type="epsilon"
        )
    
    def encode(self, images):
        """
        使用 VAE 將圖像編碼到潛在空間
        """
        # 確保圖像在正確的設備上
        images = images.to(self.device)
        
        # 縮放到 [-1, 1]
        if images.min() >= 0 and images.max() <= 1:
            images = 2 * images - 1
            
        # 使用上下文管理器避免計算梯度
        with torch.no_grad():
            # 明確設置 VAE 評估模式
            self.vae.eval()
            # 確保 VAE 在正確設備上
            self.vae = self.vae.to(self.device)
            
            # VAE 編碼
            latents = self.vae.encode(images).latent_dist.sample()
            # 使用 SD VAE 的縮放因子
            latents = latents * self.vae_scale_factor
        
        return latents
    
    def decode(self, latents):
        """
        使用 VAE 將潛在表示解碼回圖像
        """
        # 確保潛在表示在正確的設備上
        latents = latents.to(self.device)
        
        # 使用上下文管理器避免計算梯度
        with torch.no_grad():
            # 明確設置 VAE 評估模式
            self.vae.eval()
            # 確保 VAE 在正確設備上
            self.vae = self.vae.to(self.device)
            
            # 使用 SD VAE 的反向縮放
            latents = latents / self.vae_scale_factor
            
            # VAE 解碼
            images = self.vae.decode(latents).sample
        
        # 調整到 [0, 1] 範圍
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        return images
            
    def prepare_condition_embedding(self, labels):
        """
        準備條件嵌入，將 one-hot 標籤轉換為模型可用的嵌入
        """
        # 確保標籤在正確的設備上
        labels = labels.to(self.device)
        
        # 條件嵌入
        condition_embeddings = self.condition_embedding(labels)

        # 添加噪聲以增加多樣性
        if self.training:
            noise = torch.randn_like(condition_embeddings) * 0.1
            condition_embeddings = condition_embeddings + noise
        
        # 轉換為適合 cross-attention 格式的張量
        # 模型期望的格式為 [batch_size, sequence_length=1, hidden_size]
        condition_embeddings = condition_embeddings.unsqueeze(1)
        
        return condition_embeddings
        
    def forward(self, images, labels):
        """
        前向傳播函數，用於訓練
        
        Args:
            images: 批次圖像 [B, C, H, W]
            labels: one-hot 標籤 [B, num_classes]
            
        Returns:
            loss: 噪聲預測損失
        """
        # 確保輸入在正確的設備上
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        batch_size = images.shape[0]
        
        # 編碼到潛在空間
        latents = self.encode(images)
        
        # 加噪
        noise = torch.randn_like(latents, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 條件嵌入
        encoder_hidden_states = self.prepare_condition_embedding(labels)
        
        # 通過 UNet 預測噪聲
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # 計算損失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def sample(
        self,
        conditions,
        evaluator=None,
        guidance_scale=3.0,
        classifier_scale=0.5,
        num_inference_steps=50,
        generator=None,
    ):
        """
        根據條件生成圖像
        
        Args:
            conditions: one-hot 標籤 [B, num_classes]
            evaluator: 評估器，用於分類器引導
            guidance_scale: 無條件引導的強度
            classifier_scale: 分類器引導的強度
            num_inference_steps: 推理步數
            generator: 隨機生成器
            
        Returns:
            images: 生成的圖像 [B, C, H, W]
        """
        self.eval()
        
        # 確保條件在正確的設備上
        conditions = conditions.to(self.device)
        
        batch_size = conditions.shape[0]
        
        # 準備條件嵌入
        encoder_hidden_states = self.prepare_condition_embedding(conditions)
        
        # 如果使用無條件引導，還需要空白（無條件）嵌入
        if guidance_scale > 1.0:
            uncond_embedding = torch.zeros_like(conditions, device=self.device)
            uncond_encoder_hidden_states = self.prepare_condition_embedding(uncond_embedding)
            
            # 連接條件與無條件嵌入
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states])
        
        # 創建採樣排程器
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 初始化潛在向量
        latents_shape = (batch_size, self.unet.config.in_channels, 
                         self.unet.config.sample_size, self.unet.config.sample_size)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
        )
        
        # 迭代時間步進行去噪
        for i, t in enumerate(tqdm(scheduler.timesteps)):
            # 如果使用無條件引導，需要複製潛在向量
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
                # 創建正確維度和設備的timestep批次
                timestep_batch = torch.full((latent_model_input.shape[0],), t, device=self.device)
            else:
                latent_model_input = latents
                timestep_batch = torch.full((latents.shape[0],), t, device=self.device)
                
            # 預測噪聲
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    timestep_batch,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
            # 執行無條件引導
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 更新潛在向量
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # 分類器引導（如果提供了評估器）
            if evaluator is not None and classifier_scale > 0 and t < 500 and t % 5 == 0:
                try:
                    # 解碼到圖像空間
                    with torch.no_grad():
                        decoded_images = self.decode(latents)
                        
                    # 標準化用於評估器
                    norm_images = (decoded_images - 0.5) / 0.5
                    
                    # 使用更簡單的引導方法
                    with torch.enable_grad():
                        norm_images.requires_grad_(True)
                        
                        # 計算分類器分數
                        logits = evaluator.resnet18(norm_images)
                        score = torch.sum(logits * conditions, dim=1).mean()
                        
                        # 計算梯度
                        grad = torch.autograd.grad(score, norm_images)[0]
                        
                        # 應用梯度，但使用小步長
                        scale = classifier_scale * 0.1 * (1000 - t.item()) / 1000
                        grad_images = norm_images + scale * grad.sign()  # 使用梯度符號而非精確值
                        
                        # 重新編碼回潛在空間
                        grad_latents = self.encode(grad_images * 0.5 + 0.5)  # 轉回[0,1]範圍
                        
                        # 向正確方向調整潛在表示
                        latents = latents + 0.05 * (grad_latents - latents)
                except Exception as e:
                    print(f"分類器引導時出錯 (t={t}): {e}")
                    # 繼續執行，忽略此步的分類器引導
        
        # 解碼最終的潛在表示為圖像
        with torch.no_grad():
            images = self.decode(latents)
        
        return images
        
    def visualize_denoising(
        self,
        conditions,
        num_inference_steps=50,
        num_images=8,
        generator=None,
    ):
        """
        生成去噪過程的可視化
        
        Args:
            conditions: one-hot 標籤 [1, num_classes]
            num_inference_steps: 推理步數
            num_images: 要保存的中間結果數量
            generator: 隨機生成器
            
        Returns:
            process_images: 去噪過程中的圖像 [num_images, 1, C, H, W]
        """
        self.eval()
        
        # 確保條件在正確的設備上
        conditions = conditions.to(self.device)
        
        batch_size = conditions.shape[0]
        
        # 準備條件嵌入
        encoder_hidden_states = self.prepare_condition_embedding(conditions)
        
        # 創建採樣排程器
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 初始化潛在向量
        latents_shape = (batch_size, self.unet.config.in_channels, 
                         self.unet.config.sample_size, self.unet.config.sample_size)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
        )
        
        # 收集去噪過程中的圖像
        process_images = []
        
        # 選擇均勻分佈的時間步用於視覺化
        vis_indices = torch.linspace(0, len(scheduler.timesteps)-1, num_images).long()
        
        # 迭代時間步進行去噪
        for i, t in enumerate(tqdm(scheduler.timesteps)):
            # 預測噪聲
            with torch.no_grad():
                # 創建正確維度和設備的timestep批次
                timestep_batch = torch.full((batch_size,), t, device=self.device)
                
                noise_pred = self.unet(
                    latents,
                    timestep_batch,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
            # 更新潛在向量
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # 在選定的時間步收集圖像
            if i in vis_indices:
                # 解碼到圖像空間
                with torch.no_grad():
                    images = self.decode(latents)
                
                process_images.append(images)
        
        # 將圖像沿時間維度堆疊
        process_images = torch.stack(process_images)
        
        return process_images


# 如果需要單獨執行下載功能
if __name__ == "__main__":
    # 下載並保存 VAE 模型
    download_vae_model()