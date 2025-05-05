import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class DDPMScheduler:
    """
    DDPM 噪聲排程器，處理前向過程（增加噪聲）和反向過程（去噪）
    """
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        prediction_type="epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        # 設置 beta 排程
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            # 使用余弦排程
            steps = num_train_timesteps + 1
            t = torch.linspace(0, num_train_timesteps, steps) / num_train_timesteps
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 計算相關參數
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 計算前向過程相關的係數
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 計算採樣相關的係數
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod.roll(1, 0)) / (1.0 - self.alphas_cumprod)
        self.posterior_variance[0] = self.posterior_variance[1]
        self.posterior_log_variance = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod.roll(1, 0)) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod.roll(1, 0)) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        # 設置時間步的轉換係數
        self.timesteps = torch.arange(0, num_train_timesteps).float()[::-1].clone()
    
    def set_timesteps(self, num_inference_steps, device="cpu"):
        """
        設置推理時使用的時間步
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.to(device).long()
        self.timesteps = timesteps
        
        return timesteps
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        根據時間步 t 添加噪聲到原始樣本
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # 添加適當的維度以支持批處理
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.reshape(-1, 1, 1, 1)
        
        # 噪聲添加公式
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples
    
    def _get_mean_and_variance(self, x_0, x_t, t):
        """
        計算後驗分佈 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * x_0 +
            self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        posterior_log_variance = self.posterior_log_variance[t].reshape(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def step(self, model_output, timestep, sample, prev_timestep=None, return_dict=True):
        """
        DDPM採樣步驟
        """
        if prev_timestep is None:
            prev_timestep = timestep - 1
        
        # 用於數值穩定性的夾值
        prev_timestep = torch.max(torch.tensor(0), prev_timestep)
        
        # 1. 根據模型輸出類型計算前一步的預測樣本x_0
        if self.prediction_type == "epsilon":
            # 如果預測噪聲 (epsilon)
            alpha_prod_t = self.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            
            if timestep > 0:
                noise = model_output
            else:
                noise = 0.0
            
            pred_original_sample = (sample - beta_prod_t.sqrt() * noise) / alpha_prod_t.sqrt()
            
            if self.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        elif self.prediction_type == "sample":
            # 如果預測樣本 x_0
            pred_original_sample = model_output
            
            if self.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
                
        elif self.prediction_type == "v_prediction":
            # 如果預測速度 v
            alpha_prod_t = self.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            
            model_output_denom = alpha_prod_t.sqrt() * beta_prod_t.sqrt()
            pred_original_sample = (alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output) / model_output_denom
            
            if self.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        else:
            raise ValueError(f"prediction_type {self.prediction_type} not supported.")
        
        # 2. 獲取後驗參數 q(x_{t-1} | x_t, x_0)
        posterior_mean, posterior_variance, posterior_log_variance = self._get_mean_and_variance(
            pred_original_sample, sample, timestep
        )
        
        # 3. 計算x_{t-1}的均值
        # 如果 t = 0，直接取 posterior_mean
        if timestep > 0:
            noise = torch.randn_like(model_output)
            prev_sample = posterior_mean + posterior_variance.sqrt() * noise
        else:
            noise = 0.0
            prev_sample = posterior_mean
        
        # 封裝返回結果
        if return_dict:
            return {
                "prev_sample": prev_sample,
                "pred_original_sample": pred_original_sample,
                "posterior_mean": posterior_mean,
                "posterior_variance": posterior_variance,
                "posterior_log_variance": posterior_log_variance,
                "noise": noise,
            }
        else:
            return prev_sample

class DDIMScheduler:
    """
    DDIM 採樣排程器，更快速的去噪過程
    """
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=True,
        prediction_type="epsilon",
        eta=0.0,  # eta=0.0 對應於DDIM, eta=1.0 對應於DDPM
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.eta = eta
        
        # 設置 beta 排程
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            # 使用余弦排程
            steps = num_train_timesteps + 1
            t = torch.linspace(0, num_train_timesteps, steps) / num_train_timesteps
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 計算相關參數
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 計算前向過程相關的係數
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 初始化時間步
        self.timesteps = torch.arange(0, num_train_timesteps).float()[::-1].clone()
    
    def set_timesteps(self, num_inference_steps, device="cpu"):
        """
        設置推理時使用的時間步
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = self.num_train_timesteps - timesteps - 1
        timesteps = timesteps.to(device).long()
        self.timesteps = timesteps
        
        return timesteps
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        根據時間步 t 添加噪聲到原始樣本
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # 添加適當的維度以支持批處理
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.reshape(-1, 1, 1, 1)
        
        # 噪聲添加公式
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples
    
    def step(self, model_output, timestep, sample, prev_timestep=None, return_dict=True):
        """
        DDIM採樣步驟
        """
        if prev_timestep is None:
            prev_timestep = timestep - 1
        
        # 用於數值穩定性的夾值
        prev_timestep = torch.max(torch.tensor(0), prev_timestep)
        
        # 獲取 alpha 相關參數
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod[prev_timestep]
        
        # 根據預測類型獲取x_0
        if self.prediction_type == "epsilon":
            # 從噪聲預測獲取原始樣本
            pred_original_sample = (sample - torch.sqrt(1 - alpha_cumprod_t) * model_output) / torch.sqrt(alpha_cumprod_t)
            
            if self.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
                
        elif self.prediction_type == "sample":
            # 直接使用模型輸出作為x_0
            pred_original_sample = model_output
            
            if self.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
                
        elif self.prediction_type == "v_prediction":
            # 從速度預測獲取原始樣本
            pred_original_sample = alpha_cumprod_t.sqrt() * sample - (1 - alpha_cumprod_t).sqrt() * model_output
            
            if self.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
                
        else:
            raise ValueError(f"prediction_type {self.prediction_type} not supported.")
        
        # 計算方差
        variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)
        
        # 生成噪聲
        std_dev = torch.sqrt(variance) * self.eta
        
        # 計算無噪聲的均值部分
        pred_mean = torch.sqrt(alpha_cumprod_prev) * (
            (sample - torch.sqrt(alpha_cumprod_t) * pred_original_sample) / torch.sqrt(1 - alpha_cumprod_t)
        ) * torch.sqrt(1 - alpha_cumprod_prev) / torch.sqrt(1 - alpha_cumprod_t) + torch.sqrt(alpha_cumprod_prev) * pred_original_sample
        
        # 生成新樣本
        noise = torch.randn_like(model_output) if self.eta > 0 else 0.0
        prev_sample = pred_mean + std_dev * noise
        
        # 封裝返回結果
        if return_dict:
            return {
                "prev_sample": prev_sample,
                "pred_original_sample": pred_original_sample,
            }
        else:
            return prev_sample