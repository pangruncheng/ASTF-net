from comet_ml import start
from comet_ml.integration.pytorch import log_model
import torch
import h5py
import numpy as np
from obspy import read
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import gc
import torch.optim as optim
from tqdm import tqdm
import random

####################################noise、dropout、trend#####################################
class SeismicDatasetHDF5(Dataset):
    def __init__(self, hdf5_file, enable_noise=True, enable_dropout=True, enable_trend=True, max_noise_level=0.1):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as hf:
            self.target_waveforms = hf['target_waveforms'][:]
            self.egfs = hf['egfs'][:]
            self.astfs = hf['astfs'][:]
        self.enable_noise = enable_noise
        self.enable_dropout = enable_dropout
        self.enable_trend = enable_trend
        self.max_noise_level = max_noise_level

    def __len__(self):
        return len(self.target_waveforms)

    def __getitem__(self, idx):
        target_waveform = self.target_waveforms[idx]
        egf = self.egfs[idx]
        astf = self.astfs[idx]

        # 默认值
        target_noise_level = 0.0
        egf_noise_level = 0.0

        # 先定义默认无掉点和无趋势
        target_drop_region = None
        egf_drop_region = None
        target_trend_deg = 0.0
        egf_trend_deg = 0.0

        if self.enable_noise:
            target_noise_level = np.random.uniform(0, self.max_noise_level)
            target_waveform, target_drop_region, target_trend_deg = self._augment_waveform(target_waveform, target_noise_level)

            egf_noise_level = np.random.uniform(0, self.max_noise_level)
            egf, egf_drop_region, egf_trend_deg = self._augment_waveform(egf, egf_noise_level)

        return {
            'target': torch.tensor(target_waveform, dtype=torch.float32),
            'egf': torch.tensor(egf, dtype=torch.float32),
            'astf': torch.tensor(astf, dtype=torch.float32),
            'meta': {
                'target_noise_level': target_noise_level,
                'egf_noise_level': egf_noise_level,
                'target_dropout': target_drop_region,
                'egf_dropout': egf_drop_region,
                'target_trend_deg': target_trend_deg,
                'egf_trend_deg': egf_trend_deg,
            }
        }


    def _augment_waveform(self, waveform, noise_level):
        # 加噪声
        if self.enable_noise:
            noise = np.random.normal(0, noise_level * np.max(np.abs(waveform)), size=waveform.shape)
            waveform = waveform + noise

        # 加掉点
        drop_region = None
        if self.enable_dropout:
            drop_start = np.random.randint(0, len(waveform) // 2)
            drop_end = drop_start + np.random.randint(0, 10)
            waveform[drop_start:drop_end] = 0
            drop_region = (drop_start, drop_end)

        # 加趋势，使用角度±5度限制
        trend_deg = 0.0
        if self.enable_trend:
            # 随机选取一个角度，范围-5到5度
            trend_deg = np.random.uniform(-2, 2)

            # 波形长度作为x轴长度，计算对应的斜率（tan(angle)）
            slope = np.tan(np.deg2rad(trend_deg))

            # 生成趋势线 y = slope * x，x从0到len(waveform)-1
            x = np.arange(len(waveform))
            trend = slope * x

            # 为了让趋势线的振幅与波形振幅级别相近，可以按波形最大绝对值归一化趋势幅度
            # 例如令趋势线最大值与波形最大绝对值相当
            max_abs = np.max(np.abs(waveform))
            if max_abs > 0:
                # 归一化趋势线振幅
                trend = trend / np.max(np.abs(trend)) * max_abs

            waveform = waveform + trend

        return waveform, drop_region, trend_deg

    
def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        if key == 'meta':
            collated[key] = [b[key] for b in batch]
        else:
            collated[key] = torch.stack([b[key] for b in batch])
    return collated


def get_dataloader_from_hdf5(hdf5_file, batch_size, enable_noise=True, enable_dropout=True, enable_trend=True, max_noise_level=0.1):
    if not (0 <= max_noise_level <= 0.1):
        raise ValueError("噪声比例必须在0到0.1之间")

    dataset = SeismicDatasetHDF5(
        hdf5_file,
        enable_noise=enable_noise,
        enable_dropout=enable_dropout,
        enable_trend=enable_trend,
        max_noise_level=max_noise_level
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=16, pin_memory=True, collate_fn=custom_collate)

class SimpleCNN(nn.Module):
     def __init__(self, in_channels=2, output_length=501):
         """
         一个简单的 CNN 网络，用于反卷积学习任务。

         Args:
             in_channels (int): 输入的通道数（例如，目标波形和EGF的通道数为2）。
             output_length (int): 输出序列的长度（例如，震源时间函数的长度）。
         """
         super(SimpleCNN, self).__init__()

         # 特征提取模块
         self.feature_extractor = nn.Sequential(
             nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),  # 输出通道数 32，卷积核大小 3
             nn.ReLU(),
             nn.MaxPool1d(kernel_size=2),  # 池化，序列长度减半

             nn.Conv1d(32, 64, kernel_size=3, padding=1),  # 输出通道数 64
             nn.ReLU(),
             nn.MaxPool1d(kernel_size=2),  # 再次池化，序列长度再减半

             nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 输出通道数 128
             nn.ReLU(),
             nn.MaxPool1d(kernel_size=2)  # 最后一次池化
         )

         # 全连接模块
         self.classifier = nn.Sequential(
             nn.Flatten(),  # 展平特征
             nn.Linear(4096, 1024),  # 假设经过 3 次池化后序列长度减为 output_length // 8
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(1024, output_length),  # 输出为震源时间函数的长度
             nn.Softplus()  # 用于确保输出非负
         )

     def forward(self, target_waveform, egf):
         """
         前向传播函数。

         Args:
             target_waveform (torch.Tensor): 目标波形 (batch_size, seq_len)。
             egf (torch.Tensor): 经验格林函数 (batch_size, seq_len)。

         Returns:
             torch.Tensor: 预测的震源时间函数 (batch_size, output_length)。
         """
         # 将 target_waveform 和 egf 拼接为 (batch_size, 2, seq_len)
         x = torch.stack([target_waveform, egf], dim=1)  # 在通道维度拼接
         x = self.feature_extractor(x)  # 特征提取
         x = self.classifier(x)  # 全连接分类器
         return x

class EnhancedCNN(nn.Module):
    def __init__(self, in_channels=2, output_length=501):
        """
        增加层数的 CNN 网络，用于反卷积学习任务。

        Args:
            in_channels (int): 输入的通道数（例如，目标波形和EGF的通道数为2）。
            output_length (int): 输出序列的长度（例如，震源时间函数的长度）。
        """
        super(EnhancedCNN, self).__init__()

        # 特征提取模块
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),  # 输出通道数 32，卷积核大小 3
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 池化，序列长度减半

            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # 输出通道数 64
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 再次池化，序列长度再减半

            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # 输出通道数 128
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 最后一次池化

            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # 增加的卷积层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 增加的池化层，继续减小序列长度

            nn.Conv1d(256, 512, kernel_size=3, padding=1),  # 增加的卷积层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 再次池化
        )

        # 全连接模块
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平特征
            nn.Linear(4096, 1024),  # 增加的全连接层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_length),  # 输出为震源时间函数的长度
            nn.Softplus()  # 用于确保输出非负
            # nn.ReLU(inplace=True)
        )

    def forward(self, target_waveform, egf):
        """
        前向传播函数。

        Args:
            target_waveform (torch.Tensor): 目标波形 (batch_size, seq_len)。
            egf (torch.Tensor): 经验格林函数 (batch_size, seq_len)。

        Returns:
            torch.Tensor: 预测的震源时间函数 (batch_size, output_length)。
        """
        # 将 target_waveform 和 egf 拼接为 (batch_size, 2, seq_len)
        x = torch.stack([target_waveform, egf], dim=1)  # 在通道维度拼接
        x = self.feature_extractor(x)  # 特征提取
        x = self.classifier(x)  # 全连接分类器
        return x

class EnhancedCNN_Modified(nn.Module):
    def __init__(self, in_channels=2, output_length=501):
        """
        改进版 EnhancedCNN：防止前移+补零带来的信息丢失和过拟合。
        """
        super(EnhancedCNN_Modified, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 256 → 128

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 128 → 64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 64 → 32

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 32 → 16
            # 注意：没有第5层池化，保留更多有效信息
        )

        # 使用AdaptiveAvgPool进行全局特征压缩，防止flatten稀疏问题
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 输出维度: (batch, channels, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (batch, 256)
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_length),
            nn.Softplus()  # 用于确保输出非负
            # 不使用Softplus，loss中mask padding
        )

    def forward(self, target_waveform, egf):
        """
        Args:
            target_waveform: (B, L)
            egf: (B, L)

        Returns:
            astf_pred: (B, output_length)
        """
        x = torch.stack([target_waveform, egf], dim=1)  # (B, 2, L)
        x = self.feature_extractor(x)                   # (B, 256, 16)
        x = self.global_pool(x)                         # (B, 256, 1)
        x = self.classifier(x)                          # (B, output_length)
        return x


class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()
        
        # Down: 编码器
        self.down1 = self.conv_block(2, 32)     # (B, 32, 256)
        self.pool1 = nn.MaxPool1d(2)            # → (B, 32, 128)

        self.down2 = self.conv_block(32, 64)    # (B, 64, 128)
        self.pool2 = nn.MaxPool1d(2)            # → (B, 64, 64)

        self.down3 = self.conv_block(64, 128)   # (B, 128, 64)
        self.pool3 = nn.MaxPool1d(2)            # → (B, 128, 32)

        # Bottom
        self.bottom = self.conv_block(128, 256)  # (B, 256, 32)

        # Up: 解码器
        self.up3 = self.up_block(256, 128)       # upsample to (B, 128, 64)
        self.dec3 = self.conv_block(256, 128)    # concat with down3 → (B, 128, 64)

        self.up2 = self.up_block(128, 64)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = self.up_block(64, 32)
        self.dec1 = self.conv_block(64, 32)

        # Final output layer
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)  # (B, 2, 256)

        # Down path
        d1 = self.down1(x)         # (B, 32, 256)
        p1 = self.pool1(d1)        # (B, 32, 128)

        d2 = self.down2(p1)        # (B, 64, 128)
        p2 = self.pool2(d2)        # (B, 64, 64)

        d3 = self.down3(p2)        # (B, 128, 64)
        p3 = self.pool3(d3)        # (B, 128, 32)

        # Bottom
        bn = self.bottom(p3)       # (B, 256, 32)

        # Up path
        up3 = self.up3(bn)         # (B, 128, 64)
        up3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(up3)      # (B, 128, 64)

        up2 = self.up2(dec3)       # (B, 64, 128)
        up2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(up2)      # (B, 64, 128)

        up1 = self.up1(dec2)       # (B, 32, 256)
        up1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(up1)      # (B, 32, 256)

        out = self.final_conv(dec1)  # (B, 1, 256)
        return out.squeeze(1)        # (B, 256)

####################################Loss Functions####################################
class AmplitudeWeightedMSELoss(nn.Module):
    def __init__(self, epsilon, a):
        super(AmplitudeWeightedMSELoss, self).__init__()
        self.epsilon = epsilon
        self.a = a

    def forward(self, y_pred, y_true):
        # 计算每组数据的振幅范围
        amplitude = torch.max(torch.abs(y_true), dim=1, keepdim=True)[0] + self.epsilon
        
        # 使用振幅范围归一化
        normalized_error = (y_pred - y_true) / amplitude**self.a
        # 计算加权 MSE 损失
        return torch.mean(normalized_error ** 2)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):  # 避免除以0
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

class EffectiveMSELoss(nn.Module):
    def __init__(self, threshold=1e-3):
        """
        去掉振幅加权的有效段 MSE 损失函数。
        只在预测或标签中非零的主信号段（[start:end]）计算 MSE。

        参数:
        - threshold: 相对最大值的非零判断阈值（建议 1e-3）
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        B, L = y_pred.shape

        # 最大幅值（防止除以0）
        max_p = torch.amax(torch.abs(y_pred), dim=1, keepdim=True) + 1e-6
        max_t = torch.amax(torch.abs(y_true), dim=1, keepdim=True) + 1e-6

        # 非零掩码（只要预测或标签大于阈值）
        mask_p = torch.abs(y_pred) > (self.threshold * max_p)
        mask_t = torch.abs(y_true) > (self.threshold * max_t)
        mask = mask_p | mask_t  # 合并：预测或标签有信号都算有效

        # 找有效区间 [start, end]
        idx_start = mask.float().argmax(dim=1)  # 每行第一个True
        idx_end = L - mask.flip(dims=[1]).float().argmax(dim=1) - 1  # 最后一个True

        # 构造掩码 [start, end]
        range_idx = torch.arange(L, device=y_pred.device)[None, :]  # (1, L)
        mask_eff = (range_idx >= idx_start[:, None]) & (range_idx <= idx_end[:, None])  # (B, L)

        # 误差平方
        diff_sq = (y_pred - y_true) ** 2  # (B, L)
        masked_error = diff_sq * mask_eff  # 只保留有效段误差

        # 平均每个样本有效段的 MSE
        valid_counts = mask_eff.sum(dim=1).clamp(min=1)  # (B,)
        loss_per_sample = masked_error.sum(dim=1) / valid_counts

        return loss_per_sample.mean()

class EffectiveAmplitudeWeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6, a=0.8, threshold=1e-3):
        """
        使用 mask = mask_true | mask_pred 来构造有效区间掩码。
        仅在预测值或真实值为非零的位置之间计算振幅加权 MSE。
        
        参数：
        - epsilon: 防止除零
        - a: 振幅归一幂次
        - threshold: 相对最大值阈值，控制非零判定
        """
        super().__init__()
        self.epsilon = epsilon
        self.a = a
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        B, L = y_pred.shape

        # 振幅最大值（预测 & 标签）
        max_p = torch.amax(torch.abs(y_pred), dim=1, keepdim=True) + self.epsilon
        max_t = torch.amax(torch.abs(y_true), dim=1, keepdim=True) + self.epsilon

        # 相对阈值掩码（判断非零区）
        mask_p = torch.abs(y_pred) > (self.threshold * max_p)
        mask_t = torch.abs(y_true) > (self.threshold * max_t)

        # 合并掩码（预测或标签中有信号都视为有效）
        mask = mask_p | mask_t  # shape: (B, L)

        # 找起止点（非零段的起点和终点）
        idx_start = mask.float().argmax(dim=1)  # 每行第一个 True
        idx_end = L - mask.flip(dims=[1]).float().argmax(dim=1) - 1  # 每行最后一个 True

        # 构造掩码 [start, end] 之间为 True
        range_idx = torch.arange(L, device=y_pred.device)[None, :]  # shape: (1, L)
        mask_eff = (range_idx >= idx_start[:, None]) & (range_idx <= idx_end[:, None])  # (B, L)

        # 使用标签的最大幅度进行振幅加权归一
        amp = torch.amax(torch.abs(y_true), dim=1, keepdim=True) + self.epsilon
        weight = amp ** self.a  # shape: (B, 1)

        # 归一化误差平方
        norm_diff_sq = ((y_pred - y_true) / weight) ** 2  # (B, L)

        # 掩码过滤
        masked_error = norm_diff_sq * mask_eff  # (B, L)

        # 平均每个样本的 loss（除以有效长度）
        valid_counts = mask_eff.sum(dim=1).clamp(min=1)
        loss_per_sample = masked_error.sum(dim=1) / valid_counts

        return loss_per_sample.mean()

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device, loss_file, pth_file, scheduler=None, patience=20, device_ids=[1, 2]):
    # print(f"Training on device: {device}")
    # model = nn.DataParallel(model, device_ids=[1])  # 使用第二个GPU
    # model.to(device)
    print(f"🟢 Training on device(s): {device_ids}")
    
    # 如果尚未封装，则封装 DataParallel
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model, device_ids=device_ids)

    model.to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            # 从 batch 中取出输入数据
            target_waveform = batch['target'].to(device)
            egf = batch['egf'].to(device)
            astf = batch['astf'].to(device)

            optimizer.zero_grad()
            predicted_astf = model(target_waveform, egf)
            loss = criterion(predicted_astf, astf)
            # loss = criterion(predicted_astf, astf.unsqueeze(1))  # [B, 501] -> [B, 1, 501]
            train_loss += loss.item()
            loss.backward()

            if epoch == 0:
                print("\n梯度检查:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name:30} grad mean: {param.grad.mean().item():.4f} (shape: {tuple(param.grad.shape)})")
                    else:
                        print(f"{name:30} has no gradient")

            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                target_waveform = batch['target'].to(device)
                egf = batch['egf'].to(device)
                astf = batch['astf'].to(device)

                predicted_astf = model(target_waveform, egf)
                loss = criterion(predicted_astf, astf)
                # loss = criterion(predicted_astf, astf.unsqueeze(1))  # [B, 501] -> [B, 1, 501]
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

        if scheduler:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} - Current Learning Rate: {current_lr:.6f}")

        save_losses(train_losses, val_losses, loss_file)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(model.state_dict(), pth_file)
            print(f"Model saved with Validation Loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"EarlyStopping counter: {early_stop_counter} out of {patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    print(f"Training completed. Best model achieved at epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}")
    return train_losses, val_losses


def save_losses(train_losses, val_losses, filename):
    with open(filename, 'w') as f:
        f.write("Epoch, Train Loss, Validation Loss\n")  # 写入表头
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{epoch+1}, {train_loss:.4f}, {val_loss:.4f}\n")  # 格式化并写入每一行


################加载数据，定义超参数#################
# train_h5 = "/group/data/dataset_ASTF-net/Target_waveforms/Train_pairs.h5"
# valid_h5 = "/group/data/dataset_ASTF-net/Target_waveforms/Validation_pairs.h5"
train_h5 = "/group/data/dataset_ASTF-net/Target_waveforms/Train_pairs_3_Samezero_501_new_3_1.h5"
valid_h5 = "/group/data/dataset_ASTF-net/Target_waveforms/Validation_pairs_3_Samezero_501_new_3_1.h5"

loss_file = "/group/prc/data/dataset_ASTF-net/loss_file/best_seismic_CNN5_3_Samezero_501_new_3_cuda1_noise_dropout_EffectiveAmpWeiMSE_1_losses.txt"
pth_file = "/group/prc/data/dataset_ASTF-net/pth_file/best_seismic_CNN5_3_Samezero_501_new_3_cuda1_noise_dropout_EffectiveAmpWeiMSE_1.pth"

batch_size = 1024 * 4

# 加载HDF5数据并创建DataLoader
train_loader = get_dataloader_from_hdf5(train_h5, batch_size, enable_noise=True, enable_dropout=True, enable_trend=False, max_noise_level=0.1)
valid_loader = get_dataloader_from_hdf5(valid_h5, batch_size, enable_noise=False, enable_dropout=False, enable_trend=False, max_noise_level=0.1)

# 参数和超参数设置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1,2")  # 选择空闲的 GPU
device = torch.device("cuda:1")  # device_ids[0]
device_ids = [1]
learning_rate = 1e-3
num_epochs = 700
# weight_decay = 1e-5

# 使用之前完成的数据加载

# 初始化模型
# model = SimpleCNN(in_channels=2, output_length=501)# 初始化模型
# model = UNet1D()

model = EnhancedCNN(in_channels=2, output_length=501)# 初始化模型
# model = AlexNet1D(in_channels=2, output_length=301)# 初始化模型

# criterion = AmplitudeWeightedMSELoss(epsilon=1e-6, a=0.8)  # 使用自定义的相对均方误差损失
criterion = EffectiveAmplitudeWeightedMSELoss(a=0.8, threshold=1e-3)
# criterion = EffectiveMSELoss(threshold=1e-3)
# criterion = RMSELoss()  # 使用自定义的均方根误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# 训练模型
train_model(model, train_loader, valid_loader, num_epochs, criterion, optimizer, device, loss_file, pth_file, scheduler, patience=50, device_ids=device_ids)
                                  
