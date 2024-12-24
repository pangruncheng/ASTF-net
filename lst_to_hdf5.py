import torch
import h5py
import numpy as np
from obspy import read
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 获取时间窗（P波和S波到达时间）
def get_window_times(Z_hdr):
    before_sec = 0.25
    after_sec = 1.85
    a = Z_hdr.get('a', None)
    t0 = Z_hdr.get('t0', None)
    
    if a is not None and t0 is not None:
        t_p = a  # P波到达时间
        t_s = t0  # S波到达时间
    elif Z_hdr.t3 > 0 and Z_hdr.t4 > 0:
        t_p = Z_hdr.t3  # P波到达时间
        t_s = Z_hdr.t4  # S波到达时间
    elif a is not None and Z_hdr.t4 > 0:
        t_p = a  # P波到达时间
        t_s = Z_hdr.t4  # S波到达时间
    elif Z_hdr.t3 > 0 and t0 is not None:
        t_p = Z_hdr.t3  # P波到达时间
        t_s = t0  # S波到达时间
    else:
        raise ValueError("没有找到有效的P波或S波到达时间。")
    
    # 根据P波或S波确定时间窗
    if Z_hdr.kcmpnm == 'HHZ':  # Z分量
        start_time = int((t_p - before_sec) / Z_hdr.delta)
        end_time = int((t_p + after_sec) / Z_hdr.delta)
    else:  # T分量
        start_time = int((t_s - before_sec) / Z_hdr.delta)
        end_time = int((t_s + after_sec) / Z_hdr.delta)

    return start_time, end_time

# 震级转换为M0
def compute_M0(magnitude):
    Mw = 0.754 * magnitude + 0.88  # Mw ML转换
    return 10 ** (1.5 * Mw + 9.105)

# 函数：读取.lst文件中的每一行，返回对应的文件路径
def read_lst_file(lst_filename):
    """读取.lst文件中的每行路径，返回每行的文件路径（Target waveform, EGF, ASTF）"""
    with open(lst_filename, 'r') as f:
        # 每一行是 "Target_waveform_path EGF_path ASTF_path"
        return [line.strip().split() for line in f.readlines()]

# 函数：加载SAC文件并返回波形数据
def load_sac_file(sac_file):
    """读取SAC文件，返回波形数据"""
    waveform = read(sac_file)[0]
    return waveform

# 函数：补零到最大长度
def pad_to_max_length(waveform_data, max_length):
    """将波形数据补零到最大长度"""
    if len(waveform_data) < max_length:
        padded_data = np.pad(waveform_data, (0, max_length - len(waveform_data)), mode='constant')
    else:
        padded_data = waveform_data
    return padded_data

def normalize_waveform(waveform):
    """对波形进行归一化"""
    # 确保waveform是numpy.ndarray类型
    if isinstance(waveform, np.ndarray):
        # 计算波形的最大值和最小值
        max_value = np.max(waveform)
        min_value = np.min(waveform)
        
        # 计算最大幅度，取最大值和最小值的绝对值
        normalization_coefficient = max(abs(max_value), abs(min_value))
        
        # 归一化
        norm_waveform = waveform / normalization_coefficient
        return normalization_coefficient, norm_waveform
    else:
        raise TypeError("Input waveform should be a numpy ndarray")

# 数据加载函数：返回Target Waveform、EGF和ASTF的张量对
def load_data_pair(target_waveform_path, egf_path, astf_path):
    """加载一对数据，返回Target、EGF、ASTF张量"""
    
    # 加载EGF和Target Waveform
    target_waveform = load_sac_file(target_waveform_path)
    egf = load_sac_file(egf_path)
    M0 = compute_M0(egf.stats.sac.mag)
    
    # 获取时间窗（P波和S波到达时间）
    start_time, end_time = get_window_times(egf.stats.sac)
    
    # 截取数据
    target_waveform_data = target_waveform.data
    egf_data = egf.data[start_time:end_time]
    
    # 获取较长的长度
    max_length = max(len(target_waveform_data), len(egf_data))
    
    # 补零
    # target_waveform_data = pad_to_max_length(target_waveform_data, max_length)
    egf_data = pad_to_max_length(egf_data, max_length)
    
    # 加载ASTF
    astf_data = load_sac_file(astf_path).data  # 直接加载ASTF数据
    astf_data /= M0
    
    # 转换为PyTorch张量
    target_waveform_tensor = torch.tensor(target_waveform_data, dtype=torch.float32)
    egf_tensor = torch.tensor(egf_data, dtype=torch.float32)
    astf_tensor = torch.tensor(astf_data, dtype=torch.float32)
    
    return target_waveform_tensor, egf_tensor, astf_tensor

def save_data_to_hdf5(lst_file, hdf5_filename, batch_size=10000, compress=True):
    """从.lst文件中读取数据路径，加载并保存数据为HDF5格式，优化为批量写入"""

    # 打开 HDF5 文件，并创建数据集
    with h5py.File(hdf5_filename, 'w') as hf:
        # 设置压缩选项
        compression_opts = 'gzip' if compress else None
        
        # 创建数据集的结构
        target_waveforms_ds = hf.create_dataset(
            'target_waveforms', 
            shape=(0, 510), 
            maxshape=(None, 510), 
            dtype='float32', 
            chunks=(batch_size, 510), 
            compression=compression_opts
        )
        egfs_ds = hf.create_dataset(
            'egfs', 
            shape=(0, 510), 
            maxshape=(None, 510), 
            dtype='float32', 
            chunks=(batch_size, 510), 
            compression=compression_opts
        )
        astfs_ds = hf.create_dataset(
            'astfs', 
            shape=(0, 301), 
            maxshape=(None, 301), 
            dtype='float32', 
            chunks=(batch_size, 301), 
            compression=compression_opts
        )

        # 读取.lst文件中的路径
        data_pairs = read_lst_file(lst_file)

        # 缓存数据
        target_waveforms_batch = []
        egfs_batch = []
        astfs_batch = []

        for idx, (target_waveform_path, egf_path, astf_path) in enumerate(data_pairs):
            # 加载数据对
            target_waveform_tensor, egf_tensor, astf_tensor = load_data_pair(target_waveform_path, egf_path, astf_path)
            
            # 将数据转换为NumPy数组并加入批次
            target_waveforms_batch.append(target_waveform_tensor.numpy())
            egfs_batch.append(egf_tensor.numpy())
            astfs_batch.append(astf_tensor.numpy())
            
            # 如果当前批次达到指定的大小，就写入 HDF5 文件
            if len(target_waveforms_batch) >= batch_size:
                target_waveforms_ds.resize(target_waveforms_ds.shape[0] + len(target_waveforms_batch), axis=0)
                egfs_ds.resize(egfs_ds.shape[0] + len(egfs_batch), axis=0)
                astfs_ds.resize(astfs_ds.shape[0] + len(astfs_batch), axis=0)

                target_waveforms_ds[-len(target_waveforms_batch):] = np.array(target_waveforms_batch)
                egfs_ds[-len(egfs_batch):] = np.array(egfs_batch)
                astfs_ds[-len(astfs_batch):] = np.array(astfs_batch)

                # 清空批次缓存
                target_waveforms_batch = []
                egfs_batch = []
                astfs_batch = []

            # 每5000个样本输出一次进度
            if (idx + 1) % 50000 == 0:
                print(f"已处理 {idx + 1}/{len(data_pairs)} 数据对")

        # 如果还有剩余数据（小于一个批次），写入到 HDF5 文件
        if target_waveforms_batch:
            target_waveforms_ds.resize(target_waveforms_ds.shape[0] + len(target_waveforms_batch), axis=0)
            egfs_ds.resize(egfs_ds.shape[0] + len(egfs_batch), axis=0)
            astfs_ds.resize(astfs_ds.shape[0] + len(astfs_batch), axis=0)

            target_waveforms_ds[-len(target_waveforms_batch):] = np.array(target_waveforms_batch)
            egfs_ds[-len(egfs_batch):] = np.array(egfs_batch)
            astfs_ds[-len(astfs_batch):] = np.array(astfs_batch)

        print(f"数据已保存到 HDF5 文件：{hdf5_filename}")


# 路径
test_level2_lst = "/group/data/dataset_ASTF-net/Target_waveforms/EGFs_test_level2_pairs.lst"
test_level3_lst = "/group/data/dataset_ASTF-net/Target_waveforms/EGFs_test_level3_pairs.lst"
# test_level1_lst = "/group/data/dataset_ASTF-net/Target_waveforms/EGFs_test_level1_pairs.lst"

# 示例：将数据保存为 HDF5 文件
save_data_to_hdf5(test_level2_lst, '/group/data/dataset_ASTF-net/Target_waveforms/Test_level2_pairs.h5')
save_data_to_hdf5(test_level3_lst, '/group/data/dataset_ASTF-net/Target_waveforms/Test_level3_pairs.h5')
# save_data_to_hdf5(test_level1_lst, '/group/data/dataset_ASTF-net/Target_waveforms/Test_level1_pairs.h5')