import pandas as pd
from obspy import read
from scipy.signal import convolve
from obspy.core import UTCDateTime
import os

# 读取lst文件，获取SAC文件路径
def read_lst_file(lst_file):
    with open(lst_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# 震级转换为M0
def compute_M0(magnitude):
    Mw = 0.754 * magnitude + 0.88  # Mw ML转换
    return 10 ** (1.5 * Mw + 9.105)

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

# 对EGF波形进行降采样至100Hz
def resample_to_100hz(EGF_waveform):
    """检查EGF采样率并进行降采样到100Hz"""
    current_sampling_rate = EGF_waveform.stats.sampling_rate
    if current_sampling_rate != 100:
        print(f"采样率 {current_sampling_rate} Hz，正在降采样至100Hz...")
        EGF_waveform = EGF_waveform.resample(100)
        EGF_waveform.stats.sac.delta = 0.01 
    return EGF_waveform

# 卷积操作：ASTF除以M0后再卷积
def convolve_waveforms(EGF_waveform, ASTF_waveform, start_time, end_time):
    # 获取震级并计算M0
    Z_hdr = EGF_waveform.stats.sac  # 获取SAC头信息
    magnitude = Z_hdr.mag
    M0 = compute_M0(magnitude)

    # 截取时间窗内的信号数据
    EGF_segment = EGF_waveform.data[start_time:end_time]
    ASTF_segment = ASTF_waveform.data

    # 将ASTF除以M0
    ASTF_segment /= M0

    # 执行卷积
    target_waveform = convolve(EGF_segment, ASTF_segment, mode='full')
    return target_waveform

# 设置SAC头段
def set_sac_header(trace, ASTF_waveform, EGF_waveform):
    trace.stats.network = EGF_waveform.stats.network  # 使用EGF的网络名称
    trace.stats.station = EGF_waveform.stats.station  # 使用EGF的台站名称
    trace.stats.sac.kcmpnm = EGF_waveform.stats.sac.kcmpnm  # 分量设置为EGF的分量

    # 设置台站和事件经纬度
    trace.stats.sac.evla = EGF_waveform.stats.sac.evla  # 事件纬度
    trace.stats.sac.evlo = EGF_waveform.stats.sac.evlo  # 事件经度
    trace.stats.sac.stla = EGF_waveform.stats.sac.evla  # 台站纬度
    trace.stats.sac.stlo = EGF_waveform.stats.sac.evlo  # 台站经度

    # 自定义SAC字段
    trace.stats.sac.user8 = ASTF_waveform.stats.sac.user8  # 方位
    trace.stats.sac.mag = ASTF_waveform.stats.sac.mag  # 震级
    trace.stats.sac.user0 = ASTF_waveform.stats.sac.user0  # 地震矩
    trace.stats.sac.user1 = ASTF_waveform.stats.sac.user1  # 应力降
    trace.stats.sac.user2 = ASTF_waveform.stats.sac.user2  # 长度
    trace.stats.sac.user3 = ASTF_waveform.stats.sac.user3  # 宽度
    trace.stats.sac.user4 = ASTF_waveform.stats.sac.user4  # 破裂起始点
    trace.stats.sac.user5 = ASTF_waveform.stats.sac.user5  # 椭圆长轴与strike的夹角

    # 采样频率和起始时间
    trace.stats.sampling_rate = ASTF_waveform.stats.sampling_rate  # 采样频率
    trace.stats.starttime = UTCDateTime(0)  # 假设波形从时间0开始
    

# 在主处理函数中调用新的头段设置函数
def process_files(EGF_files, ASTF_files):
    for EGF_file, ASTF_file in zip(EGF_files, ASTF_files):
        target_pairs = []  # 用于记录EGF、ASTF和生成的目标波形路径
        EGF_paths = read_lst_file(EGF_file)  # 读取EGF文件路径
        ASTF_paths = read_lst_file(ASTF_file)  # 读取ASTF文件路径

        # 获取目标保存路径，并生成对应的文件夹
        target_waveform_dir_name = os.path.basename(EGF_file).split('.')[0]
        target_waveform_dir = os.path.join('/group/data/dataset_ASTF-net/Target_waveforms', f"Target_waveforms_{target_waveform_dir_name}")
        os.makedirs(target_waveform_dir, exist_ok=True)

        # 创建新的lst文件记录生成的SAC文件路径
        target_lst_filename = os.path.join(target_waveform_dir, f"Target_waveforms_{target_waveform_dir_name}.lst")

        with open(target_lst_filename, 'w') as lst_file:
            for EGF_path in EGF_paths:
                # 提取文件名
                EGF_filename = os.path.basename(EGF_path)

                # 使用split()方法分割，并取前两个时间戳部分
                event_time = '.'.join(EGF_filename.split('.')[0:2])
                # 读取EGF波形
                EGF_waveform = read(EGF_path)[0]  # 读取EGF文件

                # 进行降采样
                EGF_waveform = resample_to_100hz(EGF_waveform)

                # 获取时间窗
                Z_hdr = EGF_waveform.stats.sac  # 获取SAC头信息
                start_time, end_time = get_window_times(Z_hdr)

                # 生成子文件夹
                subfolder = f"{event_time}.{EGF_waveform.stats.network}.{EGF_waveform.stats.station}"
                subfolder_path = os.path.join(target_waveform_dir, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)

                # 对应Z分量或T分量的卷积操作
                if EGF_waveform.stats.sac.kcmpnm == 'HHZ':  # Z分量与所有ASTF中的P波卷积
                    for ASTF_path in ASTF_paths:
                        ASTF_waveform = read(ASTF_path)[0]  # 读取ASTF波形

                        if ASTF_waveform.stats.network == 'ASTF_P':  # Z分量与所有ASTF中的P波卷积

                            # 卷积操作
                            target_waveform = convolve_waveforms(EGF_waveform, ASTF_waveform, start_time, end_time)

                            # 保存卷积结果到新的SAC文件
                            trace = EGF_waveform.copy()  # 复制原始波形
                            trace.data = target_waveform  # 更新为卷积结果
                            set_sac_header(trace, ASTF_waveform, EGF_waveform)  # 设置SAC头部信息

                            # 生成保存路径
                            output_filename = os.path.join(subfolder_path, f"Targetwaveform.{event_time}.{EGF_waveform.stats.network}.{EGF_waveform.stats.station}_Mw{ASTF_waveform.stats.sac.mag:.2f}_StressDrop{ASTF_waveform.stats.sac.user1:.2f}_Az{ASTF_waveform.stats.sac.user8:.0f}..HHZ.SAC")
                            trace.write(output_filename, format="SAC")
                            lst_file.write(f"{output_filename}\n")  # 写入到lst文件
                            print(f"保存卷积后的SAC文件: {output_filename}")

                            # 记录EGF、ASTF和目标波形路径
                            target_pairs.append((output_filename, EGF_path, ASTF_path))

                else:  # T分量与所有ASTF中的S波卷积
                    for ASTF_path in ASTF_paths:
                        ASTF_waveform = read(ASTF_path)[0]  # 读取ASTF波形

                        if ASTF_waveform.stats.network == 'ASTF_S':  # T分量与所有ASTF中的S波卷积

                            # 卷积操作
                            target_waveform = convolve_waveforms(EGF_waveform, ASTF_waveform, start_time, end_time)

                            # 保存卷积结果到新的SAC文件
                            trace = EGF_waveform.copy()  # 复制原始波形
                            trace.data = target_waveform  # 更新为卷积结果
                            set_sac_header(trace, ASTF_waveform, EGF_waveform)  # 设置SAC头部信息

                            # 生成保存路径
                            output_filename = os.path.join(subfolder_path, f"Targetwaveform.{event_time}.{EGF_waveform.stats.network}.{EGF_waveform.stats.station}_Mw{ASTF_waveform.stats.sac.mag:.2f}_StressDrop{ASTF_waveform.stats.sac.user1:.2f}_Az{ASTF_waveform.stats.sac.user8:.0f}..T.SAC")
                            trace.write(output_filename, format="SAC")
                            lst_file.write(f"{output_filename}\n")  # 写入到lst文件
                            print(f"保存卷积后的SAC文件: {output_filename}")

                            # 记录EGF、ASTF和目标波形路径
                            target_pairs.append((output_filename, EGF_path, ASTF_path))

        # 创建目标波形路径对的lst文件
        target_pairs_lst_filename = os.path.join('/group/data/dataset_ASTF-net/Target_waveforms', f"{target_waveform_dir_name}_pairs.lst")
        with open(target_pairs_lst_filename, 'w') as f:
            for egf_path, astf_path, target_waveform_path in target_pairs:
                f.write(f"{egf_path} {astf_path} {target_waveform_path}\n")

        print(f"目标波形路径对已保存到: {target_pairs_lst_filename}")

# 文件路径
EGF_files = [
    "/group/data/dataset_ASTF-net/EGFs/EGFs_test_level1.lst",
    "/group/data/dataset_ASTF-net/EGFs/EGFs_test_level2.lst",
    "/group/data/dataset_ASTF-net/EGFs/EGFs_test_level3.lst",
    # "/group/data/dataset_ASTF-net/EGFs/EGFs_train.lst",
    "/group/data/dataset_ASTF-net/EGFs/EGFs_validation.lst"
]

ASTF_files = [
    "/group/data/dataset_ASTF-net/ASTFs/ASTFs_test_level1.lst",
    "/group/data/dataset_ASTF-net/ASTFs/ASTFs_test_level2.lst",
    "/group/data/dataset_ASTF-net/ASTFs/ASTFs_test_level3.lst",
    # "/group/data/dataset_ASTF-net/ASTFs/ASTFs_train.lst",
    "/group/data/dataset_ASTF-net/ASTFs/ASTFs_validation.lst"
    ]

# 执行文件处理
process_files(EGF_files, ASTF_files)


