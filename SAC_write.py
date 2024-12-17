import os
import numpy as np
import pandas as pd
from obspy import Trace, Stream
from obspy.core import UTCDateTime
from pathlib import Path

# 输入文件夹路径
input_folder = "/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_S_data/ASTF_S_data_Even_mag3to4.5_M0_randperm/"
output_folder = "/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_S_data/ASTF_S_data_Even_mag3to4.5_M0_randperm_SAC/"

# 获取文件夹下所有CSV文件
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# 遍历所有CSV文件
for csv_file in csv_files:
    # 构建CSV文件的完整路径
    csv_file_path = os.path.join(input_folder, csv_file)

    # 读取CSV文件
    df_1 = pd.read_csv(csv_file_path, sep=',', quotechar='"', header=0, on_bad_lines='skip')
    df_2 = pd.read_csv(csv_file_path, sep=',', quotechar='"', skiprows=2, header=None, on_bad_lines='skip')

    # 第一行（azimuths）提取az列
    azimuths = df_1.iloc[0, 7].split(",")  # 假设az是以逗号分隔的字符串
    azimuths = [float(az) for az in azimuths]  # 转换为浮动类型的方位列表

    # 从第三行开始读取波形数据
    waveforms = df_2.iloc[0:, 0:].values  # 假设从第0列开始是波形数据

    # 创建以CSV文件名命名的文件夹
    csv_folder_name = Path(csv_file).stem
    folder_to_save = os.path.join(output_folder, csv_folder_name)
    
    # 如果文件夹不存在，则创建它
    os.makedirs(folder_to_save, exist_ok=True)

    for i in range(waveforms.shape[0]):
        row = df_1
        
        # 创建一个Trace对象
        trace = Trace()

        # 设置SAC头部字段
        trace.stats.network = "ASTF_S"  # 假设网络名称为KISK
        trace.stats.station = f"AZ{int(azimuths[i])}"  # 使用azimuth作为台站名称的一部分
        trace.stats.sac = {}

        # 设置自定义的SAC字段
        trace.stats.sac.user8 = azimuths[i]  # 方位
        trace.stats.sac.mag = row['mw'].iloc[0]  # 震级
        trace.stats.sac.user0 = row['m0'].iloc[0]  # 地震矩
        trace.stats.sac.user1 = row['stress_drop'].iloc[0]  # 应力降
        trace.stats.sac.user2 = row['Lc'].iloc[0]  # 长度
        trace.stats.sac.user3 = row['Wc'].iloc[0]  # 宽度
        trace.stats.sac.user4 = row['hyp'].iloc[0]  # 破裂起始点的位置沿着长轴中心的比例(-1,1)
        trace.stats.sac.user5 = row['drp'].iloc[0]  # 椭圆长轴与strike的夹角(-180,180)
        # trace.stats.sac.user6 = "delta(t)" # rise函数
        # trace.stats.sac.user7 = "sqrt(1-r^2)" # slip分布

        # 设置采样频率和起始时间
        trace.stats.sampling_rate = row['Fs'].iloc[0]  # 采样频率
        trace.stats.starttime = UTCDateTime(0)  # 假设波形从时间0开始

        # 设置波形数据
        trace.data = np.array(waveforms[i, :])  # 提取每一行波形数据

        # 创建一个Stream对象
        stream = Stream(traces=[trace])

        # 保存SAC文件
        sac_file = os.path.join(folder_to_save, f"{Path(csv_file).stem}_az_{int(azimuths[i])}.SAC")
        stream.write(sac_file, format="SAC")
        print(f"Saved SAC file: {sac_file}")

