import os
import random
from collections import defaultdict

# 输入文件路径
input_file = '/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_main_dataset.lst'

# 读取文件内容
with open(input_file, 'r') as f:
    lines = f.readlines()

# 创建字典按事件和台站进行分组
event_station_groups = defaultdict(list)

for line in lines:
    line = line.strip()
    if 'HHZ.SAC' in line or 'T.SAC' in line:
        # 提取事件ID和分量标识（事件+台站和台网名）
        parts = line.split('/')
        event_id = '.'.join(parts[9].split('.')[:2])  # 事件ID是路径中的前两个时间戳部分
        component = '.'.join(parts[9].split('.')[2:4])  # 分量标识是事件后面加台站和台网名部分，如XL.MG05
        event_station_groups[(event_id, component)].append(line)

# 创建全局的文件列表，收集所有事件ID下的Z和T分量
all_files = []

# 收集每个事件ID和台站的Z和T文件（确保它们一起抽取）
for event_station, files in event_station_groups.items():
    all_files.append(files)

# 随机抽取并划分数据集
random.shuffle(all_files)

# 计算划分点
total_len = len(all_files)
split_7 = int(0.7 * total_len)
split_9 = int(0.9 * total_len)

# 按照7:2:1划分
train_files = all_files[:split_7]
validation_files = all_files[split_7:split_9]
test_files = all_files[split_9:]

# 输出结果到文件
output_dir = '/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/'
os.makedirs(output_dir, exist_ok=True)

# 保存每个集合到不同的文件
def save_set(file_name, file_list):
    with open(file_name, 'w') as f:
        for file_group in file_list:
            for file in file_group:
                f.write(f"{file}\n")

# 保存文件
save_set(os.path.join(output_dir, 'EGFs_train.lst'), train_files)
save_set(os.path.join(output_dir, 'EGFs_validation.lst'), validation_files)
save_set(os.path.join(output_dir, 'EGFs_test_level1.lst'), test_files)

print("文件已成功划分并保存！")




