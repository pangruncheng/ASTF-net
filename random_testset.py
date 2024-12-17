import pandas as pd
import numpy as np

# 输入CSV文件路径
input_file = '/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_P_data/ASTF_P_data_Even_mag3to5_M0_1.csv'

# 输出文件路径（保存每次抽取的结果）
output_file_prefix = '/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_P_data/ASTF_P_data_Even_mag3to5_M0/ASTF_P_dataset_'

# 读取CSV文件
df = pd.read_csv(input_file)

# 确保数据不小于1000行
if len(df) < 1000:
    raise ValueError("数据集的行数小于1000，无法进行随机抽取！")

# 抽取3次不重复的1000行
for i in range(1, 4):
    # 随机抽取1000行
    sampled_df = df.sample(n=1000, random_state=np.random.randint(0, 10000))
    
    # 保存到新的CSV文件中
    output_file = f"{output_file_prefix}{i}.csv"
    sampled_df.to_csv(output_file, index=False)
    
    # 从原数据中删除已抽取的行
    df = df.drop(sampled_df.index)

    print(f"第{i}次抽取的结果已保存到 {output_file}")

print("所有抽取完成！")
