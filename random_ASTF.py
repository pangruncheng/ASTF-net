import os
import shutil
import random

def extract_and_copy_files(z_folder, t_folder, lst_file_paths, output_folders):
    """
    从给定的Z和T文件夹中，按照2:1比例抽取文件，确保不重复并复制到目标文件夹。
    """
    # 创建目标文件夹
    for output_folder in output_folders:
        os.makedirs(output_folder, exist_ok=True)

    # 用来存储所有已抽取的文件路径，确保文件不重复
    all_selected_files = set()

    # 遍历5个LST文件
    for lst_file_path, output_folder in zip(lst_file_paths, output_folders):
        print(f"Processing LST file: {lst_file_path}")

        with open(lst_file_path, 'r') as f:
            lines = f.readlines()
        
        # 提取Z和T分量的路径
        z_files = [line.strip() for line in lines if 'Z' in line]
        t_files = [line.strip() for line in lines if 'T' in line]

        # 计算所需的P波和S波文件数量
        num_z = len(z_files)
        num_t = len(t_files)

        # 抽取的P波和S波文件的数量（2倍）
        num_p = 2 * num_z
        num_s = 2 * num_t

        # 从z_files和t_files中选取不重复的文件
        p_files = random.sample(z_files, num_p)
        s_files = random.sample(t_files, num_s)

        # 检查文件是否重复
        while not len(p_files) == len(set(p_files)) or not len(s_files) == len(set(s_files)):
            print(f"Found duplicates, re-selecting files for {lst_file_path}")
            p_files = random.sample(z_files, num_p)
            s_files = random.sample(t_files, num_s)

        # 复制文件到目标文件夹
        selected_files = []
        for p_file in p_files + s_files:
            file_name = os.path.basename(p_file)
            destination_path = os.path.join(output_folder, file_name)

            # 确保文件不重复
            if file_name not in all_selected_files:
                shutil.copy(p_file, destination_path)
                selected_files.append(destination_path)
                all_selected_files.add(file_name)
            else:
                print(f"File {file_name} already copied, skipping.")

        # 生成新的LST文件（将复制的文件路径写入LST）
        new_lst_file = os.path.join(output_folder, os.path.basename(lst_file_path))
        with open(new_lst_file, 'w') as f_out:
            for file_path in selected_files:
                f_out.write(file_path + '\n')

        print(f"Generated new LST file: {new_lst_file}")


# 设置输入输出路径
z_folder = "/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_P_data/ASTF_P_data_Even_mag3to4.5_M0_randperm_SAC"
t_folder = "/media/admin123/Datastorage/data_files/data/data_files/Kiskatinaw_Events_local/ASTF_S_data/ASTF_S_data_Even_mag3to4.5_M0_randperm_SAC"

lst_file_paths = [
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_test_level1.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_test_level2.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_test_level3.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_train.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/EGFs/EGFs_validation.lst"
]

output_folders = [
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_test_level1.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_test_level2.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_test_level3.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_train.lst",
    "/media/admin123/Datastorage/data_files/dataset_ASTF-net/ASTFs/ASTFs_validation.lst"
]

# 执行函数
extract_and_copy_files(z_folder, t_folder, lst_file_paths, output_folders)

