import pandas as pd
import glob
import os

# 定义一个键函数，用于获取每个文件F字段的第一个值
def get_f_first_value(file):
    try:
        df = pd.read_csv(file)
        # 返回F字段的第一个值，如果F字段不存在或为空，返回None
        return df['F'].iloc[0] if 'F' in df.columns and len(df['F']) > 0 else None
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return None

def process_csv_files(folder_path, output_file):
    """
    处理融合文件夹内的所有CSV文件，合并、转换并输出结果

    :param folder_path: 包含CSV文件的文件夹路径
    :param output_file: 输出文件路径
    """
    # 读取所有CSV文件
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # 根据F字段的第一个值对文件进行排序
    sorted_files = sorted(all_files, key=lambda x: get_f_first_value(x))

    # 按排序后的文件顺序读取CSV内容到dfs列表
    dfs = []
    for i, file in enumerate(sorted_files, start=1):
        try:
            df = pd.read_csv(file)

            original_value = df['F'].iloc[0] if not df['F'].empty else None
            if original_value is None:
                print(f"Warning: Column 'F' in {file} is empty, skipping mapping.")
            else:
                # 更新映射字典
                mapping_dict[int(original_value)] = i

            df['F'] = i
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # 合并所有数据
    combined_df = pd.concat(dfs, ignore_index=True)

    # 按原始F字段排序
    # sorted_df = combined_df.sort_values('F').reset_index(drop=True)
    sorted_df = combined_df

    # # 删除X2、Y2列
    # sorted_df = sorted_df.drop(['X2', 'Y2'], axis=1)
    #
    # # 列重命名
    # sorted_df = sorted_df.rename(columns={
    #     'F': 'Frequency_ID',
    #     'X1': 'X',
    #     'Y1': 'Y',
    #     'X': 'deltaX',
    #     'Y': 'deltaY'
    # })

    # 删除X、Y列
    sorted_df = sorted_df.drop(['X', 'Y'], axis=1)

    # 列重命名
    sorted_df = sorted_df.rename(columns={
        'F': 'Frequency_ID',
        # 'X1': 'X',
        # 'Y1': 'Y',
        # 'X2': 'deltaX',
        # 'Y2': 'deltaY'
    })


    # 保存结果
    sorted_df.to_csv(output_file, index=False)
    print(f"处理完成！共合并 {len(all_files)} 个文件，生成 {len(sorted_df)} 条记录")

mapping_dict = {}  # 用于存储字段原值和计数器i的映射关系
# 使用示例
process_csv_files(
    folder_path="csv",
    output_file="full_data_1.csv"
)
print(f"Mapping Dictionary: {mapping_dict}")