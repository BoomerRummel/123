from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 使用 pathlib 定义路径
data_path = Path("heart_failure_clinical_records_dataset.csv")
output_path = Path("cleaned_heart_failure_clinical_records.csv")

# 检查文件是否存在
if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"Successfully loaded dataset from: {data_path.resolve()}")
else:
    raise FileNotFoundError(f"未找到文件，请检查路径: {data_path.absolute()}")

# 2. 描述性统计分析
print("--- 数据结构概览 ---")
print(df.info())
print("\n--- 描述性统计 ---")
print(df.describe())

# 3. 检查缺失值
print("\n--- 缺失值统计 ---")
print(df.isnull().sum())

# 4. 异常值分析与合理处理 (IQR法)
numerical_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                  'platelets', 'serum_creatinine', 'serum_sodium', 'time']

def handle_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 截断处理，而非删除
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    return data

# 对数值列应用处理
for col in numerical_cols:
    df = handle_outliers_iqr(df, col)

print("\n--- 异常值处理完成 ---")

# 5. 可视化检查分布 (可选)
plt.figure(figsize=(12, 6))
df.boxplot(column=numerical_cols, rot=45)
plt.title("Boxplot of Numerical Features after Outlier Capping")
plt.show()

# 6. 保存清洗后的数据，供后续步骤读取
df.to_csv(output_path, index=False)
print(f"\n--- 数据清洗完成，已保存至: {output_path.resolve()} ---")