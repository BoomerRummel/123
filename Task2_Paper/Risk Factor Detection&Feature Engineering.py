"""
医学研究脚本：心衰临床特征重要性评估 (Random Forest)

注：无需进行标准化/归一化处理。
原因：随机森林基于决策树，分裂过程（Split）是基于特征阈值比较，而非特征间的距离度量（如SVM）
或梯度更新（如逻辑回归）。特征的缩放（Scaling）不改变分裂点的位置和模型性能。
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# 1. 加载数据
file_path = Path("cleaned_heart_failure_clinical_records.csv")
if not file_path.exists():
    raise FileNotFoundError("未找到数据集文件，请确保文件名正确且在当前目录下。")

df = pd.read_csv(file_path)

# 2. 相关性热力图 (Correlation Heatmap)
# 作用：量化各临床指标与 DEATH_EVENT 以及各指标间的线性相关性
plt.figure(figsize=(12, 8))
# 使用 Spearman 相关性，因为它对非线性关系和异常值更稳健
correlation_matrix = df.corr(method='spearman') 
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Clinical Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 3. 基于随机森林的危险因子检测 (Top 3)
# 作用：通过树模型捕捉非线性影响，筛选出对预测死亡事件贡献最大的特征
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
rf_model.fit(X, y)

# 提取特征重要性
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_3_factors = importances.sort_values(ascending=False).head(3)

print("--- Top 3 危险因子 ---")
print(top_3_factors)

# 可视化重要性
plt.figure(figsize=(8, 5))
top_3_factors.plot(kind='barh', color='skyblue')
plt.title('Top 3 Risk Factors for Heart Failure Mortality')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis() # 将最重要的因子排在最上方
plt.tight_layout()
plt.savefig("top_3_risk_factors.png")
plt.show()