import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier

# 1. 读取清洗后的数据
df = pd.read_csv("cleaned_heart_failure_clinical_records.csv")
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 训练 XGBoost 模型
# scale_pos_weight 参数用于处理心衰数据集中可能存在的正负样本不平衡问题
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=4, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 4. 模型评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] # 获取死亡概率

print("--- XGBoost 模型评估指标 ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# 5. 绘制 ROC 曲线与计算 AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost ROC (AUC = {roc_auc:.2f})', color='purple', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Model ROC Curve')
plt.legend(loc='lower right')
plt.savefig("roc_curve (XGBoost).png")
plt.show()