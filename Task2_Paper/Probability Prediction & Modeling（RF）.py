from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, classification_report)

# 1. 加载数据
cleaned_data_path = Path("cleaned_heart_failure_clinical_records.csv")

if not cleaned_data_path.exists():
    raise FileNotFoundError("请先运行数据清洗脚本生成 cleaned_heart_failure.csv")

df = pd.read_csv(cleaned_data_path)

# 2. 划分特征与标签
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 3. 划分训练集与测试集 (80% 训练, 20% 测试)
# 设置 stratify=y 确保训练集和测试集中死亡事件的比例一致，这对临床数据非常关键
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 训练模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. 概率预测与评估
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1] # 获取预测为死亡的概率

# 输出各项指标
print("--- 模型评估指标 ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. 绘制 ROC 曲线并计算 AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) for Heart Failure Mortality')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png")
plt.show()

# 7. 新患者风险预测函数
def predict_new_patient_risk(model, patient_data):
    """
    patient_data: 字典格式，包含所有特征
    """
    input_df = pd.DataFrame([patient_data])
    # 确保列顺序与训练时一致
    input_df = input_df[X_train.columns]
    
    probability = model.predict_proba(input_df)[0][1]
    return probability

# 使用示例 (请替换为实际临床数据)
new_patient = {
    'age': 65.0, 'anaemia': 0, 'creatinine_phosphokinase': 150, 'diabetes': 1,
    'ejection_fraction': 35, 'high_blood_pressure': 0, 'platelets': 260000,
    'serum_creatinine': 1.2, 'serum_sodium': 137, 'sex': 1, 'smoking': 0, 'time': 100
}

risk_score = predict_new_patient_risk(rf_model, new_patient)
print(f"\n--- 新患者预测结果 ---")
print(f"该患者随访期内死亡概率预测为: {risk_score:.2%}")