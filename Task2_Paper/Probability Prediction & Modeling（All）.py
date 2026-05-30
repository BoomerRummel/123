import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. 统一数据准备
df = pd.read_csv("cleaned_heart_failure_clinical_records.csv")
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. MLP 数据标准化 (仅针对 MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 初始化模型字典
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric='logloss'),
    "MLP": Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}
models["MLP"].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 训练与评估
results = []
plt.figure(figsize=(10, 8))

for name, model in models.items():
    if name == "MLP":
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)
        y_prob = model.predict(X_test_scaled).ravel()
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_prob > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": auc_val
    })
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.2f})')

# 5. 可视化与输出
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves')
plt.legend()
plt.savefig("model_comparison_roc.png")
plt.show()

# 结果对比表
results_df = pd.DataFrame(results)
print("\n--- 模型性能对比表 (用于论文) ---")
print(results_df.to_string(index=False))
print(results_df.to_latex(index=False))
