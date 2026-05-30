import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. 加载清洗后的数据
df = pd.read_csv("cleaned_heart_failure_clinical_records.csv")
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 2. 必须进行标准化 (MLP 对特征尺度敏感)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 3. 构建轻量化 MLP
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # 防止过拟合
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid') # 输出概率
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)

# 5. 模型评估
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("--- MLP 模型评估指标 ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# 6. ROC 曲线与 AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'MLP ROC (AUC = {roc_auc:.2f})', color='green', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP Model ROC Curve')
plt.legend(loc='lower right')
plt.savefig("roc_curve (MLP).png")
plt.show()