# Medical AI Research & Clinical Data Analysis

## 项目概述

本项目是一个医学研究系列任务，涵盖了从医学文献数据提取、机器学习预测模型构建到学术论文撰写的完整研究流程，同时包含AI辅助医学研究的方法论总结。

---

## 任务概览

### Task 1: 医学文献信息提取 API

**目标**: 利用大语言模型从非结构化医学PDF中提取结构化临床信息。

**技术栈**:
- 语言: Python
- PDF处理: PyMuPDF (fitz)
- LLM API: 阿里云千问 Max
- 数据格式: JSON

**主要功能**:
- 自动化提取病例报告中的关键医学实体
- 提取内容包括：患者基本信息、主要症状、既往史、诊断结果、治疗方案
- 采用Few-Shot Learning增强提取准确性
- 支持对提取结果进行JSON格式化存储

**文件**:
- `API.py`: 主要提取逻辑
- `extracted_medical_info.json`: 提取结果示例

---

### Task 2: 心力衰竭死亡率预测与学术论文

**目标**: 基于心力衰竭临床记录数据集，构建机器学习预测模型并撰写学术论文。

**数据集**: Heart Failure Clinical Records Dataset (~300 patients)

**技术栈**:
- 机器学习: scikit-learn, XGBoost
- 深度学习: TensorFlow/Keras
- 数据分析: pandas, numpy, seaborn, matplotlib
- 论文撰写: LaTeX

**研究流程**:
1. **数据探索与预处理**
   - 描述性统计分析
   - 缺失值检测
   - IQR法异常值处理（保留病理意义）
   - Z-score标准化

2. **特征工程与风险因子检测**
   - Spearman相关性分析
   - 随机森林特征重要性评估
   - 识别Top 3关键风险因子: 随访时间、血清肌酐、射血分数

3. **概率预测与建模**
   - **随机森林 (Random Forest)**: AUC = 0.89
   - **XGBoost**: AUC = 0.83
   - **多层感知机 (MLP)**: AUC = 0.85
   - 多模型性能对比与ROC曲线可视化

4. **学术论文撰写**
   - 涵盖引言、方法、结果、讨论完整结构
   - 包含数学公式、图表、参考文献
   - Cambridge University Press格式

**文件**:
- `Data Exploratory & Preprocessing.py`: 数据清洗
- `Risk Factor Detection&Feature Engineering.py`: 特征选择
- `Probability Prediction & Modeling（RF/XGBoost/MLP）.py`: 单模型实现
- `Probability Prediction & Modeling（All）.py`: 综合对比
- `Task2_Paper.tex`: LaTeX论文源码
- `Task2_Paper.pdf`: 论文PDF
- 图像文件: correlation_heatmap.png, model_comparison_roc.png, roc_curve_*.png, top_3_risk_factors (Tree).png, work flow.png

---

### Task 3: AI辅助医学研究指南

**目标**: 为医学专业研究人员提供AI辅助编程和科研的方法论总结。

**内容模块**:

1. **工具篇**
   - 主流AI模型对比 (ChatGPT, Gemini, Claude)
   - 各模型的优势与适用场景
   - 提示词设计技巧

2. **实战篇**
   - Debug策略
   - 任务拆分方法
   - 多线程工作流
   - 上下文回溯机制

3. **进阶篇**
   - AI图像生成
   - 幻觉问题解决
   - 科研流程图制作

**目标受众**: 非计算机背景的医学研究人员

**文件**:
- `Task3_Guide/Guide.md`: 完整指南文档

---

## 环境要求

```
Python >= 3.8
```

**主要依赖包**:
```
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost, tensorflow
pymupdf, openai, python-dotenv
```

## 项目结构

```
.
├── README.md                              # 项目说明
├── AI_Chat_Records.md                     # AI对话记录
├── Task1_API/                             # 医学文献信息提取
│   ├── API.py
│   ├── extracted_medical_info.json
│   └── A case of portal vein recanalization and symptomatic heart failure.pdf
├── Task2_Paper/                           # 心衰预测研究
│   ├── Data Exploratory & Preprocessing.py
│   ├── Risk Factor Detection&Feature Engineering.py
│   ├── Probability Prediction & Modeling（All）.py
│   ├── Task2_Paper.tex
│   └── [可视化图表]
├── Task3_Guide/                           # AI辅助指南
│   └── Guide.md
└── heart_failure_clinical_records_dataset.csv
```

## 作者

W. Rummel - STAT, JBJI

## 许可证

本项目仅供学术研究与学习使用。

---

*本项目展示了AI技术在医学数据分析和研究中的实际应用，包括自动化信息提取、预测模型构建和科研方法论总结。*