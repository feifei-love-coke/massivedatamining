import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import time
import psutil
import os

def get_memory_usage():
    """返回当前进程的内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # 转换为MB

# 记录初始时间和内存
start_time = time.time()
initial_memory = get_memory_usage()

# 1. 数据加载
print("\n=== 数据加载阶段 ===")
df = pd.read_csv('emails.csv')
df.columns = ['text', 'label']
print(f"数据加载完成，内存占用: {get_memory_usage() - initial_memory:.2f} MB")

# 2. 特征提取
print("\n=== 特征提取阶段 ===")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label'].values
print(f"特征提取完成，内存占用: {get_memory_usage() - initial_memory:.2f} MB")

# 3. 数据拆分
print("\n=== 数据拆分阶段 ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"数据拆分完成，内存占用: {get_memory_usage() - initial_memory:.2f} MB")

# 4. 训练和预测（Naive Bayes）
print("\n=== Naive Bayes 训练和预测 ===")
nb_start_time = time.time()
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_pred_proba = nb.predict_proba(X_test)[:, 1]
nb_time = time.time() - nb_start_time
print(f"Naive Bayes 训练和预测时间: {nb_time:.4f} 秒")
print(f"当前内存占用: {get_memory_usage() - initial_memory:.2f} MB")

# 5. 训练和预测（Logistic Regression）
print("\n=== Logistic Regression 训练和预测 ===")
lr_start_time = time.time()
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_pred_proba = lr.predict_proba(X_test)[:, 1]
lr_time = time.time() - lr_start_time
print(f"Logistic Regression 训练和预测时间: {lr_time:.4f} 秒")
print(f"当前内存占用: {get_memory_usage() - initial_memory:.2f} MB")

# 6. 评估模型
print("\n=== 模型评估 ===")
print("Naive Bayes Metrics:")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall:", recall_score(y_test, nb_pred))
print("F1 Score:", f1_score(y_test, nb_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, nb_pred_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, nb_pred))

print("\nLogistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, lr_pred_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# 7. 绘制 ROC 曲线
print("\n=== 绘制 ROC 曲线 ===")
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='blue', label=f'Naive Bayes (AUC = {roc_auc_score(y_test, nb_pred_proba):.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve')
plt.legend()
plt.show()

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='red', label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_pred_proba):.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend()
plt.show()

# 8. 计算总时间和内存
total_time = time.time() - start_time
final_memory = get_memory_usage()
print("\n=== 最终统计 ===")
print(f"总运行时间: {total_time:.4f} 秒")
print(f"Naive Bayes 训练和预测时间: {nb_time:.4f} 秒")
print(f"Logistic Regression 训练和预测时间: {lr_time:.4f} 秒")
print(f"峰值内存占用: {final_memory - initial_memory:.2f} MB")