import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, SimpleRNN, GRU
import time
import psutil
import os

# 记录初始内存
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # 返回MB

initial_memory = get_memory_usage()

# 1. 数据加载和预处理
print("\n=== 数据加载和预处理 ===")
start_time = time.time()

df = pd.read_csv('email.csv')
df.columns = ['label', 'text']
df = df[df['label'].isin(['ham', 'spam'])]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df['text'] = df['text'].apply(lambda x: re.sub(r'\W+', ' ', x.lower()))

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
y = to_categorical(df['label_num'], num_classes=2)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess_time = time.time() - start_time
print(f"预处理完成，耗时: {preprocess_time:.4f} 秒")
print(f"当前内存占用: {get_memory_usage() - initial_memory:.2f} MB")

# 2. 定义评估函数（带时间和内存记录）
def evaluate_model(model, X_test, y_test, model_name):
    start_time = time.time()
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=['ham', 'spam'], digits=4, output_dict=True)
    print(f"\n{model_name} Classification Report:")
    print(f"{'':<10}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    for k, v in report.items():
        if k in ['ham', 'spam']:
            print(f"{k:<10}{v['precision']:>10.4f}{v['recall']:>10.4f}{v['f1-score']:>10.4f}{v['support']:>10.0f}")
    print(f"{'accuracy':<10}{'':>10}{'':>10}{report['accuracy']:>10.4f}{report['macro avg']['support']:>10.0f}")
    
    fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    eval_time = time.time() - start_time
    print(f"评估耗时: {eval_time:.4f} 秒")
    print(f"当前内存占用: {get_memory_usage() - initial_memory:.2f} MB")
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'auc': roc_auc,
        'eval_time': eval_time
    }

results = {}

# 3. 训练和评估 FNN
print("\n=== 训练和评估 FNN ===")
start_time = time.time()

model_fnn = Sequential([
    Embedding(5000, 64, input_length=100),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
model_fnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_fnn.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32, verbose=0)

train_time_fnn = time.time() - start_time
print(f"FNN 训练完成，耗时: {train_time_fnn:.4f} 秒")
print(f"当前内存占用: {get_memory_usage() - initial_memory:.2f} MB")

results['FNN'] = evaluate_model(model_fnn, X_test, y_test, "FNN")
results['FNN']['train_time'] = train_time_fnn

# 4. 训练和评估 RNN
print("\n=== 训练和评估 RNN ===")
start_time = time.time()

model_rnn = Sequential([
    Embedding(5000, 64, input_length=100),
    SimpleRNN(64),
    Dense(2, activation='softmax')
])
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_rnn.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32, verbose=0)

train_time_rnn = time.time() - start_time
print(f"RNN 训练完成，耗时: {train_time_rnn:.4f} 秒")
print(f"当前内存占用: {get_memory_usage() - initial_memory:.2f} MB")

results['RNN'] = evaluate_model(model_rnn, X_test, y_test, "RNN")
results['RNN']['train_time'] = train_time_rnn

# 5. 训练和评估 GRU
print("\n=== 训练和评估 GRU ===")
start_time = time.time()

model_gru = Sequential([
    Embedding(5000, 64, input_length=100),
    GRU(64),
    Dense(2, activation='softmax')
])
model_gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_gru.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32, verbose=0)

train_time_gru = time.time() - start_time
print(f"GRU 训练完成，耗时: {train_time_gru:.4f} 秒")
print(f"当前内存占用: {get_memory_usage() - initial_memory:.2f} MB")

results['GRU'] = evaluate_model(model_gru, X_test, y_test, "GRU")
results['GRU']['train_time'] = train_time_gru

# 6. 最终性能对比
print("\n=== 模型性能对比 ===")
print(f"{'Model':<10}{'Accuracy':>10}{'Precision':>10}{'Recall':>10}{'F1':>10}{'AUC':>10}{'Train Time':>12}{'Eval Time':>12}")
for name, metrics in results.items():
    print(f"{name:<10}{metrics['accuracy']:>10.4f}{metrics['precision']:>10.4f}{metrics['recall']:>10.4f}{metrics['f1']:>10.4f}{metrics['auc']:>10.4f}{metrics['train_time']:>12.4f}{metrics['eval_time']:>12.4f}")

# 7. 内存占用总结
final_memory = get_memory_usage()
print(f"\n=== 内存占用总结 ===")
print(f"初始内存: {initial_memory:.2f} MB")
print(f"峰值内存: {final_memory:.2f} MB")
print(f"总内存增量: {final_memory - initial_memory:.2f} MB")