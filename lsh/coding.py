import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('docs_for_lsh.csv') # 读取文件

def create_hash_func(size):  # 创建哈希函数
    hash_ex = list(range(1, size))
    shuffle(hash_ex)
    return hash_ex

def build_minhash_func(vocab_size, n):  # 最小哈希函数 P68
    hashes = []
    for i in range(n):
        hashes.append(create_hash_func(vocab_size))
    return hashes


def create_hash(vocab_size, vector, minhash_func):  #哈希函数签名 P69
    signature = []
    for func in minhash_func:
        for i in range(vocab_size - 1):
            idx = func[i]
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(idx)
                break
    return signature


vocab_size = 201 # 大小
n = 30           # 数量
minhash_func = build_minhash_func(vocab_size, n)

signature_matrix = []

for i in range(df.shape[0]): # df.shape返回表格的形式（200,999999）
    signature_matrix.append(create_hash(vocab_size, df.iloc[i], minhash_func))
    if (i % 10000 == 0):
        print(f"{i} has done") # 读取文件中各行的数据

# LSH 哈希函数 P85
def lsh_hash(signature_matrix, b, r):
    n_rows, n_cols = len(signature_matrix), len(signature_matrix[0])
    band_hashes = defaultdict(set)
    band_hash0 = set()
    for i in range(n_rows):
        for j in range(b):
            band = tuple(signature_matrix[i][j * r:(j + 1) * r])
            band_hash = hashlib.md5(str(band).encode('utf-8')).hexdigest()
            if (i == 0):
                band_hash0.add(band_hash)
            band_hashes[band_hash].add(i)
    band_hashes = {k: list(v) for k, v in band_hashes.items()}

    return band_hashes, band_hash0


b = 10 # LSH函数参数，需要进行讨论
r = 3
band_hashes, band_hash0 = lsh_hash(signature_matrix, b, r)

y_true = df.iloc[0][1:]
similars = set()  # 相似度
similarity_scores = []  # 存放相似度
for t in band_hash0:
    for i in band_hashes[t]:
        if i == 0:
            continue
        y_pred = df.iloc[i][1:]
        score = jaccard_score(y_true, y_pred)
        similarity_scores.append(score)
        if (score > 0.8):  # 0.8作为分界点
            similars.add(i)

print(len(similars))
print(similars)

true_labels = [1 if jaccard_score(y_true, df.iloc[i][1:]) > 0.8 else 0 for i in range(1, df.shape[0])]
pred_labels = [1 if i in similars else 0 for i in range(1, df.shape[0])]

precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")   # 分类模型指标，召回率、准确率、F1值

plt.figure(figsize=(10, 6))
plt.hist(similarity_scores, bins=20, edgecolor='k')
plt.title('Jaccard Similarity Distribution')
plt.xlabel('Jaccard Similarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()  # 画图