import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score


# 读取CSV文件
df = pd.read_csv('docs_for_lsh.csv')


def create_hash_func(size):
    # function for creating the hash vector/function
    hash_ex = list(range(1, size))
    shuffle(hash_ex)
    return hash_ex


def build_minhash_func(vocab_size, n):
    # function for building multiple minhash vectors
    hashes = []
    for i in range(n):
        hashes.append(create_hash_func(vocab_size))
    return hashes


def create_hash(vocab_size, vector, minhash_func):
    # use this function for creating the signatures
    signature = []
    for func in minhash_func:
        for i in range(vocab_size - 1):
            idx = func[i]
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(idx)
                break
    return signature


vocab_size = 201
n = 30
minhash_func = build_minhash_func(vocab_size, n)

signature_matrix = []

for i in range(df.shape[0]):
    signature_matrix.append(create_hash(vocab_size, df.iloc[i], minhash_func))
    if (i % 10000 == 0):
        print(f"{i} has done")

# LSH 哈希函数
def lsh_hash(signature_matrix, b, r):
    n_rows, n_cols = len(signature_matrix), len(signature_matrix[0])
    band_hashes = defaultdict(set)
    band_hash0 = set()
    for i in range(n_rows):  # 对每个文档
        for j in range(b):  # 对每个 band
            # 提取当前 band 的签名
            band = tuple(signature_matrix[i][j * r:(j + 1) * r])
            # 为当前 band 生成哈希值
            band_hash = hashlib.md5(str(band).encode('utf-8')).hexdigest()
            if (i == 0):
                band_hash0.add(band_hash)
            # 将文档映射到对应的哈希桶
            band_hashes[band_hash].add(i)
    band_hashes = {k: list(v) for k, v in band_hashes.items()}

    return band_hashes, band_hash0


# 选取五组不同的 b 和 r 值
parameters = [(5, 6), (10, 3), (15, 2), (20, 1.5), (30, 1)]
results = []

# 计算真实的相似文档集合
y_true = df.iloc[0][1:]
true_similars = set()
for i in range(1, df.shape[0]):
    y_pred = df.iloc[i][1:]
    score = jaccard_score(y_true, y_pred)
    if score > 0.8:
        true_similars.add(i)

for b, r in parameters:
    band_hashes, band_hash0 = lsh_hash(signature_matrix, b, int(r))
    y_true = df.iloc[0][1:]
    lsh_similars = set()
    for t in band_hash0:
        for i in band_hashes[t]:
            if i == 0:
                continue
            y_pred = df.iloc[i][1:]
            score = jaccard_score(y_true, y_pred)
            if score > 0.8:
                lsh_similars.add(i)

    # 计算误判概率
    false_positives = lsh_similars - true_similars
    false_negative = true_similars - lsh_similars
    total_docs = df.shape[0] - 1
    false_positive_rate = len(false_positives) / total_docs

    result = {
        'b': b,
        'r': r,
        'false_positive_rate': false_positive_rate
    }
    results.append(result)
    print(f"b = {b}, r = {r}, 误判概率: {false_positive_rate}")

# 可视化不同参数下的误判概率
b_values = [result['b'] for result in results]
false_positive_rates = [result['false_positive_rate'] for result in results]

plt.figure(figsize=(10, 6))
plt.bar([str((b, r)) for b, r in zip(b_values, [result['r'] for result in results])], false_positive_rates)
plt.title('不同 b 和 r 参数下的误判概率')
plt.xlabel('(b, r) 参数组合')
plt.ylabel('误判概率')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()