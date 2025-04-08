import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

plt.rcParams['font.sans-serif'] = ['SimHei'] 
df = pd.read_csv('docs_for_lsh.csv')


def create_hash_func(size):
    hash_ex = list(range(1, size))
    shuffle(hash_ex)
    return hash_ex


def build_minhash_func(vocab_size, n):
    hashes = []
    for i in range(n):
        hashes.append(create_hash_func(vocab_size))
    return hashes


def create_hash(vocab_size, vector, minhash_func):
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
    for i in range(n_rows):
        for j in range(b):
            band = tuple(signature_matrix[i][j * r:(j + 1) * r])
            band_hash = hashlib.md5(str(band).encode('utf-8')).hexdigest()
            if (i == 0):
                band_hash0.add(band_hash)
            band_hashes[band_hash].add(i)
    band_hashes = {k: list(v) for k, v in band_hashes.items()}

    return band_hashes, band_hash0

parameters = [(5, 6), (10, 3), (15, 2), (20, 1.5), (30, 1)]  # b和r特定参数的选择
results = []

for b, r in parameters:
    band_hashes, band_hash0 = lsh_hash(signature_matrix, b, int(r))
    y_true = df.iloc[0][1:]
    similars = set()
    similarity_scores = []
    for t in band_hash0:
        for i in band_hashes[t]:
            if i == 0:
                continue
            y_pred = df.iloc[i][1:]
            score = jaccard_score(y_true, y_pred)
            similarity_scores.append(score)
            if (score > 0.8):
                similars.add(i)
    result = {
        'b': b,
        'r': r,
        'num_similar_docs': len(similars)
    }
    results.append(result)
    print(f"b = {b}, r = {r}, 相似文档数量: {len(similars)}")

b_values = [result['b'] for result in results]
num_similar_docs = [result['num_similar_docs'] for result in results]

plt.figure(figsize=(10, 6))
plt.bar([str((b, r)) for b, r in zip(b_values, [result['r'] for result in results])], num_similar_docs)
plt.title('不同 b 和 r 参数下的相似文档数量')
plt.xlabel('(b, r) 参数组合')
plt.ylabel('相似文档数量')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()