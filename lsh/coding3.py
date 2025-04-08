import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

plt.rcParams['font.sans-serif'] = ['SimHei']
# 读取数据
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

# 只处理前 200 个文件
num_files = min(100, df.shape[0])
for i in range(num_files):
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

b = 10
r = 3
band_hashes, band_hash0 = lsh_hash(signature_matrix, b, r)

# 计算前 200 个文件之间的相似度
similarity_matrix = np.zeros((num_files, num_files))

for i in range(num_files):   # 热力图绘制的数据准备
    for j in range(i, num_files):
        y_true = df.iloc[i][1:]
        y_pred = df.iloc[j][1:]
        score = jaccard_score(y_true, y_pred)
        similarity_matrix[i, j] = score
        similarity_matrix[j, i] = score

# 绘制热力图
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('文件相似度热力图（前 100 个文件）')
plt.xlabel('文件索引')
plt.ylabel('文件索引')
plt.show()