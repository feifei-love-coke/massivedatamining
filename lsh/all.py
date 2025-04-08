import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict
from random import shuffle
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import chardet


# 读取CSV文件
def read_csv_file(file_path):
    try:
        # 读取文件的部分内容以检测编码
        with open(file_path, 'rb') as f:
            rawdata = f.read(100000)  # 读取前100000字节

        # 检测编码
        result = chardet.detect(rawdata)
        encoding = result['encoding']

        # 使用检测到的编码读取文件
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        return None


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
            signature_val = vector.iloc[idx]
            if signature_val == 1:
                signature.append(idx)
                break
    return signature


# LSH 哈希函数
def lsh_hash(signature_matrix, b, r):
    n_rows, n_cols = len(signature_matrix), len(signature_matrix[0])
    band_hashes = defaultdict(set)
    for i in range(n_rows):  # 对每个文档
        for j in range(b):  # 对每个 band
            # 提取当前 band 的签名
            band = tuple(signature_matrix[i][j * r:(j + 1) * r])
            # 为当前 band 生成哈希值
            band_hash = hashlib.md5(str(band).encode('utf-8')).hexdigest()
            # 将文档映射到对应的哈希桶
            band_hashes[band_hash].add(i)
    band_hashes = {k: list(v) for k, v in band_hashes.items()}
    return band_hashes


file_path = "docs_for_lsh.csv"
df = read_csv_file(file_path)

if df is not None:
    vocab_size = 201
    n = 30
    minhash_func = build_minhash_func(vocab_size, n)
    signature_matrix = []

    for i in range(df.shape[0]):
        signature_matrix.append(create_hash(vocab_size, df.iloc[i], minhash_func))
        if (i % 10000 == 0):
            print(f"{i} has done")

    # 这里暂时使用默认的 b 和 r 值，后续可以根据 optimal_b_r.py 的结果修改
    b = 10
    r = 3

    band_hashes = lsh_hash(signature_matrix, b, r)

    # 计算 Jaccard 相似度
    y_true = df.iloc[0][1:]
    similars = set()
    scores = []
    for bucket in band_hashes.values():
        for i in bucket:
            if i == 0:
                continue
            y_pred = df.iloc[i][1:]
            score = jaccard_score(y_true, y_pred)
            scores.append(score)
            if score > 0.8:
                similars.add(i)

    print(len(similars))
    print(similars)

    # 可视化相似度得分分布
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='k')
    plt.title('Jaccard Similarity Scores Distribution')
    plt.xlabel('Jaccard Similarity Score')
    plt.ylabel('Frequency')
    plt.show()

    # 计算评价指标
    # 假设我们有一个真实的相似文档集合 true_similars
    # 这里简单假设前 10 个文档是真实相似的
    true_similars = set(range(1, 11))
    y_true_binary = [1 if i in true_similars else 0 for i in range(1, df.shape[0])]
    y_pred_binary = [1 if i in similars else 0 for i in range(1, df.shape[0])]

    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")