import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# 数据加载与预处理
df = pd.read_csv('wine-clustering.csv')
df = df.select_dtypes(include='number')  # 仅保留数值列

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# PCA降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='rainbow', edgecolor='k', s=50)
plt.title('Hierarchical Clustering Results (k=3)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 计算轮廓系数
silhouette_hierarchical = silhouette_score(X_scaled, hierarchical_labels)
print(f"层次聚类轮廓系数: {silhouette_hierarchical:.3f}")

# 聚类大小统计
cluster_sizes = np.bincount(hierarchical_labels)
for i, size in enumerate(cluster_sizes):
    print(f"聚类 {i} 大小: {size}")

# 添加聚类标签到原始数据
df['hierarchical_cluster'] = hierarchical_labels

# 聚类特征分析
print("\n各聚类的特征均值:")
print(df.groupby('hierarchical_cluster').mean())

# 绘制树状图
plt.figure(figsize=(12, 6))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, truncate_mode='lastp', p=20, show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.show()