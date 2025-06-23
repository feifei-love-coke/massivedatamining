import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据加载与预处理
df = pd.read_csv('wine-clustering.csv')
df = df.select_dtypes(include='number')

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 层次聚类
n_clusters = 3
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

silhouette_hierarchical = silhouette_score(X_scaled, hierarchical_labels)
print(f"层次聚类轮廓系数: {silhouette_hierarchical:.3f}")

cluster_sizes = np.bincount(hierarchical_labels)
for i, size in enumerate(cluster_sizes):
    print(f"聚类 {i} 大小: {size}")

df['hierarchical_cluster'] = hierarchical_labels

cluster_centers = []
for i in range(n_clusters):
    cluster_data = X_scaled[hierarchical_labels == i]
    cluster_center = np.mean(cluster_data, axis=0)
    cluster_centers.append(cluster_center)
cluster_centers = np.array(cluster_centers)
cluster_centers_pca = pca.transform(cluster_centers)

plt.figure(figsize=(10, 8))
colors = ['#00A1FF', '#5ed935', '#f8ba00', '#ff2501', '#d31876', '#919292']
for i in range(n_clusters):
    plt.scatter(
        X_pca[hierarchical_labels == i, 0],
        X_pca[hierarchical_labels == i, 1],
        c=colors[i],
        edgecolor='k',
        s=80,
        alpha=0.8,
        label=f'聚类 {i} (n={cluster_sizes[i]})'
    )

plt.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    c='black',
    marker='*',
    s=300,
    edgecolor='white',
    linewidth=2,
    label='聚类中心'
)

plt.legend(fontsize=10, framealpha=0.9)
plt.xlabel('主成分1', fontsize=12)
plt.ylabel('主成分2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.figure(figsize=(12, 8))
linked = linkage(X_scaled, method='ward')
dendrogram(
    linked,
    truncate_mode='lastp',
    p=30, 
    show_contracted=True,
    leaf_rotation=90,
    leaf_font_size=8,
    show_leaf_counts=True
)
plt.xlabel('样本索引', fontsize=12)
plt.ylabel('距离', fontsize=12)
plt.tight_layout()

cluster_means = df.groupby('hierarchical_cluster').mean()
features = df.columns.drop('hierarchical_cluster')
n_features = len(features)

print("\n各聚类的特征均值:")
print(df.groupby('hierarchical_cluster').mean())