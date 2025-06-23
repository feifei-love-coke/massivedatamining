import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据加载与预处理
df = pd.read_csv('wine-clustering.csv')
df = df.select_dtypes(include='number')  # 仅保留数值列

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"K={k}的轮廓系数: {silhouette:.3f}")

cluster_sizes = np.bincount(kmeans_labels)
for i, size in enumerate(cluster_sizes):
    print(f"聚类 {i} 大小: {size}")

df['kmeans_cluster'] = kmeans_labels

# PCA降维可视化 (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50)
#plt.title(f'K-Means聚类结果 (k={k}) - PCA 2D')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.colorbar(label='聚类')
plt.tight_layout()
plt.show()

# PCA降维可视化 (3D)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
    c=kmeans_labels, cmap='viridis', edgecolor='k', s=50
)
#ax.set_title(f'K-Means聚类结果 (k={k}) - PCA 3D')
ax.set_xlabel('主成分 1')
ax.set_ylabel('主成分 2')
ax.set_zlabel('主成分 3')
fig.colorbar(scatter, ax=ax, label='聚类')
plt.tight_layout()
plt.show()

# t-SNE降维可视化
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50)
#plt.title(f'K-Means聚类结果 (k={k}) - t-SNE')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='聚类')
plt.tight_layout()
plt.show()

# 特征重要性分析 - 计算各聚类的特征均值
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=df.columns.drop('kmeans_cluster')
)
cluster_centers['聚类大小'] = cluster_sizes

# 特征重要性热图
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_centers.drop('聚类大小', axis=1), annot=True, cmap='coolwarm', fmt='.2f')
#plt.title('各聚类的特征均值热图')
plt.tight_layout()
plt.show()

# 特征重要性雷达图 (每个聚类的特征分布)
features = df.columns.drop('kmeans_cluster')
n_features = len(features)
angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

fig, axes = plt.subplots(1, k, figsize=(5*k, 5), subplot_kw=dict(polar=True))

for i in range(k):
    values = cluster_centers.iloc[i, :-1].tolist()  # 排除最后一列"聚类大小"
    values += values[:1]  # 闭合雷达图
    
    if k == 1:
        ax = axes
    else:
        ax = axes[i]
    
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), features, fontsize=10)
    ax.set_ylim(0, cluster_centers.iloc[:, :-1].values.max() * 1.1)
    #ax.set_title(f'聚类 {i} (大小: {cluster_sizes[i]})', fontsize=12)

plt.tight_layout()
plt.show()

# 聚类特征分析
print("\n各聚类的特征均值:")
print(cluster_centers)