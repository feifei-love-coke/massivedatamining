import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# 设置中文字体和显示效果
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 数据加载与预处理
df = pd.read_csv('wine-clustering.csv')
df = df.select_dtypes(include='number')  # 仅保留数值列

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

# 计算聚类中心（用于PCA图标记）
cluster_centers = []
for i in range(n_clusters):
    cluster_data = X_scaled[hierarchical_labels == i]
    cluster_center = np.mean(cluster_data, axis=0)
    cluster_centers.append(cluster_center)
cluster_centers = np.array(cluster_centers)
cluster_centers_pca = pca.transform(cluster_centers)


# ====================
# 可视化部分（生成图片）
# ====================

# 1. 创建主图：PCA聚类散点图
plt.figure(figsize=(10, 8))

# 绘制散点图，使用不同颜色区分聚类
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

# 绘制聚类中心
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

# 添加图例和标题
plt.legend(fontsize=10, framealpha=0.9)
#plt.title(f'层次聚类结果 (k={n_clusters}, 轮廓系数={silhouette_hierarchical:.3f})', fontsize=14)
plt.xlabel('主成分1', fontsize=12)
plt.ylabel('主成分2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存PCA图
plt.savefig('hierarchical_clustering_pca.png', bbox_inches='tight')
plt.close()


# 2. 绘制树状图（单独保存）
plt.figure(figsize=(12, 8))
linked = linkage(X_scaled, method='ward')
dendrogram(
    linked,
    truncate_mode='lastp',
    p=30,              # 显示最后30个叶节点
    show_contracted=True,  # 压缩距离较远的节点
    leaf_rotation=90,      # 旋转叶节点标签
    leaf_font_size=8,      # 叶节点字体大小
    show_leaf_counts=True  # 显示叶节点数量
)
#plt.title('层次聚类树状图', fontsize=14)
plt.xlabel('样本索引', fontsize=12)
plt.ylabel('距离', fontsize=12)
plt.tight_layout()

# 保存树状图
plt.savefig('hierarchical_clustering_dendrogram.png', bbox_inches='tight')
plt.close()


# 3. 绘制特征雷达图（聚类特征对比）
cluster_means = df.groupby('hierarchical_cluster').mean()
features = df.columns.drop('hierarchical_cluster')
n_features = len(features)

# 准备雷达图数据
theta = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
theta += theta[:1]  # 闭合雷达图

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# 为每个聚类绘制雷达图
for i in range(n_clusters):
    values = cluster_means.loc[i].values.tolist()
    values += values[:1]  # 闭合雷达图
    ax.plot(theta, values, 'o-', linewidth=2, label=f'聚类 {i}')
    ax.fill(theta, values, alpha=0.25)

# 设置雷达图标签
ax.set_thetagrids(np.degrees(theta[:-1]), features)
#plt.title('各聚类的特征均值分布', fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()

# 保存雷达图
plt.savefig('cluster_feature_radar.png', bbox_inches='tight')
plt.close()


# 打印聚类特征均值
print("\n各聚类的特征均值:")
print(df.groupby('hierarchical_cluster').mean())