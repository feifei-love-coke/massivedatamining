import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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

# 定义要测试的簇数量范围
cluster_range = range(2, 11)  # 测试2到10个簇
silhouette_scores = []
best_score = -1
best_n_clusters = 0

# 计算不同簇数量下的轮廓系数
for n_clusters in cluster_range:
    # 层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = hierarchical.fit_predict(X_scaled)
    
    # 计算轮廓系数
    score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(score)
    
    # 记录最佳轮廓系数及其对应的簇数量
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters
    
    # 打印当前簇数量的轮廓系数
    print(f"簇数量: {n_clusters}, 轮廓系数: {score:.4f}")

# 可视化轮廓系数随簇数量的变化
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, 'bo-', linewidth=2)
plt.xlabel('簇数量', fontsize=12)
plt.ylabel('轮廓系数', fontsize=12)
#plt.title(f'层次聚类轮廓系数随簇数量变化\n最佳簇数量: {best_n_clusters}, 轮廓系数: {best_score:.4f}', fontsize=14)
plt.xticks(cluster_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 标记最佳点
plt.scatter(best_n_clusters, best_score, color='red', s=100, zorder=5)
plt.annotate(f'最佳: {best_n_clusters}簇', 
             xy=(best_n_clusters, best_score), 
             xytext=(best_n_clusters+0.5, best_score+0.01),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# 保存轮廓系数分析图
plt.savefig('silhouette_analysis.png', bbox_inches='tight')
plt.close()

print(f"\n最佳簇数量: {best_n_clusters}, 轮廓系数: {best_score:.4f}")    