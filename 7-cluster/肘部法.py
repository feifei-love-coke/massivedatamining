import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与预处理
df = pd.read_csv('wine-clustering.csv')
df = df.select_dtypes(include='number')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 使用肘部法确定最佳簇数量
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    if k > 1:
        labels = kmeans.labels_
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

plt.plot(k_range, wcss, 'bo-')
#plt.title('肘部法确定最佳簇数量')
plt.xlabel('簇的数量 (k)')
plt.ylabel('WCSS (惯性)')
plt.grid(True)
plt.show()

plt.plot(k_range, silhouette_scores, 'ro-')
#plt.title('轮廓系数与簇数量关系')
plt.xlabel('簇的数量 (k)')
plt.ylabel('轮廓系数')
plt.grid(True)
plt.show()

# 确定肘部法的"拐点"
# 计算相邻点之间的斜率变化率
diffs = []
for i in range(1, len(wcss)-1):
    diff = abs((wcss[i+1] - wcss[i]) / (wcss[i] - wcss[i-1]))
    diffs.append(diff)

best_k_index = np.argmax(diffs) + 2
print(f"肘部法建议的最佳簇数量: {best_k_index}")

best_silhouette_k = k_range[np.argmax(silhouette_scores)]
print(f"轮廓系数最高的簇数量: {best_silhouette_k}")