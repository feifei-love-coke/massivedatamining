import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

#  Step 1: Load your dataset
df = pd.read_csv('wine-clustering.csv')  # Update with your file name
df = df.select_dtypes(include='number')  # Use only numeric columns

#  Step 2: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#  Step 3: K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Step 4: Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

#  Step 5: Visualize K-Means clusters (2D with PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("K-Means Clusters")

plt.subplot(1,2,2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='rainbow')
plt.title("Hierarchical Clusters")

plt.show()

linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 4))

# K-Means Clusters
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='Set1')
plt.title("K-Means Clustering")

# Hierarchical Clusters
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='Set2')
plt.title("Hierarchical Clustering")

plt.show()

from sklearn.metrics import silhouette_score

sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
sil_hier = silhouette_score(X_scaled, hierarchical_labels)

print(f"K-Means Silhouette Score: {sil_kmeans:.3f}")
print(f"Hierarchical Silhouette Score: {sil_hier:.3f}")

import numpy as np
print("K-Means cluster sizes:", np.bincount(kmeans_labels))
print("Hierarchical cluster sizes:", np.bincount(hierarchical_labels))

df['kmeans_cluster'] = kmeans_labels
df['hier_cluster'] = hierarchical_labels

# K-Means Cluster Summary
print("K-Means Cluster Description:")
display(df.groupby('kmeans_cluster').mean())

# Hierarchical Cluster Summary
print("Hierarchical Cluster Description:")
display(df.groupby('hier_cluster').mean())

print("K-Means Cluster Sizes:")
print(df['kmeans_cluster'].value_counts())

print("Hierarchical Cluster Sizes:")
print(df['hier_cluster'].value_counts())

