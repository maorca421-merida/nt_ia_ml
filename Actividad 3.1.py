import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Generar un conjunto de datos con 3 dimensiones
X, y = make_blobs(n_samples=300, centers=4, n_features=3, cluster_std=1.0, random_state=42)

# Visualización inicial de los datos en 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='gray', marker='o', s=50)
ax.set_title("Datos originales en 3D")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
plt.show()

# Aplicar K-Means en el conjunto de datos original (3D)
kmeans_original = KMeans(n_clusters=4, random_state=42)
kmeans_original.fit(X)
y_kmeans_original = kmeans_original.predict(X)

# Evaluación del clustering en el espacio original
score_original = silhouette_score(X, y_kmeans_original)
print(f"Silhouette Score en espacio original: {score_original:.2f}")

# Visualización de los clusters en 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans_original, cmap='viridis', marker='o', s=50)
ax.set_title("Clusters formados por K-Means en el espacio original (3D)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
plt.show()

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Aplicar K-Means en el espacio reducido
kmeans_reduced = KMeans(n_clusters=4, random_state=42)
kmeans_reduced.fit(X_reduced)
y_kmeans_reduced = kmeans_reduced.predict(X_reduced)

# Evaluación del clustering en el espacio reducido
score_reduced = silhouette_score(X_reduced, y_kmeans_reduced)
print(f"Silhouette Score en espacio reducido: {score_reduced:.2f}")

# Visualización de los clusters en el espacio reducido (2D)
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_kmeans_reduced, cmap='viridis', marker='o', s=50)
plt.title("Clusters formados por K-Means en el espacio reducido (2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

# Comparación de la varianza explicada por las componentes principales
explained_variance = pca.explained_variance_ratio_
print(f"Varianza explicada por cada componente: {explained_variance}")

plt.figure(figsize=(8, 4))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.title("Varianza explicada por las componentes principales")
plt.xlabel("Componente Principal")
plt.ylabel("Proporción de varianza explicada")
plt.show()
