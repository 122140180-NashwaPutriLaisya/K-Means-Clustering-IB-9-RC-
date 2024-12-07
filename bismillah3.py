import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Membaca dataset asli
data = pd.read_csv("top_expensive_leagues.csv")

# Normalisasi atribut Revenue (USD)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['Revenue (USD) Normalized'] = scaler.fit_transform(data[['Revenue (USD)']])

# Data yang akan digunakan
X = data[['Revenue (USD) Normalized']].values

# Inisialisasi jumlah cluster
k = 3

# Inisialisasi centroid secara acak
clusters = {}
np.random.seed(23)
for idx in range(k):
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
    clusters[idx] = {'center': center, 'points': []}

# Fungsi menghitung jarak Euclidean
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Implementasi E-step: Menentukan cluster untuk setiap titik
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = [distance(X[idx], clusters[i]['center']) for i in range(k)]
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(X[idx])
    return clusters

# Implementasi M-step: Memperbarui centroid cluster
def update_clusters(clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            clusters[i]['center'] = points.mean(axis=0)
        clusters[i]['points'] = []  # Reset points setelah diperbarui
    return clusters

# Prediksi cluster untuk data baru
def pred_cluster(X, clusters):
    pred = []

    for i in range(X.shape[0]):
        dist = [distance(X[i], clusters[j]['center']) for j in range(k)]
        pred.append(np.argmin(dist))
    return pred

# Iterasi K-Means
for iteration in range(10):  # Iterasi maksimal
    clusters = assign_clusters(X, clusters)
    clusters = update_clusters(clusters)

# Prediksi akhir
pred = pred_cluster(X, clusters)

# Visualisasi hasil clustering
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=pred, cmap='viridis', s=100, label="Data Points")
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], 0, marker='*', color='red', s=200, label=f"Cluster {i+1} Center" if i == 0 else "")

plt.title("Hasil Clustering Liga Olahraga Berdasarkan Pendapatan")
plt.xlabel("Revenue (USD) (Normalized)")
plt.legend()
plt.grid()
plt.show()