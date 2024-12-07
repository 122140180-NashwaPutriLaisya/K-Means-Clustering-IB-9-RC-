import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca dataset asli
data = pd.read_csv("top_expensive_leagues.csv")
print("Informasi Dataset:")
print(data.info())


# Normalisasi atribut Revenue (USD) dan Viewership
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Revenue (USD)', 'Viewership']])

# Validasi jumlah cluster
k = 3

# Clustering menggunakan KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Menambahkan jenis cluster berdasarkan kombinasi atribut
data['Cluster_Type'] = data['Cluster'].map({
    0: 'Low Revenue, Low Viewership',
    1: 'High Revenue, Low Viewership',
    2: 'High Revenue, High Viewership'
})

# Menampilkan jumlah data per cluster dan jenisnya
print("\nDistribusi Data per Cluster dengan Jenisnya:")
print(data.groupby('Cluster_Type')['Cluster'].count())

# Visualisasi clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data_scaled[:, 0],
    y=data_scaled[:, 1],  # Gunakan Viewership untuk sumbu Y
    hue=data['Cluster'],
    palette='viridis',
    s=80,  # Ukuran titik
    alpha=0.6  # Transparansi
)

# Plot pusat cluster
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
            c='red', s=200, marker='*', label='Cluster Centers')

# Menambahkan anotasi pusat cluster
for i, center in enumerate(cluster_centers):
    plt.text(center[0], center[1] + 0.005, f"Cluster {i+1}", color='red', fontsize=10, ha='center')

# Menambahkan label dan gaya plot
plt.title("Hasil Clustering Liga Olahraga Berdasarkan Pendapatan dan Penayangan")
plt.xlabel("Revenue (USD) (Normalized)")
plt.ylabel("Viewership (Normalized)")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()