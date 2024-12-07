# Importing libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("top_expensive_leagues.csv")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Revenue (USD)']])

# Clustering using KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Visualizing clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data_scaled[:, 0],
    y=[0] * len(data_scaled),  # Keep y constant
    hue=data['Cluster'],
    palette='viridis',
    s=80,  # Set size of points
    alpha=0.6  # Add transparency
)

# Plot cluster centers
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], [0] * len(cluster_centers), 
            c='red', s=200, marker='*', label='Cluster Centers')

plt.title("Hasil Clustering Liga Olahraga Berdasarkan Pendapatan")
plt.xlabel("Revenue (USD) (Normalized)")
plt.ylabel("")  # Empty y-label
plt.yticks([])  # Remove y-axis ticks
plt.legend(loc="upper left")
plt.grid(True)
plt.show()