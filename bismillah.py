# Langkah 1: Impor Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Langkah 2: Memuat Dataset
data = pd.read_csv('top_expensive_leagues.csv')
print(data.head())
print(data.info())

print(data.columns)

# Langkah 3: Eksplorasi Data
# Periksa missing values
print(data.isnull().sum())

# Langkah 4: Visualisasi Data
sns.histplot(data['Revenue (USD)'], kde=True)
plt.title('Distribusi Pendapatan Liga Olahraga')
plt.xlabel('Pendapatan')
plt.ylabel('Frekuensi')
plt.show()

sns.countplot(y=data['Sport'], palette='viridis')
plt.title('Distribusi Liga Berdasarkan Jenis Olahraga')
plt.xlabel('Jumlah')
plt.ylabel('Jenis Olahraga')
plt.show()