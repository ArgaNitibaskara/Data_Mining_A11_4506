"""
Tugas Data Mining - Minggu ke-3

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & EXPLORATION
# ==========================================
print("="*60)
print("TAHAP 1: DATA LOADING & EXPLORATION")
print("="*60)

try:
    df = pd.read_csv('musicgenre-small.csv', delimiter=';', decimal=',', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('musicgenre-small.csv', delimiter=';', decimal=',', encoding='latin-1')
    except:
        df = pd.read_csv('musicgenre-small.csv', delimiter=';', decimal=',', encoding='cp1252')

print("\n✓ Dataset berhasil dimuat!")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")

# Tampilkan 5 baris pertama
print("\n5 Baris Pertama Data:")
print(df.head())

# Informasi tipe data
print("\nInformasi Dataset:")
print(df.info())

# Statistik deskriptif
print("\nStatistik Deskriptif:")
print(df.describe())

# Cek nama kolom
print("\nNama Kolom:")
print(df.columns.tolist())

# ==========================================
# 2. HANDLING MISSING VALUES
# ==========================================
print("\n" + "="*60)
print("TAHAP 2: HANDLING MISSING VALUES")
print("="*60)

# Cek missing values
print("\nMissing Values per Kolom:")
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
    
    # Strategi: Imputasi dengan mean untuk kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print("\n✓ Missing values pada kolom numerik telah diisi dengan mean")
    
    # Untuk kolom kategorikal, isi dengan mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    print("✓ Missing values pada kolom kategorikal telah diisi dengan mode")
else:
    print("✓ Tidak ada missing values dalam dataset")

# ==========================================
# 3. DATA CLEANING
# ==========================================
print("\n" + "="*60)
print("TAHAP 3: DATA CLEANING")
print("="*60)

# Cek duplikasi
print(f"\nJumlah baris duplikat: {df.duplicated().sum()}")
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    print("✓ Baris duplikat telah dihapus")
else:
    print("✓ Tidak ada baris duplikat")

# Remove whitespace dari kolom string
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()
print("\n✓ Whitespace pada kolom kategorikal telah dibersihkan")

# Cek nilai yang tidak valid atau anomali
print("\nDistribusi Genre:")
print(df['genre'].value_counts())

# ==========================================
# 4. OUTLIER DETECTION & HANDLING
# ==========================================
print("\n" + "="*60)
print("TAHAP 4: OUTLIER DETECTION & HANDLING")
print("="*60)

# Pilih kolom numerik untuk deteksi outlier
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Metode IQR (Interquartile Range)
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Deteksi outlier pada beberapa fitur penting
important_features = ['loudness', 'tempo', 'duration'] if all(col in numeric_features for col in ['loudness', 'tempo', 'duration']) else numeric_features[:3]

print("\nDeteksi Outlier (Metode IQR):")
for feature in important_features:
    n_outliers, lower, upper = detect_outliers_iqr(df, feature)
    print(f"{feature}: {n_outliers} outliers (Range: {lower:.2f} - {upper:.2f})")

# Opsi 1: Capping outliers (lebih aman daripada menghapus)
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    return data

# Terapkan capping pada fitur penting
df_cleaned = df.copy()
for feature in important_features:
    df_cleaned = cap_outliers(df_cleaned, feature)

print("\n✓ Outliers telah di-handle dengan metode capping")

# ==========================================
# 5. FEATURE SCALING/NORMALIZATION
# ==========================================
print("\n" + "="*60)
print("TAHAP 5: FEATURE SCALING/NORMALIZATION")
print("="*60)

# Buat copy untuk scaling
df_scaled = df_cleaned.copy()

# Pisahkan kolom kategorikal dan numerik
categorical_cols = df_scaled.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

# Metode 1: StandardScaler (Z-score normalization)
scaler_standard = StandardScaler()
df_standard_scaled = df_scaled.copy()
df_standard_scaled[numeric_cols] = scaler_standard.fit_transform(df_scaled[numeric_cols])

print("\n✓ StandardScaler (Z-score) diterapkan pada kolom numerik")
print("Sample data setelah StandardScaler:")
print(df_standard_scaled[numeric_cols].head())

# Metode 2: MinMaxScaler (0-1 normalization)
scaler_minmax = MinMaxScaler()
df_minmax_scaled = df_scaled.copy()
df_minmax_scaled[numeric_cols] = scaler_minmax.fit_transform(df_scaled[numeric_cols])

print("\n✓ MinMaxScaler (0-1) diterapkan pada kolom numerik")
print("Sample data setelah MinMaxScaler:")
print(df_minmax_scaled[numeric_cols].head())

# ==========================================
# 6. ENCODING CATEGORICAL VARIABLES
# ==========================================
print("\n" + "="*60)
print("TAHAP 6: ENCODING CATEGORICAL VARIABLES")
print("="*60)

# Label Encoding untuk kolom genre
if 'genre' in df_cleaned.columns:
    le = LabelEncoder()
    df_encoded = df_standard_scaled.copy()
    df_encoded['genre_encoded'] = le.fit_transform(df_cleaned['genre'])
    
    print("\n✓ Label Encoding diterapkan pada kolom 'genre'")
    print("\nMapping Genre:")
    genre_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    for genre, code in genre_mapping.items():
        print(f"  {genre} → {code}")

# Untuk kolom artist_name dan title, kita bisa drop atau encode
# Karena terlalu banyak unique values, lebih baik di-drop untuk modeling
df_final = df_encoded.copy()
if 'artist_name' in df_final.columns:
    df_final = df_final.drop(['artist_name', 'title', 'genre'], axis=1)
    print("\n✓ Kolom 'artist_name', 'title', dan 'genre' (original) telah dihapus")
    print("  (Menggunakan 'genre_encoded' sebagai target)")

# ==========================================
# 7. EXPORT CLEANED DATA
# ==========================================
print("\n" + "="*60)
print("TAHAP 7: EXPORT CLEANED DATA")
print("="*60)

# Save hasil preprocessing
df_cleaned.to_csv('musicgenre_cleaned.csv', index=False)
print("\n✓ Data cleaned disimpan ke: musicgenre_cleaned.csv")

df_final.to_csv('musicgenre_preprocessed.csv', index=False)
print("✓ Data preprocessed (scaled + encoded) disimpan ke: musicgenre_preprocessed.csv")

# ==========================================
# 8. VISUALISASI (BONUS)
# ==========================================
print("\n" + "="*60)
print("TAHAP 8: VISUALISASI DATA")
print("="*60)

# Buat folder untuk menyimpan visualisasi
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Plot 1: Distribusi Genre
plt.figure(figsize=(12, 6))
df_cleaned['genre'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribusi Genre dalam Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/genre_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi 'genre_distribution.png' disimpan")
plt.close()

# Plot 2: Correlation Heatmap (untuk fitur penting)
plt.figure(figsize=(14, 10))
correlation_features = ['loudness', 'tempo', 'duration', 'avg_timbre1', 'avg_timbre2', 
                        'avg_timbre3', 'var_timbre1', 'var_timbre2', 'var_timbre3']
# Filter hanya kolom yang ada
correlation_features = [col for col in correlation_features if col in df_cleaned.columns]
correlation_matrix = df_cleaned[correlation_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Fitur Penting', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi 'correlation_heatmap.png' disimpan")
plt.close()

# Plot 3: Boxplot untuk deteksi outlier
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Boxplot untuk Deteksi Outlier', fontsize=16, fontweight='bold')

boxplot_features = important_features[:4] if len(important_features) >= 4 else important_features
for idx, feature in enumerate(boxplot_features):
    row = idx // 2
    col = idx % 2
    axes[row, col].boxplot(df_cleaned[feature].dropna(), vert=True)
    axes[row, col].set_title(f'{feature}', fontsize=12)
    axes[row, col].set_ylabel('Value')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/outlier_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi 'outlier_boxplot.png' disimpan")
plt.close()

# ==========================================
# RINGKASAN
# ==========================================
print("\n" + "="*60)
print("RINGKASAN PREPROCESSING")
print("="*60)
print(f"\nDataset Original:")
print(f"  - Baris: {df.shape[0]}")
print(f"  - Kolom: {df.shape[1]}")
print(f"\nDataset Setelah Preprocessing:")
print(f"  - Baris: {df_final.shape[0]}")
print(f"  - Kolom: {df_final.shape[1]}")
print(f"\nFile Output:")
print(f"  1. musicgenre_cleaned.csv - Data setelah cleaning")
print(f"  2. musicgenre_preprocessed.csv - Data siap untuk modeling")
print(f"  3. visualizations/ - Folder berisi visualisasi")
print("\n" + "="*60)
print("PREPROCESSING SELESAI!")
print("="*60)
