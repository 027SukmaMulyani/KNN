import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics.pairwise import pairwise_distances

# 1. Load Dataset
df = pd.read_csv("C:\\Users\\ASUS\\Documents\\Tugas Akhir\\Hasil\\Youtube-Spam-Dataset-TFIDF.csv")  # Ganti dengan path file yang benar


# Pastikan kolom target bernama 'CLASS'
if 'CLASS' not in df.columns:
    raise ValueError("Kolom target 'CLASS' tidak ditemukan dalam dataset.")

# Split features and target
X = df.drop(columns=['CLASS'])
y = df['CLASS']

# Binarisasi data TF-IDF (threshold: nilai > 0 menjadi 1, sebaliknya 0)
X_binary = (X > 0).astype(int)

# Nilai k yang akan digunakan
k_value = 1 # Anda bisa mengganti nilai k sesuai kebutuhan

# Simpan hasil dari setiap iterasi
results = []
all_reports = []
all_conf_matrices = []

# Perulangan sebanyak 20 kali dengan random_state berbeda
for i in range(1, 21):  # Loop 20 kali dengan random_state dari 1 sampai 20
    
    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=i, stratify=y)


    # Ubah ke NumPy array
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()

    # Hitung Jaccard distance untuk train
    jaccard_distances_train = pairwise_distances(X_train_np, metric='jaccard')

    # Inisialisasi model KNN dengan precomputed distance matrix
    knn = KNeighborsClassifier(n_neighbors=k_value, metric='precomputed')
    
    # Fit model dengan matriks jarak train
    knn.fit(jaccard_distances_train, y_train)

    # Hitung Jaccard distance untuk test
    X_test_dist = pairwise_distances(X_test_np, X_train_np, metric='jaccard')

    # Evaluasi model pada data uji
    y_pred_test = knn.predict(X_test_dist)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)

    # Evaluasi model pada data latih
    y_pred_train = knn.predict(jaccard_distances_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    # Simpan hasil iterasi
    results.append((i, k_value, accuracy_test))
    all_reports.append((i, class_report))
    all_conf_matrices.append((i, conf_matrix))

    # Output hasil setiap iterasi
    print(f"Train Set Accuracy: {accuracy_train:.4f}")
    print("Train Set Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred_train))
    print(f"Iterasi {i}: k={k_value}, Test Accuracy={accuracy_test:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

  # Menampilkan distribusi kelas pada y_train dan y_test dalam bentuk proporsi
    print(f"Train Set (Proportion) y_train:")
    print(y_train.value_counts(normalize=True))
    print(f"Test Set (Proportion) y_test:")
    print(y_test.value_counts(normalize=True))
    print("-" * 50)

# Menampilkan hasil akhir
print("\nHasil Akhir dari 20 Iterasi:")
for res in results:
    print(f"Iterasi {res[0]} - k: {res[1]}, Test Accuracy: {res[2]:.4f}")
