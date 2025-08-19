import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Load Dataset
df = pd.read_csv("C:\\Users\\ASUS\\Documents\\Tugas Akhir\\data\\Youtube-Spam-Dataset-TFIDF11.csv")  # Ganti dengan path file yang benar

# Pastikan kolom target bernama 'CLASS'
if 'CLASS' not in df.columns:
    raise ValueError("Kolom target 'CLASS' tidak ditemukan dalam dataset.")

# Split features and target
X = df.drop(columns=['CLASS'])
y = df['CLASS']

# Nilai k yang akan digunakan
k_value = 19 # Anda bisa mengganti nilai k sesuai kebutuhan

# Simpan hasil dari setiap iterasi
results = []
all_reports = []
all_conf_matrices = []

# Perulangan sebanyak 20 kali dengan random_state berbeda
for i in range(1, 21):  # Loop 20 kali dengan random_state dari 1 sampai 20
    
    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    # Inisialisasi model KNN dengan nilai k yang telah ditentukan
    knn = KNeighborsClassifier(n_neighbors=k_value, metric='manhattan')
    knn.fit(X_train, y_train)
    
    # Evaluasi model pada data uji
    y_pred_test = knn.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)
    
    # Evaluasi model pada data latih
    y_pred_train = knn.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    # Simpan hasil iterasi
    results.append((i, k_value, accuracy_test))
    all_reports.append((i, class_report))
    all_conf_matrices.append((i, conf_matrix))
    
    # Output hasil setiap iterasi
    print(f"Train Set Accuracy: {accuracy_train:.4f}")
    # Menampilkan confusion matrix pada data latih
    print("Train Set Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred_train))
    print(f"Iterasi {i}: k={k_value}, Test Accuracy={accuracy_test:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print("-" * 50)

# Menampilkan hasil akhir
print("\nHasil Akhir dari 20 Iterasi:")
for res in results:
    print(f"Iterasi {res[0]} - k: {res[1]}, Test Accuracy: {res[2]:.4f}")
