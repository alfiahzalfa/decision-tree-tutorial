# decision-tree-tutorial

## Apa itu Decision Tree?

Decision Tree (Pohon Keputusan) adalah salah satu algoritma machine learning yang paling populer dan mudah dipahami. Decision tree adalah model pembelajaran yang menggunakan struktur mirip pohon untuk membuat keputusan berdasarkan fitur-fitur (features) dari data.

**Karakteristik Decision Tree:**
- Struktur berbentuk pohon dengan nodes (simpul) dan branches (cabang)
- Setiap node internal mewakili pertanyaan tentang suatu fitur
- Setiap branch mewakili hasil dari pertanyaan tersebut
- Setiap leaf (daun) mewakili keputusan atau hasil akhir
- Mudah divisualisasikan dan diinterpretasikan

**Keuntungan Decision Tree:**
- Mudah dipahami dan dijelaskan
- Tidak memerlukan normalisasi data
- Dapat menangani data numerik dan kategorikal
- Dapat mendeteksi hubungan non-linear
- Efisien untuk prediksi

**Kerugian Decision Tree:**
- Rentan terhadap overfitting
- Dapat menjadi kompleks untuk dataset besar
- Tidak stabil terhadap perubahan kecil pada data

## Cara Membangun Decision Tree

### 1. **Persiapan Data**
   - Kumpulkan dan bersihkan dataset Anda
   - Pisahkan fitur (features) dan target (label)
   - Bagi data menjadi training set dan testing set (misal: 80-20)

### 2. **Memilih Kriteria Pemisahan (Splitting Criteria)**
   Decision tree menggunakan dua metode utama untuk memilih fitur terbaik:
   
   - **Information Gain (Entropy)**: Mengukur pengurangan ketidakpastian setelah pemisahan
   - **Gini Impurity**: Mengukur kemungkinan kesalahan klasifikasi
   
   Fitur yang memberikan information gain atau Gini impurity terendah dipilih sebagai node.

### 3. **Proses Rekursif Membangun Pohon**
   - Mulai dari root node (simpul akar) dengan semua data
   - Cari fitur dan nilai threshold terbaik untuk membagi data
   - Bagi data menjadi dua kelompok berdasarkan kondisi
   - Ulangi proses untuk setiap subset hingga kondisi berhenti terpenuhi

### 4. **Kondisi Berhenti**
   - Semua sampel dalam node termasuk satu class
   - Kedalaman pohon mencapai maximum depth
   - Jumlah sampel dalam node lebih kecil dari minimum samples
   - Tidak ada peningkatan information gain

### 5. **Prediksi**
   - Untuk prediksi data baru, ikuti path dari root ke leaf node
   - Keputusan di setiap node ditentukan berdasarkan nilai fitur
   - Hasil di leaf node adalah prediksi akhir

### 6. **Evaluasi Model**
   - Hitung akurasi pada test set
   - Gunakan metrics seperti precision, recall, F1-score
   - Cek apakah model mengalami overfitting

### Langkah Implementasi dengan Python:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Persiapan data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Buat dan latih model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# 3. Prediksi
y_pred = dt_model.predict(X_test)

# 4. Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy}")
```
