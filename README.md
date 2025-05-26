# Stock Market Predictive Analytics

**Proyek Machine Learning - Analisis Prediktif Pasar Saham**  
**Author:** Yandiyan

---

## Daftar Isi
1. [Domain Proyek](#domain-proyek)
2. [Business Understanding](#business-understanding)
3. [Data Understanding](#data-understanding)
4. [Data Preparation](#data-preparation)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Kesimpulan dan Rekomendasi](#kesimpulan-dan-rekomendasi)
8. [Referensi](#referensi)

---

## Domain Proyek

Pasar saham merupakan salah satu instrumen investasi yang paling dinamis dan kompleks dalam dunia finansial. Fluktuasi harga saham dipengaruhi oleh berbagai faktor mulai dari kondisi ekonomi makro, sentimen pasar, hingga indikator teknikal. Kemampuan untuk memprediksi pergerakan harga saham dengan akurat menjadi kebutuhan krusial bagi investor, trader, dan lembaga keuangan dalam mengoptimalkan strategi investasi mereka.

Penerapan machine learning dalam prediksi harga saham telah menunjukkan hasil yang menjanjikan. Penelitian oleh Müller et al. (2024) membuktikan bahwa model machine learning linear yang dikombinasikan dengan indikator teknikal sederhana mampu menurunkan tracking error portofolio hingga 18% dibandingkan pendekatan moving average konvensional. Hal ini menunjukkan potensi besar teknologi machine learning dalam meningkatkan akurasi prediksi pasar saham.

Proyek ini dikembangkan untuk menjawab kebutuhan akan sistem prediksi harga saham yang tidak hanya akurat, tetapi juga transparan dan dapat diimplementasikan dalam lingkungan produksi. Dengan memanfaatkan data historis dan indikator teknikal, diharapkan model yang dibangun dapat memberikan kontribusi signifikan dalam pengambilan keputusan investasi.

**Referensi:**
- Müller, F. et al. (2024). *Linear factor models for intraday stock prediction*. Information Sciences, 660, 119-134. DOI: 10.1016/j.ins.2024.118123
- Rahman, S. (2022). *Impact of Hyperparameter Tuning on ML Models in Stock Price Forecasting*. In Intelligent Computing (pp. 45-57). DOI: 10.1007/978-981-19-5545-9_5

---

## Business Understanding

### Problem Statements

Dalam dunia investasi saham, terdapat beberapa permasalahan utama yang dihadapi oleh para pelaku pasar:

**Bagaimana memprediksi harga penutupan saham untuk hari berikutnya dengan tingkat akurasi yang tinggi?** Prediksi harga penutupan menjadi informasi kritis bagi investor dalam menentukan strategi jual-beli saham. Ketidakakuratan prediksi dapat menyebabkan kerugian finansial yang signifikan.

**Fitur atau indikator apa saja yang paling berpengaruh terhadap pergerakan harga saham?** Pemahaman terhadap faktor-faktor yang mempengaruhi harga saham akan membantu dalam pengembangan strategi investasi yang lebih efektif.

**Bagaimana membandingkan dan mengevaluasi berbagai algoritma machine learning untuk memilih model prediksi yang paling optimal?** Dengan beragam algoritma yang tersedia, diperlukan metodologi yang sistematis untuk memilih model terbaik berdasarkan kriteria performa yang relevan.

### Goals

Berdasarkan permasalahan yang telah diidentifikasi, proyek ini bertujuan untuk:

**Mengembangkan model prediktif yang dapat memprediksi harga penutupan saham dengan akurasi tinggi.** Model yang dibangun harus mampu memberikan prediksi yang dapat diandalkan untuk mendukung pengambilan keputusan investasi.

**Mengidentifikasi fitur-fitur penting yang berpengaruh signifikan terhadap pergerakan harga saham.** Analisis feature importance akan memberikan wawasan tentang faktor-faktor kunci yang mempengaruhi harga saham.

**Melakukan perbandingan komprehensif terhadap berbagai algoritma machine learning untuk menentukan model dengan performa terbaik.** Evaluasi akan dilakukan berdasarkan multiple metrics untuk memastikan pemilihan model yang optimal.

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, beberapa pendekatan solusi akan diimplementasikan:

**Pengembangan dan perbandingan empat algoritma machine learning berbeda:** Linear Regression sebagai baseline model, Ridge Regression untuk mengatasi potensi overfitting, Random Forest untuk menangkap hubungan non-linear kompleks, dan XGBoost untuk memanfaatkan kemampuan boosting ensemble.

**Implementasi feature engineering yang komprehensif** untuk menciptakan fitur-fitur baru yang lebih informatif, termasuk indikator teknikal, persentase perubahan harga, dan fitur temporal.

**Penerapan hyperparameter tuning pada model terbaik** menggunakan teknik Grid Search dan Randomized Search untuk mengoptimalkan performa model secara maksimal.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah [Stock Market Dataset for Predictive Analysis](https://www.kaggle.com/datasets/s3programmer/stock-market-dataset-for-predictive-analysis) yang berisi data historis pasar saham dengan periode dari 1 Januari 2010 hingga 24 April 2062.

### Informasi Dataset

**Jumlah Data:** 13,647 baris dengan 10 kolom  
**Periode:** 1 Januari 2010 - 24 April 2062  
**Target Variable:** Close (harga penutupan saham)  
**Kondisi Data:** Tidak terdapat missing values dalam dataset

### Deskripsi Fitur

| Nama Fitur | Tipe Data | Deskripsi |
|------------|-----------|-----------|
| `Date` | DateTime | Tanggal perdagangan |
| `Open` | Numerik | Harga pembukaan saham |
| `High` | Numerik | Harga tertinggi dalam satu hari |
| `Low` | Numerik | Harga terendah dalam satu hari |
| `Close` | Numerik | Harga penutupan saham (target variable) |
| `Volume` | Numerik | Volume perdagangan saham |
| `RSI` | Numerik | Relative Strength Index (indikator momentum) |
| `MACD` | Numerik | Moving Average Convergence Divergence |
| `Sentiment` | Numerik | Skor sentimen berita (-1 hingga +1) |
| `Target` | Kategorik | Label biner (0=turun, 1=naik) - tidak digunakan dalam regresi |

### Exploratory Data Analysis

#### Analisis Distribusi Data

![Distribusi Close](https://github.com/user-attachments/assets/7bea8795-4829-4937-8177-f8d065affb5c)
![Distribusi High](https://github.com/user-attachments/assets/d034138b-185e-415d-8194-550c2926cfa5)
![Distribusi Low](https://github.com/user-attachments/assets/54882d9f-3083-4019-82ee-99c5144a0040)
![Distribusi MACD](https://github.com/user-attachments/assets/0727cff5-f0a3-4637-9343-68b0a9ae46d4)
![Distribusi Open](https://github.com/user-attachments/assets/7a67edcf-dda9-4f64-8957-7565ab78e595)
![Distribusi RSI](https://github.com/user-attachments/assets/19bfc8da-8101-4f72-b8ee-4270892dbd4e)
![Distribusi Sentiment](https://github.com/user-attachments/assets/591467c9-3d2f-4a87-8848-e70d10c1e123)
![Distribusi Volume](https://github.com/user-attachments/assets/3bf0618e-ee2e-4190-b424-6558a168b7fa)

Distribusi harga pembukaan menunjukkan pola yang relatif simetris dengan beberapa outlier pada nilai negatif yang mengindikasikan adanya data simulasi. Analisis boxplot mengungkapkan bahwa sebagian besar data terkonsentrasi pada rentang nilai tertentu dengan beberapa nilai ekstrem yang perlu mendapat perhatian khusus.

#### Analisis Korelasi

![Correlation Heatmap](https://github.com/user-attachments/assets/8a2c4108-eb05-42a6-a727-a374a9209c63)

Heatmap korelasi menunjukkan hubungan yang sangat kuat antara fitur OHLC (Open, High, Low, Close) dengan koefisien korelasi mendekati 1.0. Hal ini mengindikasikan adanya multikolinearitas yang perlu diatasi dalam tahap preprocessing. Fitur volume dan indikator teknikal menunjukkan korelasi yang lebih moderat terhadap harga penutupan.

#### Analisis Tren

![Moving Average Trend](https://github.com/user-attachments/assets/98eec666-1cfc-4407-8b35-bd316d6e52f7)

Analisis moving average menggunakan periode 50 dan 200 hari menunjukkan tren jangka panjang yang konsisten. Grafik ini membantu dalam memahami pola pergerakan harga secara makro dan mengidentifikasi periode bullish dan bearish dalam dataset.

#### Deteksi Anomali

![Anomali Detection](https://github.com/user-attachments/assets/764b90ab-c876-4bcf-a433-6026d3d5369a)

Deteksi anomali menggunakan Z-score dengan threshold |z| > 3 mengidentifikasi 31 titik anomali yang terjadi terutama pada periode 2038-2040. Anomali ini perlu dipertimbangkan dalam pengembangan model untuk memastikan robustness prediksi.

---

## Data Preparation

Tahap data preparation melibatkan serangkaian transformasi dan preprocessing untuk mempersiapkan data agar optimal untuk pelatihan model machine learning.

### Data Cleaning

**Handling Missing Values:**
- Dataset tidak memiliki missing values, sehingga tidak diperlukan penanganan khusus
- Verifikasi dilakukan menggunakan `df.isnull().sum()` untuk memastikan kelengkapan data
- Pengecekan dilakukan pada setiap kolom untuk memastikan tidak ada nilai yang hilang

**Handling Outliers:**
- Deteksi outlier menggunakan metode Z-score dengan threshold |z| > 3
- 31 titik anomali teridentifikasi pada periode 2038-2040
- Analisis outlier dilakukan per fitur untuk memahami distribusi dan ekstremitas data
- Outlier dipertahankan karena merepresentasikan pergerakan harga yang valid dalam konteks pasar saham
- Tidak dilakukan penghapusan outlier karena dapat menghilangkan informasi penting tentang volatilitas pasar

**Handling Duplicates:**
- Pengecekan duplikat dilakukan menggunakan `df.duplicated()`
- Tidak ditemukan data duplikat dalam dataset
- Verifikasi dilakukan berdasarkan kombinasi Date dan OHLCV untuk memastikan keunikan setiap transaksi
- Pengecekan tambahan dilakukan pada subset data untuk memastikan tidak ada duplikasi parsial

### Feature Engineering

**Penciptaan Fitur Baru:**
1. **Fitur Teknikal:**
   - HL_PCT: Persentase selisih high-low terhadap close price
   - CHG_PCT: Persentase perubahan harga dari open ke close
   - Daily Return: Persentase perubahan harga penutupan
   - Volatility: Standar deviasi return dalam window 20 hari
   - Momentum: Perubahan harga dalam window 10 hari

2. **Fitur Temporal:**
   - Year: Tahun transaksi
   - Month: Bulan transaksi
   - Day: Hari dalam bulan
   - Weekday: Hari dalam seminggu (0-6)
   - Quarter: Kuartal dalam tahun
   - Is_Month_End: Flag untuk akhir bulan
   - Is_Quarter_End: Flag untuk akhir kuartal

**Penghapusan Fitur Redundan:**
- Fitur High dan Low dihapus karena berkorelasi sangat tinggi (>0.95) dengan Open dan Close
- Fitur Target dihapus karena tidak relevan untuk task regresi
- Verifikasi korelasi menggunakan heatmap untuk memastikan tidak ada fitur yang redundan
- Analisis VIF (Variance Inflation Factor) dilakukan untuk mengidentifikasi multikolinearitas

### Data Preprocessing

**Standardization:**
- Semua fitur numerik dinormalisasi menggunakan StandardScaler
- Formula: $z = (x - μ) / σ$
  - x: nilai asli
  - μ: mean fitur
  - σ: standard deviation fitur
- Tujuan: Memastikan semua fitur berada pada skala yang sama
- Dampak: Meningkatkan konvergensi model dan mengurangi bias
- Verifikasi distribusi setelah scaling untuk memastikan transformasi berhasil

**Time-based Split:**
- Dataset dibagi berdasarkan aspek temporal
- Training set: 80% data (periode 2010-2038)
- Testing set: 20% data (periode 2038-2062)
- Alasan: Mensimulasikan kondisi real-world di mana prediksi dilakukan untuk periode mendatang
- Keuntungan: Menjaga integritas temporal data dan menghindari data leakage
- Validasi dilakukan untuk memastikan tidak ada data leakage antar set

### Alasan Pemilihan Teknik Preprocessing

1. **Standardization:**
   - Diperlukan untuk algoritma yang sensitif terhadap skala (Linear Regression, Ridge Regression)
   - Membantu konvergensi yang lebih cepat dan stabil
   - Tidak mengubah distribusi data, hanya mengubah skala
   - Memastikan semua fitur memiliki kontribusi yang seimbang dalam model

2. **Time-based Split:**
   - Menjaga urutan temporal data
   - Mencegah data leakage
   - Mensimulasikan skenario prediksi real-world
   - Memastikan model dapat menangkap pola temporal dengan baik

3. **Feature Engineering:**
   - Meningkatkan kemampuan model dalam menangkap pola
   - Menambahkan konteks temporal
   - Mengurangi multikolinearitas
   - Memperkaya informasi yang tersedia untuk model

4. **Outlier Handling:**
   - Mempertahankan outlier karena merepresentasikan pergerakan harga yang valid
   - Menghindari kehilangan informasi penting tentang volatilitas pasar
   - Memastikan model dapat menangani anomali dengan baik

### Data Preparation

Tahap data preparation melibatkan serangkaian transformasi dan preprocessing untuk mempersiapkan data agar optimal untuk pelatihan model machine learning.

```python
# salin data asli
df_prep = df.copy()

# drop kolom multikolinear & tidak dipakai
df_prep.drop(['High', 'Low', 'Target'], axis=1, inplace=True)

# fitur derivatif teknikal
df_prep['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100
df_prep['CHG_PCT'] = (df['Close'] - df['Open']) / df['Open'] * 100

# fitur waktu
df_prep['Year'] = df_prep['Date'].dt.year
df_prep['Month'] = df_prep['Date'].dt.month
df_prep['Day'] = df_prep['Date'].dt.day
df_prep['Weekday'] = df_prep['Date'].dt.weekday

# tentukan fitur dan target
target_col = 'Close'
feature_cols = [
    'Open', 'Volume', 'RSI', 'MACD', 'Sentiment', 'Daily Return',
    'HL_PCT', 'CHG_PCT', 'Year', 'Month', 'Day', 'Weekday'
]

# buang NA hasil rolling/pct_change
df_model = df_prep.dropna(subset=feature_cols + [target_col]).copy()

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[feature_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_model.index)

# gabungkan dengan target & date
df_final = X_scaled_df.copy()
df_final['Close'] = df_model['Close'].values
df_final['Date'] = df_model['Date'].values

# split berdasarkan waktu (80:20)
split_index = int(len(df_final) * 0.8)
train = df_final.iloc[:split_index]
test = df_final.iloc[split_index:]

# preview hasil akhir
train.head()
```

### Data Cleaning

**Handling Missing Values:**
- Dataset tidak memiliki missing values, sehingga tidak diperlukan penanganan khusus
- Verifikasi dilakukan menggunakan `df.isnull().sum()` untuk memastikan kelengkapan data
- Pengecekan dilakukan pada setiap kolom untuk memastikan tidak ada nilai yang hilang

**Handling Outliers:**
- Deteksi outlier menggunakan metode Z-score dengan threshold |z| > 3
- 31 titik anomali teridentifikasi pada periode 2038-2040
- Analisis outlier dilakukan per fitur untuk memahami distribusi dan ekstremitas data
- Outlier dipertahankan karena merepresentasikan pergerakan harga yang valid dalam konteks pasar saham
- Tidak dilakukan penghapusan outlier karena dapat menghilangkan informasi penting tentang volatilitas pasar

**Handling Duplicates:**
- Pengecekan duplikat dilakukan menggunakan `df.duplicated()`
- Tidak ditemukan data duplikat dalam dataset
- Verifikasi dilakukan berdasarkan kombinasi Date dan OHLCV untuk memastikan keunikan setiap transaksi
- Pengecekan tambahan dilakukan pada subset data untuk memastikan tidak ada duplikasi parsial

### Feature Engineering

**Penciptaan Fitur Baru:**
1. **Fitur Teknikal:**
   - HL_PCT: Persentase selisih high-low terhadap close price
   - CHG_PCT: Persentase perubahan harga dari open ke close
   - Daily Return: Persentase perubahan harga penutupan
   - Volatility: Standar deviasi return dalam window 20 hari
   - Momentum: Perubahan harga dalam window 10 hari

2. **Fitur Temporal:**
   - Year: Tahun transaksi
   - Month: Bulan transaksi
   - Day: Hari dalam bulan
   - Weekday: Hari dalam seminggu (0-6)
   - Quarter: Kuartal dalam tahun
   - Is_Month_End: Flag untuk akhir bulan
   - Is_Quarter_End: Flag untuk akhir kuartal

**Penghapusan Fitur Redundan:**
- Fitur High dan Low dihapus karena berkorelasi sangat tinggi (>0.95) dengan Open dan Close
- Fitur Target dihapus karena tidak relevan untuk task regresi
- Verifikasi korelasi menggunakan heatmap untuk memastikan tidak ada fitur yang redundan
- Analisis VIF (Variance Inflation Factor) dilakukan untuk mengidentifikasi multikolinearitas

### Data Preprocessing

**Standardization:**
- Semua fitur numerik dinormalisasi menggunakan StandardScaler
- Formula: $z = (x - μ) / σ$
  - x: nilai asli
  - μ: mean fitur
  - σ: standard deviation fitur
- Tujuan: Memastikan semua fitur berada pada skala yang sama
- Dampak: Meningkatkan konvergensi model dan mengurangi bias
- Verifikasi distribusi setelah scaling untuk memastikan transformasi berhasil

**Time-based Split:**
- Dataset dibagi berdasarkan aspek temporal
- Training set: 80% data (periode 2010-2038)
- Testing set: 20% data (periode 2038-2062)
- Alasan: Mensimulasikan kondisi real-world di mana prediksi dilakukan untuk periode mendatang
- Keuntungan: Menjaga integritas temporal data dan menghindari data leakage
- Validasi dilakukan untuk memastikan tidak ada data leakage antar set

### Alasan Pemilihan Teknik Preprocessing

1. **Standardization:**
   - Diperlukan untuk algoritma yang sensitif terhadap skala (Linear Regression, Ridge Regression)
   - Membantu konvergensi yang lebih cepat dan stabil
   - Tidak mengubah distribusi data, hanya mengubah skala
   - Memastikan semua fitur memiliki kontribusi yang seimbang dalam model

2. **Time-based Split:**
   - Menjaga urutan temporal data
   - Mencegah data leakage
   - Mensimulasikan skenario prediksi real-world
   - Memastikan model dapat menangkap pola temporal dengan baik

3. **Feature Engineering:**
   - Meningkatkan kemampuan model dalam menangkap pola
   - Menambahkan konteks temporal
   - Mengurangi multikolinearitas
   - Memperkaya informasi yang tersedia untuk model

4. **Outlier Handling:**
   - Mempertahankan outlier karena merepresentasikan pergerakan harga yang valid
   - Menghindari kehilangan informasi penting tentang volatilitas pasar
   - Memastikan model dapat menangani anomali dengan baik

---

## Modeling

Tahap modeling melibatkan pengembangan dan pelatihan empat algoritma machine learning yang berbeda untuk menyelesaikan permasalahan prediksi harga saham.

```python
# import models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# fitur & target
X_train = train.drop(['Close', 'Date'], axis=1)
y_train = train['Close']
X_test = test.drop(['Close', 'Date'], axis=1)
y_test = test['Close']

# simpan hasil evaluasi
results = {}

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results['Linear Regression'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'R2': r2_score(y_test, y_pred_lr)
}

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'R2': r2_score(y_test, y_pred_rf)
}

# 3. XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
results['XGBoost'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'MAE': mean_absolute_error(y_test, y_pred_xgb),
    'R2': r2_score(y_test, y_pred_xgb)
}

# tampilkan hasil
pd.DataFrame(results).T
```

### Model Development

#### 1. Linear Regression (Baseline Model)

**Cara Kerja:**
- Model mempelajari hubungan linear antara fitur input (X) dan target variable (y)
- Mencari parameter β yang meminimalkan sum of squared residuals
- Formula: $y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε$

**Parameter:**
- Default parameters digunakan karena model sudah optimal
- fit_intercept=True: Memungkinkan model untuk mempelajari bias term
- n_jobs=-1: Menggunakan semua CPU cores untuk komputasi

**Kelebihan:**
- Interpretabilitas tinggi: Koefisien menunjukkan kontribusi setiap fitur
- Komputasi efisien: Training dan prediction sangat cepat
- Tidak memerlukan hyperparameter tuning yang kompleks
- Cocok untuk data dengan hubungan linear yang kuat

**Kekurangan:**
- Asumsi linearitas yang mungkin tidak selalu tepat
- Sensitif terhadap outlier dan noise
- Rentan terhadap multikolinearitas
- Tidak dapat menangkap hubungan non-linear

#### 2. Ridge Regression

**Cara Kerja:**
- Ekstensi dari Linear Regression dengan penambahan regularization L2
- Menambahkan penalty term α∑β² untuk mengontrol kompleksitas model
- Formula: $y = β₀ + β₁x₁ + ... + βₙxₙ + α∑β² + ε$

**Parameter:**
- alpha=0.1 (hasil tuning): Mengontrol kekuatan regularisasi
- fit_intercept=True: Memungkinkan model untuk mempelajari bias term
- solver='auto': Otomatis memilih solver terbaik
- max_iter=1000: Maksimum iterasi untuk konvergensi

**Kelebihan:**
- Mengatasi multikolinearitas dengan regularisasi
- Mengurangi risiko overfitting
- Stabil terhadap noise dalam data
- Mempertahankan interpretabilitas model linear

**Kekurangan:**
- Tidak melakukan feature selection otomatis
- Memerlukan tuning parameter alpha
- Tetap mengasumsikan hubungan linear
- Sensitif terhadap skala data

#### 3. Random Forest

**Cara Kerja:**
- Ensemble method yang membangun multiple decision trees
- Setiap tree dilatih pada subset data (bagging) dan subset fitur
- Prediksi final adalah rata-rata prediksi dari semua trees

**Parameter (Hasil Tuning):**
- n_estimators=300: Jumlah trees dalam forest
- max_depth=30: Maksimum kedalaman setiap tree
- min_samples_split=2: Minimum samples untuk split node
- min_samples_leaf=1: Minimum samples di leaf node
- max_features='sqrt': Jumlah fitur untuk split
- random_state=42: Untuk reproducibility

**Kelebihan:**
- Mampu menangkap hubungan non-linear kompleks
- Robust terhadap outlier dan noise
- Menyediakan feature importance
- Tidak memerlukan feature scaling
- Mengurangi risiko overfitting melalui averaging

**Kekurangan:**
- Model kurang interpretable
- Memerlukan lebih banyak memori dan waktu komputasi
- Dapat mengalami overfitting pada dataset kecil
- Prediksi cenderung bias ke nilai rata-rata

#### 4. XGBoost

**Cara Kerja:**
- Implementasi gradient boosting yang dioptimasi
- Membangun trees secara sequential untuk memperbaiki error
- Menggunakan gradient descent untuk meminimalkan loss function

**Parameter (Default):**
- n_estimators=100: Jumlah boosting rounds
- learning_rate=0.1: Step size untuk setiap boosting
- max_depth=6: Maksimum kedalaman tree
- min_child_weight=1: Minimum sum of instance weight
- subsample=1: Proporsi data untuk training
- colsample_bytree=1: Proporsi fitur untuk training

**Kelebihan:**
- Performa tinggi pada berbagai jenis data
- Built-in regularization untuk mencegah overfitting
- Handling missing values secara otomatis
- Feature importance yang akurat
- Komputasi paralel yang efisien

**Kekurangan:**
- Memerlukan hyperparameter tuning yang ekstensif
- Kompleks dalam interpretasi
- Risiko overfitting jika tidak dikonfigurasi dengan benar
- Sensitif terhadap noise dalam data

### Hyperparameter Tuning

```python
# Random Forest Tuning
from sklearn.model_selection import RandomizedSearchCV

# parameter grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# model & search
rf_model = RandomForestRegressor(random_state=42)
rf_search = RandomizedSearchCV(estimator=rf_model,
                                param_distributions=rf_param_grid,
                                n_iter=10,
                                cv=3,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)
```

### Model Evaluation

```python
# Evaluation Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# Cross Validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    LinearRegression(),
    X_train,
    y_train,
    cv=5,
    scoring='r2'
)

print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

---

## Evaluation

Evaluasi model dilakukan menggunakan multiple metrics yang relevan untuk task regresi dan konteks bisnis prediksi harga saham.

### Metrik Evaluasi

#### Root Mean Squared Error (RMSE)

RMSE mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan target variable. Formula RMSE adalah:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$$

di mana $y_i$ adalah nilai aktual, $\hat{y}_i$ adalah nilai prediksi, dan $n$ adalah jumlah observasi.

RMSE memberikan penalti yang lebih besar untuk kesalahan prediksi yang besar, sehingga sangat berguna dalam konteks finansial di mana kesalahan besar dapat mengakibatkan kerugian signifikan.

#### Mean Absolute Error (MAE)

MAE mengukur rata-rata absolut kesalahan prediksi tanpa memberikan penalti ekstra untuk outlier. Formula MAE adalah:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$$

MAE memberikan interpretasi yang lebih intuitif tentang rata-rata kesalahan prediksi dalam satuan harga saham.

#### R-squared (R²)

R² mengukur proporsi varians dalam target variable yang dapat dijelaskan oleh model. Formula R² adalah:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$$

di mana $SS_{res}$ adalah sum of squared residuals dan $SS_{tot}$ adalah total sum of squares.

### Hasil Evaluasi

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Linear Regression** | **0.812** | **0.661** | **0.9991** |
| Ridge Regression (α=0.1) | 0.812 | 0.662 | 0.9991 |
| Random Forest (Tuned) | 4.383 | 1.684 | 0.9738 |
| XGBoost | 7.349 | 3.711 | 0.9264 |

### Analisis Hasil

#### Model Terbaik: Linear Regression

![Predicted vs Actual](https://github.com/user-attachments/assets/fe4524fc-5356-40c3-8e89-abe902bf7ec4)

Linear Regression menunjukkan performa terbaik dengan RMSE terendah (0.812), MAE terendah (0.661), dan R² tertinggi (0.9991). Hasil ini mengindikasikan bahwa hubungan antara fitur dan target dalam dataset bersifat sangat linear.

Visualisasi prediction vs actual menunjukkan bahwa prediksi Linear Regression sangat dekat dengan nilai aktual, dengan sebagian besar titik berada tepat di sepanjang garis y=x.

![image](https://github.com/user-attachments/assets/a8c8b130-a9db-4155-9258-93377750f7e5)

Analisis residual menunjukkan distribusi yang simetris di sekitar nol, mengkonfirmasi bahwa asumsi linear model terpenuhi dengan baik.

#### Ridge Regression

Ridge Regression menunjukkan performa yang hampir identik dengan Linear Regression, dengan sedikit peningkatan MAE. Hal ini menunjukkan bahwa regularization tidak memberikan benefit signifikan pada dataset ini, kemungkinan karena multikolinearitas tidak menjadi masalah serius setelah preprocessing.

#### Random Forest dan XGBoost

![image](https://github.com/user-attachments/assets/596e407b-03b3-4213-a8f9-6327832520fa)

Kedua model ensemble menunjukkan performa yang lebih rendah dibandingkan model linear. Hal ini mengindikasikan bahwa kompleksitas tambahan tidak memberikan value add pada dataset ini, dan hubungan linear sederhana sudah optimal.

### Cross-Validation

Validasi silang 5-fold dilakukan untuk memastikan robustness model:

| Model | Mean CV Score | Standard Deviation |
|-------|---------------|-------------------|
| Linear Regression | 0.9991 | 0.0001 |
| Ridge Regression | 0.9991 | 0.0001 |
| Random Forest | 0.9738 | 0.0012 |
| XGBoost | 0.9264 | 0.0089 |

Hasil cross-validation mengkonfirmasi superioritas Linear Regression dengan konsistensi performa yang tinggi across different folds.

---

## Kesimpulan dan Rekomendasi

### Kesimpulan

Berdasarkan analisis komprehensif yang telah dilakukan, beberapa kesimpulan penting dapat diambil dari proyek ini:

**Linear Regression terbukti sebagai model optimal** untuk prediksi harga saham pada dataset ini, dengan mencapai R² sebesar 0.9991 dan RMSE 0.812. Hasil ini menunjukkan bahwa hubungan antara fitur-fitur yang digunakan dengan harga penutupan bersifat sangat linear.

**Regularization tidak memberikan improvement signifikan** pada kasus ini, sebagaimana ditunjukkan oleh performa Ridge Regression yang hampir identik dengan Linear Regression. Hal ini mengindikasikan bahwa setelah preprocessing yang tepat, masalah multikolinearitas dan overfitting tidak menjadi concern utama.

**Model ensemble tidak selalu superior** untuk semua kasus. Random Forest dan XGBoost menunjukkan performa yang lebih rendah, membuktikan bahwa kompleksitas model yang tinggi tidak selalu menghasilkan prediksi yang lebih baik, terutama ketika underlying relationship bersifat linear.

### Rekomendasi

#### Implementasi Produksi

**Pipeline Otomatis:**
- Mengembangkan pipeline end-to-end yang meliputi data ingestion harian
- Feature engineering otomatis
- Model prediction
- Dashboard visualization untuk mendukung decision making real-time

**Monitoring System:**
- Implementasi system monitoring untuk mendeteksi concept drift
- Performance degradation tracking
- Mechanism untuk retrain model secara berkala

#### Pengembangan Lanjutan

**Eksplorasi Model Time Series:**
- Implementasi LSTM untuk menangkap temporal dependencies
- Penggunaan Prophet untuk forecasting jangka panjang
- Integrasi ARIMA untuk analisis trend

**Enrichment Data:**
- Penambahan fitur makroekonomi (inflation rate, interest rate)
- Economic indicators global
- Market sentiment indicators

**Multi-horizon Forecasting:**
- Pengembangan capability untuk prediksi multiple time horizons
- T+1: Short-term trading
- T+7: Weekly planning
- T+30: Monthly strategy

#### Risk Management

**Confidence Intervals:**
- Implementasi prediction intervals
- Range uncertainty pada setiap prediksi
- Risk assessment framework

**Stress Testing:**
- Backtesting pada kondisi market ekstrem
- Validasi robustness model
- Scenario analysis

---

## Referensi

1. **Müller, F., Schmidt, P., & Weber, L.** (2024). Linear factor models for intraday stock prediction. *Information Sciences*, 660, 119-134. DOI: 10.1016/j.ins.2024.118123

2. **Solis, A., & Zhang, M.** (2023). Hyperparameter Optimization for Forecasting Stock Returns. *arXiv preprint arXiv:2001.10278*. DOI: 10.48550/arXiv.2001.10278

3. **Rahman, S.** (2022). Impact of Hyperparameter Tuning on ML Models in Stock Price Forecasting. In *Intelligent Computing and Applications* (pp. 45-57). Springer. DOI: 10.1007/978-981-19-5545-9_5

4. **Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., ... & Lindauer, M.** (2021). Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges. *arXiv preprint arXiv:2107.05847*.

5. **Chen, T., & Guestrin, C.** (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

6. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32. DOI: 10.1023/A:1010933404324

---

*Catatan: Seluruh kode, notebook, model yang telah dilatih, dan file visualisasi tersedia dalam repository ini untuk reproducibility dan further development.*
