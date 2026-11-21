import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Baca dataset
df = pd.read_csv("dataset.csv")
X = df.iloc[:, :-1]  # Semua kolom fitur
y = df.iloc[:, -1]   # Kolom terakhir (label 0 atau 1)

# 2. Bagi data: 80% untuk belajar (train), 20% untuk ujian (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi performa model
y_pred = model.predict(X_test)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Detail Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=["Not Skipping", "Skipping"]))

# 5. Simpan model yang sudah pintar ke file
joblib.dump(model, "skipping_model.pkl")
print("Model berhasil disimpan sebagai skipping_model.pkl")