import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import json

# Modul 1: Prapemrosesan
class PrapemrosesanTandaTangan:
    def __init__(self, ukuran_gambar=(128, 128)):
        self.ukuran_gambar = ukuran_gambar

    def praproses(self, path_gambar):
        try:
            gambar = cv2.imread(path_gambar, cv2.IMREAD_GRAYSCALE)
            if gambar is None:
                raise ValueError(f"Gambar pada path {path_gambar} tidak dapat dimuat.")

            gambar = cv2.resize(gambar, self.ukuran_gambar)
            return gambar.flatten()
        except Exception as e:
            raise ValueError(f"Terjadi kesalahan saat memproses gambar pada {path_gambar}: {e}")

# Modul 2: Pemuatan Dataset
class PemuatDataset:
    def __init__(self, direktori_data, prapemrosesan):
        self.direktori_data = direktori_data
        self.prapemrosesan = prapemrosesan

    def muat_data(self):
        data, label = [], []
        for nama_kelas in os.listdir(self.direktori_data):
            path_kelas = os.path.join(self.direktori_data, nama_kelas)
            if os.path.isdir(path_kelas):
                for nama_file in os.listdir(path_kelas):
                    if nama_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path_file = os.path.join(path_kelas, nama_file)
                        try:
                            gambar_terproses = self.prapemrosesan.praproses(path_file)
                            data.append(gambar_terproses)
                            label.append(nama_kelas)
                        except ValueError as e:
                            print(e)
        return np.array(data), np.array(label)

# Modul 3: Model Pengenal Tanda Tangan
class ModelPengenalTandaTangan:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)

    def latih(self, X_latih, y_latih):
        self.model.fit(X_latih, y_latih)

    def validasi(self, X_validasi, y_validasi):
        y_pred = self.model.predict(X_validasi)
        akurasi = accuracy_score(y_validasi, y_pred)
        print("Akurasi Validasi:", akurasi)
        print(classification_report(y_validasi, y_pred))
        return akurasi

    def prediksi(self, X):
        return self.model.predict(X)

    def prediksi_dengan_probabilitas(self, X, label_encoder):
        probabilitas = self.model.predict_proba(X)
        prediksi = self.model.predict(X)
        label_terdekripsi = label_encoder.inverse_transform(prediksi)
        hasil = []
        for i, prob in enumerate(probabilitas):
            hasil_prediksi = {
                "label_prediksi": label_terdekripsi[i],
                "kepercayaan": round(np.max(prob) * 100, 2)  # Kepercayaan dalam persen
            }
            hasil.append(hasil_prediksi)
        return hasil

# Modul 4: Visualisasi
class Visualisasi:
    @staticmethod
    def plot_akurasi(akurasi, filename='akurasi_plot.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(akurasi, marker='o', label='Akurasi Validasi')
        plt.title('Kinerja Model')
        plt.xlabel('Epoch')
        plt.ylabel('Akurasi')
        plt.legend()
        plt.grid()
        plt.savefig(filename)  # Save plot to file
        plt.show()

# Program Utama
if __name__ == "__main__":
    # Konfigurasi
    DIREKTORI_DATA = r"D:\UAS KB\TandaTangan\DATA TTD" 
    UKURAN_GAMBAR = (128, 128)

    # Prapemrosesan dan Pemuatan Data
    prapemrosesan = PrapemrosesanTandaTangan(ukuran_gambar=UKURAN_GAMBAR)
    pemuat = PemuatDataset(DIREKTORI_DATA, prapemrosesan)

    print("Memuat data...")
    data, label = pemuat.muat_data()

    # Encode label
    encoder = LabelEncoder()
    label_terenkripsi = encoder.fit_transform(label)

    # Split data
    X_latih, X_validasi, y_latih, y_validasi = train_test_split(data, label_terenkripsi, test_size=0.2, random_state=42)

    # Latih dan Validasi Model
    print("Melatih model...")
    model = ModelPengenalTandaTangan()
    model.latih(X_latih, y_latih)

    print("Memvalidasi model...")
    akurasi = model.validasi(X_validasi, y_validasi)

    # Prediksi dan Tampilkan Hasil
    print("Menguji pada data validasi...")
    hasil = model.prediksi_dengan_probabilitas(X_validasi, encoder)
    for i, hasil_prediksi in enumerate(hasil):
        print(f"Sampel {i+1}: Diprediksi sebagai '{hasil_prediksi['label_prediksi']}' dengan kepercayaan {hasil_prediksi['kepercayaan']}%")

    # Simpan hasil prediksi ke dalam file JSON
    with open('hasil_prediksi.json', 'w') as file:
        json.dump(hasil, file)

    # Visualisasi Hasil
    Visualisasi.plot_akurasi([akurasi])

    # Load and display saved plot
    from PIL import Image
    image = Image.open('akurasi_plot.png')
    image.show()
