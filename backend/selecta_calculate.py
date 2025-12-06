import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ================================
# DATA SMA PER DAERAH
# ================================
data_daerah = {
    "yogyakarta": {
        'SMAN 1 Yogyakarta': [96, 89, 91, 92],
        'SMAN 2 Yogyakarta': [91, 78, 86, 88],
        'SMAN 3 Yogyakarta': [97, 92, 91, 93],
        'SMAN 4 Yogyakarta': [75, 66, 81, 77],
        'SMAN 5 Yogyakarta': [94, 84, 89, 91],
        'SMAN 6 Yogyakarta': [93, 86, 89, 89],
        'SMAN 7 Yogyakarta': [92, 80, 89, 88],
        'SMAN 8 Yogyakarta': [96, 88, 92, 92],
        'SMAN 9 Yogyakarta': [91, 78, 87, 89],
        'SMAN 10 Yogyakarta': [83, 71, 84, 88],
        'SMAN 11 Yogyakarta': [87, 77, 87, 88]
    },

    "bantul": {
        'SMAN 1 BANTUL': [92, 82, 89, 90],
        'SMAN 2 BANTUL': [81, 75, 84, 87],
        'SMAN 3 BANTUL': [68, 64, 74, 83],
        'SMAN 1 SANDEN': [79, 70, 83, 87],
        'SMAN 1 BANGUNTAPAN': [86, 74, 84, 88],
        'SMAN 1 PAJANGAN': [64, 61, 75, 83],
        'SMAN 1 KASIHAN': [82, 72, 84, 87],
        'SMAN 1 DLINGO': [63, 59, 70, 76],
        'SMAN 1 JETIS': [75, 67, 78, 84],
        'SMAN 1 SEWON': [79, 73, 83, 86],
        'SMAN 2 BANGUNTAPAN': [79, 70, 82, 87],
        'SMAN 1 IMOGIRI': [58, 57, 70, 77],
        'SMAN 1 SEDAYU': [76, 70, 82, 85],
        'SMAN 1 PUNDONG': [76, 67, 79, 86],
        'SMAN 1 PLERET': [58, 57, 65, 80],
        'SMAN 1 PIYUNGAN': [69, 64, 76, 85],
        'SMAN 1 KRETEK': [56, 57, 64, 80],
        'SMAN 1 SRANDAKAN': [61, 57, 66, 83],
        'SMAN 1 BAMBANGLIPURO': [70, 62, 74, 82]
    },

    "sleman": {
        'SMAN 1 GODEAN': [97, 87, 90, 90],
        'SMAN 1 KALASAN': [90, 83, 90, 88],
        'SMAN 1 SLEMAN': [97, 92, 91, 93],
        'SMAN 1 DEPOK': [89, 76, 88, 88],
        'SMAN 1 CANGKRINGAN': [66, 65, 77, 83],
        'SMAN 1 NGAGLIK': [65, 64, 77, 85],
        'SMAN 1 TEMPEL': [64, 64, 76, 85],
        'SMAN 1 PAKEM': [78, 73, 84, 86],
        'SMAN 1 GAMPING': [79, 69, 79, 85],
        'SMAN 1 SAYEGAN': [78, 70, 80, 85],
        'SMAN 1 TURI': [64, 63, 73, 84],
        'SMAN 1 PRAMBANAN': [77, 72, 74, 85],
        'SMAN 1 NGEMPLAK': [69, 67, 78, 84],
        'SMAN 1 MLATI': [80, 70, 82, 86],
        'SMAN 1 MINGGIR SLEMAN': [72, 67, 80, 85],
        'SMAN 2 NGAGLIK': [85, 74, 83, 87],
        'SMAN 2 SLEMAN': [73, 69, 82, 86]
    },

    "gunungkidul": {
        'SMAN 1 WONOSARI': [92, 82, 89, 91],
        'SMAN 2 WONOSARI': [69, 56, 80, 85],
        'SMAN 1 SEMIN': [67, 64, 73, 87],
        'SMAN 1 KARANGMOJO': [66, 65, 79, 86],
        'SMAN 1 PLAYEN': [60, 59, 66, 80],
        'SMAN 1 PATUK': [61, 61, 67, 82],
        'SMAN 1 SEMANU': [59, 58, 63, 83],
        'SMAN 1 RONGKOP': [56, 54, 60, 81],
        'SMAN 1 PANGGANG': [66, 66, 71, 81],
        'SMAN 2 PLAYEN': [64, 63, 78, 86],
        'SMAN 1 TANJUNGSARI': [56, 54, 60, 82]
    },

    "kulonprogo": {
        'SMAN 1 WATES': [94, 84, 89, 92],
        'SMAN 2 WATES': [80, 73, 80, 86],
        'SMAN 1 SENTOLO': [73, 69, 78, 85],
        'SMAN 1 KALIBAWANG': [72, 70, 80, 86],
        'SMAN 1 PENGASIH': [74, 70, 81, 85],
        'SMAN 1 TEMON': [66, 68, 81, 84],
        'SMAN 1 LENDAH': [76, 67, 80, 84],
        'SMAN 1 GALUR': [61, 60, 67, 81],
        'SMAN 1 GIRIMULYO': [59, 59, 64, 81],
        'SMAN 1 KOKAP': [57, 57, 63, 81],
        'SMAN 1 SAMIGALUH': [59, 58, 65, 82]
    }
}

# ================================
# FUNGSI MEMBUAT MODEL KNN
# ================================
def buat_model_knn(daerah):
    sma_data = data_daerah[daerah]
    X = np.array(list(sma_data.values()))
    y = list(sma_data.keys())

    model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    model.fit(X, y)
    return model

# ================================
# FUNGSI PREDIKSI
# ================================
def prediksi_sma(daerah, aspd_mat, aspd_ipa, aspd_bindo, rapor):
    model = buat_model_knn(daerah)
    siswa = np.array([[aspd_mat, aspd_ipa, aspd_bindo, rapor]])

    pred = model.predict(siswa)[0]
    probs = model.predict_proba(siswa)[0]
    kelas = model.classes_

    # Urutkan probabilitas
    sorted_idx = np.argsort(probs)[::-1]
    hasil = [
        {"sekolah": kelas[i], "probabilitas": float(probs[i])}
        for i in sorted_idx
    ]

    return {
        "daerah": daerah,
        "prediksi_utama": pred,
        "ranking": hasil
    }

# ================================
# CONTOH PAKAI DI TERMINAL
# ================================
if __name__ == "__main__":
    daerah = input("Pilih daerah (yogyakarta/bantul/sleman/gunungkidul/kulonprogo): ").lower()
    aspd_mat = float(input("ASPD Mat: "))
    aspd_ipa = float(input("ASPD IPA: "))
    aspd_bindo = float(input("ASPD B.Indo: "))
    rapor = float(input("Rapor: "))

    hasil = prediksi_sma(daerah, aspd_mat, aspd_ipa, aspd_bindo, rapor)
    print(hasil)
