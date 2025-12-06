from flask import Flask, request, jsonify
from selecta_calculate import prediksi_sma

app = Flask(__name__)

# daftar valid daerah
valid_daerah = ["yogyakarta", "bantul", "sleman", "gunungkidul", "kulonprogo"]

@app.route("/")
def home():
    return jsonify({
        "service": "SELECTA API",
        "message": "API sudah berjalan",
        "endpoint": "/predict?kota=sleman&mat=90&ipa=88&bindo=87&rapor=89"
    })

@app.route("/predict", methods=["GET"])
def predict():
    daerah = request.args.get("kota")
    aspd_mat = request.args.get("mat")
    aspd_ipa = request.args.get("ipa")
    aspd_bindo = request.args.get("bindo")
    rapor = request.args.get("rapor")

    # Validasi
    if daerah is None or daerah.lower() not in valid_daerah:
        return jsonify({"error": "Parameter 'kota' wajib & harus valid"}), 400
    
    try:
        aspd_mat = float(aspd_mat)
        aspd_ipa = float(aspd_ipa)
        aspd_bindo = float(aspd_bindo)
        rapor = float(rapor)
    except:
        return jsonify({"error": "Parameter mat, ipa, bindo, rapor harus berupa angka"}), 400

    # Jalankan model
    hasil = prediksi_sma(
        daerah.lower(),
        aspd_mat, aspd_ipa, aspd_bindo, rapor
    )

    return jsonify(hasil)

# endpoint khusus per daerah
@app.route("/predict/<kota>", methods=["GET"])
def predict_specific(kota):
    if kota.lower() not in valid_daerah:
        return jsonify({"error": "Kota tidak ditemukan"}), 404

    # ambil nilai dari query string
    try:
        aspd_mat = float(request.args.get("mat"))
        aspd_ipa = float(request.args.get("ipa"))
        aspd_bindo = float(request.args.get("bindo"))
        rapor = float(request.args.get("rapor"))
    except:
        return jsonify({"error": "Parameter nilai tidak lengkap"}), 400

    hasil = prediksi_sma(kota.lower(), aspd_mat, aspd_ipa, aspd_bindo, rapor)
    return jsonify(hasil)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
