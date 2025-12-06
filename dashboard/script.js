const API_URL = "https://selecta-api.onrender.com";

function toggleMenu() {
    const m = document.getElementById("submenu");
    m.style.display = m.style.display === "block" ? "none" : "block";
}

function loadPage(page) {
    const c = document.getElementById("content");
    if (page === "intro") {
        c.innerHTML = "<h2>Introductions</h2><p>Sistem rekomendasi SMA berbasis KNN.</p>";
    }
    if (page === "model") {
        c.innerHTML = "<h2>Model Explanation</h2><p>Menggunakan K-Nearest Neighbors Euclidean Distance.</p>";
    }
}

async function loadCity(kota) {
    const c = document.getElementById("content");

    c.innerHTML = "<p>Loading...</p>";

    const mat = 90;
    const ipa = 88;
    const bindo = 87;
    const rapor = 89;

    const res = await fetch(`${API_URL}/predict?kota=${kota}&mat=${mat}&ipa=${ipa}&bindo=${bindo}&rapor=${rapor}`);
    const data = await res.json();

    c.innerHTML = `
        <h2>Hasil Prediksi - ${kota}</h2>
        <p><b>Sekolah Utama:</b> ${data.prediksi_utama}</p>
        <h3>Ranking:</h3>
        <pre>${JSON.stringify(data.ranking, null, 2)}</pre>
    `;
}
