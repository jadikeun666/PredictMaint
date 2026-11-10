"""
config/classification_provenance.py

Penjaga provenance alert klasifikasi (Sesi E). Satu-satunya tempat nilai ini didefinisikan.

Latar belakang (Addendum ml-pipeline.md/scheduling.md 2026-11-09 dan Sesi E): model klasifikasi
dilatih hanya pada CWRU (SKF 6205-2RS JEM, drive end). Pada data PU keyakinan tidak berfungsi
sebagai penjaga, dan evaluasi kandidat penjaga statistik (G1-G4, Sesi E Giliran 1) tidak
menurunkan alarm palsu jendela sehat PU. Kebijakan yang tersisa deterministik: prediksi fault
berkeyakinan tinggi hanya dipercaya untuk bearing yang model-nya tervalidasi terhadap versi
model yang menghasilkannya.

- Flag mati (False) = perilaku persis seperti sebelum Sesi E.
- Kunci ALLOWED_BEARING_MODELS = model_version classifier; nilai = bearing_model (cocok PERSIS,
  case-sensitive, seperti lookup kompatibilitas part di inventory.md) yang tervalidasi.
- model_version yang TIDAK ada di sini diperlakukan belum tervalidasi (fail-closed) saat flag
  menyala: mendaftarkan/men-deploy classifier baru memerlukan penambahan entri di sini.
"""
CLASSIFICATION_PROVENANCE_GUARD_ENABLED = False

ALLOWED_BEARING_MODELS = {
    "rf-baseline-cwru-v1.1": ("SKF 6205-2RS JEM (drive end)",),
}
