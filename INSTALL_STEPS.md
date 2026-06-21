# Langkah Instalasi Ulang (Setelah Perbaikan v5)

Bacaan wajib sebelum `streamlit run app.py`. Urutannya penting.

## Riwayat singkat (kenapa instruksi versi sebelumnya berubah lagi)

Percobaan pertama menahan `transformers<5.0.0` GAGAL di praktiknya:
`sentence-transformers` (versi 5.2.1+) sudah melepas batasan itu dari
dependency-nya sendiri, jadi pip selalu menarik transformers v5 lagi
walau di-pin di `requirements.txt`. Keputusan final: **terima transformers
v5**. Sebagai gantinya, `qa_model.py` ditulis ulang total agar tidak
bergantung pada pipeline task `"question-answering"` yang dihapus di v5
— sekarang model di-load manual dan span jawaban dihitung manual dari
logits, bukan lewat `transformers.pipeline()`.

## Status verifikasi (jujur, baca sebelum percaya semua "pasti jalan")

| Perubahan | Status |
|---|---|
| `qa_model.py` ditulis ulang (manual forward pass, tanpa `pipeline()`) | **Logika terverifikasi** dengan model BERT-QA dummy (random weight, struktur sama) + transformers v5.12.1 asli — terbukti tidak crash dan span decoding benar. **Belum dites dengan model asli `Rifky/Indobert-QA`** karena sandbox saya tidak punya akses ke huggingface.co untuk download model itu. |
| `ner_model.py` (task `"ner"`) | **Terverifikasi langsung** lewat introspeksi `PIPELINE_REGISTRY` — task ini TIDAK dihapus di transformers v5, jadi tidak perlu diubah. |
| Fix WHO filter (span NER rusak: potongan username, nama kepotong) | **Hanya disimulasikan** dengan data palsu, BELUM dites dengan model NER asli (sebelumnya dan masih sekarang) — verifikasi ini masih jadi tugasmu. |
| Fix HOW overmatch ("dalam akun" nyangkut ke kutipan) | **Sudah dieksekusi & dibuktikan jalan** sebelumnya, tidak berubah di revisi ini. |
| `torch>=2.6.0` (fix CVE-2025-32434 saat load model `.bin` non-safetensors) | Tetap berlaku, terverifikasi via dokumentasi resmi, belum dites di mesinmu. |
| GPU RTX 2050 + CUDA 12.4 | Benar secara dokumentasi NVIDIA, belum dites di mesinmu. |
| VRAM 4GB cukup utk semua model bersamaan | **BELUM DICEK** sama sekali. |

Kalau ada langkah yang error, **kirim output mentah lengkap**, bukan ringkasan — supaya diagnosisnya tepat, bukan tebakan lagi.

---

## Langkah 1 — Bersihkan instalasi lama (hindari sisa versi konflik)

```powershell
cd D:\Documents\Downloads\chromonews_v2\chromonews
.\venv\Scripts\activate
pip uninstall torch torchvision torchaudio transformers sentence-transformers -y
```

## Langkah 2 — Install torch dengan CUDA 12.4 (RTX 2050, driver 546.18 mendukung ini)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Kalau mau coba CPU dulu (lebih sederhana untuk debugging awal):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Langkah 3 — Verifikasi torch (WAJIB sebelum lanjut)

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA tersedia:', torch.cuda.is_available())"
```

Harapan (kalau pakai GPU): `torch: 2.6.0+cu124` (boleh versi lebih baru) dan `CUDA tersedia: True`.

## Langkah 4 — Install sisa dependency (termasuk transformers v5 — ini SUDAH DISENGAJA, jangan downgrade manual)

```powershell
pip install -r requirements.txt
```

## Langkah 5 — Verifikasi transformers & task registry

```powershell
python -c "import transformers; print(transformers.__version__)"
python -c "from transformers.pipelines import PIPELINE_REGISTRY; print('ner' in PIPELINE_REGISTRY.get_supported_tasks())"
```

Harapan: versi `5.x`, dan baris kedua print `True`.

## Langkah 6 — Verifikasi qa_model.py bisa load model ASLI (ini test yang BELUM saya bisa lakukan sendiri)

```powershell
python -c "from extraction.qa_model import load_qa_pipeline, answer_question; pipe = load_qa_pipeline(); print('Load OK:', type(pipe)); r = answer_question(pipe, 'why', 'Pras mendatangi kantor KPK untuk memberikan keterangan karena ingin membantu penyidikan korupsi.'); print('Jawaban:', r)"
```

**Ini langkah paling penting untuk kamu jalankan dan laporkan hasilnya** — kirim output PERSIS apa yang tercetak (termasuk traceback kalau error). Saya sudah verifikasi logika kodenya secara struktural, tapi belum pernah lihat hasil nyata dari model `Rifky/Indobert-QA` yang sesungguhnya.

## Langkah 7 — Jalankan app

```powershell
streamlit run app.py
```

## Langkah 8 — Test artikel yang sama, kumpulkan bukti

Masukkan artikel KPK/Pulogebang yang sama, catat:
- Apakah notifikasi "Model hybrid tidak aktif" masih muncul
- Hasil WHO, WHY, HOW yang baru
- Traceback LENGKAP kalau masih ada error (jangan dipotong)

---

## Kalau kena `CUDA out of memory` (VRAM 4GB terbatas, belum tentu terjadi)

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

Atau paksa semua model jalan di CPU:
```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
```
lalu jalankan `streamlit run app.py` lagi.
