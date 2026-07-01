# PERUBAHAN_NER.md — Ringkasan Perbaikan WHO & WHERE

## Bug kritis yang ditemukan & diperbaiki

### 1. `inject_ner_pipeline()` SELALU gagal diam-diam (BUG UTAMA)
File: `extraction/rule_based_5w1h.py`

Sebelumnya:
```python
from who_extractor import inject_ner_pipeline as inject_who   # SALAH
from where_extractor import inject_ner_pipeline as inject_where  # SALAH
```
Import ini selalu `ModuleNotFoundError` karena folder `extraction/` tidak
pernah ada di `sys.path`. Karena `app.py` memanggil fungsi ini di dalam
`try/except Exception: pass`, error-nya tertelan total tanpa pesan apa pun.

**Akibatnya: model NER yang berhasil di-download dan di-load TIDAK PERNAH
benar-benar dipakai.** `_NER_PIPELINE` di `who_extractor.py` dan
`where_extractor.py` selalu `None`, sehingga `extract_who()`/`extract_where()`
selalu jatuh ke fallback regex, sepanjang waktu. Ini kemungkinan **akar
masalah terbesar** di balik banyaknya hasil WHO/WHERE yang anomali/miss --
bukan murni soal kualitas model atau limit karakter.

Sudah diperbaiki jadi:
```python
from extraction.who_extractor import inject_ner_pipeline as inject_who
from extraction.where_extractor import inject_ner_pipeline as inject_where
```

### 2. Limit 1500 karakter pertama
File lama: `extraction/who_extractor.py`, `extraction/where_extractor.py`, `extraction/ner_model.py`

76% artikel di `cleaned_news_sample.csv` panjangnya >1500 karakter (median
2099 char) -- entitas yang disebut di paragraf belakang tidak pernah
dilihat NER. Diganti dengan **chunking** (window ~1500 char + overlap 150
char) lewat `run_ner_chunked()` di `extraction/ner_model.py`, sehingga
seluruh teks ikut dianalisis.

### 3. Cap keras "ambil N pertama lalu stop"
Diganti jadi: kumpulkan SEMUA kandidat dari seluruh teks dulu, beri skor
(frekuensi kemunculan + posisi awal + confidence model), baru ambil
top-K. Nama/lokasi yang cuma disebut sekali secara kebetulan otomatis
kalah skor dari yang benar-benar relevan.

### 4. Model NER diganti
Dari `cahya/bert-base-indonesian-NER` (3 label kasar: PER/ORG/LOC) ke
`cahya/xlm-roberta-large-indonesian-NER` (18 label fine-grained, termasuk
GPE/FAC/LOC terpisah -- jauh lebih presisi untuk WHERE, tidak perlu lagi
hardcode ratusan nama gedung/kantor/jalan).

### 5. Bug tokenizer: entity terpecah jadi 2 fragmen
Ditemukan saat testing: `"Istana Negara"` bisa kepecah jadi `"Is"` +
`"tana Negara"` (artefak SentencePiece). Ditambahkan
`_merge_adjacent_entities()` di `extraction/ner_model.py` yang menggabungkan
fragmen yang bersambungan persis (tanpa gap karakter) SEBELUM divalidasi.

### 6. `dateline_location` sekarang benar-benar dipakai
Dateline media (mis. `"TEMPO.CO, Jakarta -"`) sudah di-parse sejak versi
lama tapi tidak pernah diteruskan ke `extract_where()`. Sekarang diteruskan
dan diberi bobot skor tinggi karena ini sinyal WHERE paling reliable.

### 7. Whitelist nama hardcode (~40 tokoh) dihapus
Diganti heuristik: nama 1 kata diterima kalau muncul ≥2 kali di artikel
yang sama -- lebih scalable, tidak terbatas tokoh nasional saja.

---

## File yang berubah
```
extraction/ner_model.py        <- diganti total (model baru + chunking + merge)
extraction/who_extractor.py    <- diganti total (ranking, bukan first-N)
extraction/where_extractor.py  <- diganti total (GPE/FAC/LOC + dateline)
extraction/rule_based_5w1h.py  <- 2 perbaikan kecil (import + dateline param)
extraction/evaluate_who_where.py  <- BARU, untuk evaluasi P/R/F1
requirements.txt                <- tambah sentencepiece
```

File lain (`how_extractor.py`, `why_extractor.py`, `gazetteer.py`,
`patterns.py`, `ner_helper.py`, `app.py`, dll) **tidak diubah**.
(`ner_helper.py` tampaknya modul lama yang sudah tidak dipakai di alur
manapun -- dibiarkan ada, tidak mengganggu.)

---

## LANGKAH SEBELUM RUN

1. **Aktifkan venv yang sudah ada** (sama seperti biasa):
   ```powershell
   venv\Scripts\Activate.ps1
   ```

2. **Install ulang requirements** (ada tambahan `sentencepiece`):
   ```powershell
   pip install -r requirements.txt
   ```

3. **Hapus cache model NER lama** supaya tidak ada konflik nama/file korup
   dari proses download sebelumnya (opsional tapi disarankan kalau sempat
   ada percobaan download yang gagal/terputus):
   ```powershell
   Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--cahya--bert-base-indonesian-NER" -ErrorAction SilentlyContinue
   ```
   (model baru `xlm-roberta-large-indonesian-NER` akan otomatis ke-download
   ulang ke cache terpisah saat pertama kali run -- ukurannya ~2.2GB,
   butuh waktu beberapa menit tergantung koneksi, seperti yang sudah kamu
   alami di test sebelumnya)

4. **(Opsional tapi direkomendasikan) Aktifkan Developer Mode di Windows**
   supaya cache HuggingFace pakai symlink (lebih hemat disk), berdasarkan
   warning yang muncul di testmu kemarin:
   https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
   Kalau malas, biarkan saja -- cuma soal disk space, tidak menghalangi
   fungsi.

5. **Jalankan dulu test cepat (BUKAN full app)** untuk memastikan inject
   sukses sebelum buka Streamlit:
   ```powershell
   python -c "from extraction.rule_based_5w1h import extract_5w1h, inject_ner_pipeline; from extraction.ner_model import load_ner_pipeline; pipe = load_ner_pipeline(); inject_ner_pipeline(pipe); import extraction.who_extractor as w; print('NER aktif:', w._NER_PIPELINE is not None); print(extract_5w1h({'title':'Tes','content':'Menteri Keuangan Sri Mulyani Indrawati menyampaikan kebijakan di Jakarta.','date':'2024-01-01'}))"
   ```
   Yang HARUS muncul: `NER aktif: True`, dan `who` berisi nama, `where`
   berisi `Jakarta` -- bukan `["Tidak disebutkan dalam artikel"]`.
   Kalau ini sudah benar, baru jalankan `streamlit run app.py` seperti biasa.

6. **Setelah app jalan**, cek juga apakah Streamlit benar-benar berhasil
   load modelnya -- lihat spinner "🏷️ Memuat Model NER Indonesia..." selesai
   tanpa redirect ke fallback. Kalau mau extra yakin, tambahkan sementara
   `st.write(f"NER pipeline aktif: {ner_pipeline is not None}")` di bawah
   baris `ner_pipeline = load_ner_model()` di `app.py` untuk verifikasi
   visual di UI (boleh dihapus lagi setelah yakin).

7. **Evaluasi sebelum vs sesudah** (opsional, kalau mau angka objektif):
   anotasi 30-50 sampel ground truth WHO/WHERE dari
   `cleaned_news_sample.csv`, lalu pakai `extraction/evaluate_who_where.py`
   untuk hitung precision/recall/F1.
