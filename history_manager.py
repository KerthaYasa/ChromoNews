"""
history_manager.py
===================
Fitur History / riwayat pencarian (catatan dosen):
    "Usahakan menambahkan fitur History atau penyimpanan riwayat
    percakapan. Riwayat dapat ditampilkan pada artikel yang pernah
    dicari sehingga pengguna dapat melihat kembali hasil pencarian
    sebelumnya."

Implementasi: riwayat disimpan sebagai file JSON di disk (search_history.json)
supaya bertahan lintas sesi Streamlit (bukan cuma st.session_state yang hilang
begitu browser di-refresh/ditutup). Setiap entri berisi query, waktu pencarian,
dan hasil (judul, tanggal, skor, hasil 5W1H, paragraf ringkasan) sehingga bisa
ditampilkan ulang tanpa perlu re-search / re-extract.

Desain sengaja sederhana (file JSON, bukan database) supaya tidak menambah
dependency baru dan mudah dibaca/dihapus manual kalau perlu.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

HISTORY_FILE = "search_history.json"
MAX_HISTORY_ENTRIES = 30


def _to_jsonable(value):
    """Pastikan semua nilai (termasuk numpy float32 dari skor RRF) aman di-JSON-kan."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def load_history(path: str = HISTORY_FILE) -> List[Dict[str, Any]]:
    """Muat riwayat dari file JSON. Kembalikan list kosong jika belum ada / rusak."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"[history_manager] Gagal memuat riwayat, mulai dari kosong: {e}")
        return []


def save_history(history: List[Dict[str, Any]], path: str = HISTORY_FILE) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(history), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[history_manager] Gagal menyimpan riwayat: {e}")


def add_entry(
    query: str,
    articles: List[Dict[str, Any]],
    history: List[Dict[str, Any]] = None,
    path: str = HISTORY_FILE,
) -> List[Dict[str, Any]]:
    """
    Tambah 1 entri riwayat pencarian baru (query + hasil), simpan ke disk,
    dan kembalikan list riwayat terbaru (untuk sinkron ke st.session_state).

    Entri BARU selalu ditaruh di depan (riwayat terbaru muncul pertama).
    Query yang identik dengan entri PALING BARU tidak diduplikasi (mis.
    kalau pengguna klik "Cari" dua kali berturut-turut dengan query sama),
    melainkan entri lama diganti (waktu & hasil di-update).
    """
    if history is None:
        history = load_history(path)

    entry = {
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_results": len(articles),
        "articles": [
            {
                "title": a.get("title", ""),
                "date": a.get("date", ""),
                "content": a.get("content", ""),
                "score": a.get("score", 0.0),
                "w5h1": a.get("w5h1", {}),
                "paragraph": a.get("paragraph", ""),
            }
            for a in articles
        ],
    }

    if history and history[0].get("query", "").strip().lower() == query.strip().lower():
        history[0] = entry
    else:
        history.insert(0, entry)

    history = history[:MAX_HISTORY_ENTRIES]
    save_history(history, path)
    return history


def clear_history(path: str = HISTORY_FILE) -> List[Dict[str, Any]]:
    """Hapus seluruh riwayat (dipakai tombol 'Hapus Riwayat' di UI)."""
    save_history([], path)
    return []
