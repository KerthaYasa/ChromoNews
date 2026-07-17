"""
03_hitung_validasi.py
======================
LANGKAH 3 (VERSI: MANUSIA vs AI): baca 'penilaian.xlsx' (sheet
Penilaian_Manusia dan Penilaian_AI yang sudah diisi), lalu:

  A. Hitung MAP per komponen + MAP TOTAL, TERPISAH untuk manusia dan AI.
  B. Bandingkan skor manusia vs skor AI per komponen:
     - Agreement rate (persentase artikel yang skornya SAMA persis
       antara manusia dan AI)
     - Cohen's Kappa (mengukur kesepakatan manusia-AI di luar faktor
       kebetulan)

CATATAN METODOLOGI (WAJIB masuk laporan):
"Penilai kedua" di sini adalah AI (Claude), bukan manusia kedua yang
independen. Jadi Kappa yang dihasilkan mengukur KESEPAKATAN MANUSIA-AI,
bukan reliabilitas antar-manusia (inter-rater reliability klasik).
Tetap dicantumkan sebagai keterbatasan, tapi berguna sebagai bentuk
validasi silang kuantitatif -- dan tetap lebih informatif daripada
tidak ada pembanding sama sekali.
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
except ModuleNotFoundError as e:
    print(f"Modul belum terinstall: {e.name}")
    print("Jalankan dulu: pip install pandas openpyxl scikit-learn")
    raise SystemExit(1)

BASE_DIR = Path(__file__).resolve().parent
PENILAIAN_FILE = BASE_DIR / "penilaian.xlsx"

KOMPONEN = ["who", "what", "when", "where", "why", "how"]


def load_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    for k in KOMPONEN:
        col = f"skor_{k}"
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di sheet '{sheet_name}'")
    return df


def hitung_map(df, label):
    print("=" * 60)
    print(f"MAP (Mean Average Precision) -- {label}")
    print("=" * 60)

    complete = df.dropna(subset=[f"skor_{k}" for k in KOMPONEN]).copy()
    n_kosong = len(df) - len(complete)
    if n_kosong > 0:
        print(f"PERINGATAN: {n_kosong} baris di '{label}' masih ada skor kosong "
              f"-- dikeluarkan dari perhitungan.")

    if complete.empty:
        print(f"Belum ada data lengkap untuk hitung MAP ({label}).")
        return None

    print(f"\nMAP per komponen ({label}):")
    for k in KOMPONEN:
        skor_rata = complete[f"skor_{k}"].mean()
        print(f"  {k.upper():6s}: {skor_rata:.3f}  ({skor_rata*100:.1f}%)")

    complete["AP"] = complete[[f"skor_{k}" for k in KOMPONEN]].mean(axis=1)
    map_total = complete["AP"].mean()
    print(f"\nMAP TOTAL ({label}, n={len(complete)} artikel): "
          f"{map_total:.3f}  ({map_total*100:.1f}%)\n")

    return complete


def bandingkan_manusia_ai(df_manusia, df_ai):
    print("=" * 60)
    print("PERBANDINGAN MANUSIA vs AI (agreement rate & Cohen's Kappa)")
    print("=" * 60)

    merged = df_manusia.merge(
        df_ai, on="artikel_id", suffixes=("_manusia", "_ai")
    )
    if merged.empty:
        print("Tidak ada artikel_id yang cocok antara sheet manusia dan AI.")
        return

    print(f"Jumlah artikel yang dibandingkan: {len(merged)}\n")

    hasil = {}
    for k in KOMPONEN:
        col_m = f"skor_{k}_manusia"
        col_a = f"skor_{k}_ai"
        pair = merged.dropna(subset=[col_m, col_a])
        if pair.empty:
            print(f"  {k.upper():6s}: tidak ada pasangan skor lengkap, dilewati.")
            continue

        n = len(pair)
        agree = (pair[col_m] == pair[col_a]).sum()
        agreement_rate = agree / n

        # Cohen's Kappa butuh variasi label; kalau semua skor sama (mis. semua 1),
        # kappa tidak terdefinisi secara matematis -> ditandai NaN.
        try:
            if pair[col_m].nunique() < 2 and pair[col_a].nunique() < 2:
                kappa = float("nan")
            else:
                kappa = cohen_kappa_score(pair[col_m], pair[col_a])
        except Exception:
            kappa = float("nan")

        hasil[k] = {"agreement_rate": agreement_rate, "kappa": kappa, "n": n}
        kappa_str = f"{kappa:.3f}" if kappa == kappa else "tidak terdefinisi (skor tidak bervariasi)"
        print(f"  {k.upper():6s}: agreement = {agreement_rate:.3f} ({agreement_rate*100:.1f}%), "
              f"Cohen's Kappa = {kappa_str}   (n={n})")

    # ringkasan keseluruhan (semua komponen digabung jadi satu daftar label)
    all_m, all_a = [], []
    for k in KOMPONEN:
        col_m = f"skor_{k}_manusia"
        col_a = f"skor_{k}_ai"
        pair = merged.dropna(subset=[col_m, col_a])
        all_m.extend(pair[col_m].tolist())
        all_a.extend(pair[col_a].tolist())

    if all_m:
        overall_agreement = sum(m == a for m, a in zip(all_m, all_a)) / len(all_m)
        try:
            overall_kappa = cohen_kappa_score(all_m, all_a) if len(set(all_m + all_a)) > 1 else float("nan")
        except Exception:
            overall_kappa = float("nan")
        kappa_str = f"{overall_kappa:.3f}" if overall_kappa == overall_kappa else "tidak terdefinisi"
        print(f"\nRINGKASAN SELURUH KOMPONEN (n={len(all_m)} pasangan skor):")
        print(f"  Agreement rate keseluruhan : {overall_agreement:.3f} ({overall_agreement*100:.1f}%)")
        print(f"  Cohen's Kappa keseluruhan  : {kappa_str}")

    return hasil


def main():
    if not PENILAIAN_FILE.exists():
        print(f"{PENILAIAN_FILE} belum ada / belum diisi. Jalankan dulu "
              f"02_buat_template_penilaian.py, isi sheet Penilaian_Manusia, "
              f"dan jalankan 02b_isi_skor_ai_claude.py untuk sheet Penilaian_AI.")
        sys.exit(1)

    df_manusia = load_sheet(PENILAIAN_FILE, "Penilaian_Manusia")
    df_ai = load_sheet(PENILAIAN_FILE, "Penilaian_AI")

    complete_manusia = hitung_map(df_manusia, "MANUSIA")
    complete_ai = hitung_map(df_ai, "AI (Claude)")

    if complete_manusia is not None and complete_ai is not None:
        bandingkan_manusia_ai(complete_manusia, complete_ai)

    print("\n" + "=" * 60)
    print("CATATAN UNTUK LAPORAN")
    print("=" * 60)
    print("MAP manusia dan MAP AI dilaporkan terpisah sebagai dua sudut pandang penilaian.")
    print("Agreement rate & Cohen's Kappa manusia-vs-AI dipakai sebagai validasi silang,")
    print("BUKAN pengganti reliabilitas antar-manusia (karena penilai kedua adalah AI,")
    print("bukan manusia independen). Ini dicantumkan sebagai keterbatasan metodologis.")


if __name__ == "__main__":
    main()