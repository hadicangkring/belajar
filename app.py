import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from collections import Counter, defaultdict
from itertools import product

# ===============================
# KONFIGURASI AWAL
# ===============================
st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka", layout="wide")
st.title("🔮 Prediksi Kombinasi Angka — File A, B, dan C")
st.caption("Analisis probabilistik dengan angka samaran & Markov sederhana")

DATA_PATH = "./data"
FILES = ["a.csv", "b.csv", "c.csv"]

# ===============================
# PETA ANGKA SAMARAN
# ===============================
alias_map = {
    0: [1, 8],
    1: [0, 7],
    2: [5, 6],
    3: [8, 9],
    4: [7, 5],
    5: [2, 4],
    6: [9, 2],
    7: [4, 1],
    8: [3, 0],
    9: [6, 3],
}

# ===============================
# FUNGSI PEMROSESAN
# ===============================
def clean_number(x):
    """Ambil angka dari string, kembalikan None jika bukan angka"""
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    if s == "":
        return None
    return int(s)

def pad4(num):
    """Pastikan angka 4 digit"""
    s = str(int(num))[-4:]
    return s.zfill(4)

def to_digits(num):
    """Pisahkan angka jadi ribuan, ratusan, puluhan, satuan"""
    s = pad4(num)
    return [int(s[0]), int(s[1]), int(s[2]), int(s[3])]

def apply_alias(num):
    """Tambahkan variasi alias (angka samaran)"""
    digits = to_digits(num)
    variants = []
    for d in digits:
        variants.append([d] + alias_map[d])
    combos = [
        int(f"{a}{b}{c}{d}")
        for a in variants[0]
        for b in variants[1]
        for c in variants[2]
        for d in variants[3]
    ]
    return combos

def build_weighted_markov(values, weights):
    """Bangun model transisi probabilitas berbobot"""
    transitions = defaultdict(lambda: defaultdict(float))
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        transitions[a][b] += weights[i]
    probs = {}
    for a in transitions:
        total = sum(transitions[a].values())
        probs[a] = {b: transitions[a][b] / total for b in transitions[a]}
    return probs

def predict_next(numbers):
    """Prediksi nilai berikutnya berdasarkan model Markov berbobot"""
    if len(numbers) < 3:
        return np.nan, {}
    weights = np.linspace(0.5, 1.0, len(numbers))
    probs = build_weighted_markov(numbers, weights)
    last = numbers[-1]
    if last not in probs:
        return int(np.median(numbers)), {}
    dist = probs[last]
    next_val = max(dist, key=dist.get)
    return int(next_val), dist

def top5_from_distribution(dist):
    """Ambil 5 angka teratas dari distribusi probabilitas"""
    return sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]

# ===============================
# FUNGSI PROSES SATU FILE
# ===============================
def process_file(path, name):
    st.subheader(f"📁 File {name.upper()}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Gagal membaca file {path}: {e}")
        return None, None

    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    numeric_df = df.applymap(clean_number)
    st.dataframe(numeric_df)

    preds = {}
    distribs = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna().astype(int).tolist()
        if len(series) == 0:
            continue

        # Perluas dengan alias (angka samaran)
        expanded = []
        for n in series:
            expanded.extend(apply_alias(n))

        pred, dist = predict_next(expanded)
        preds[col] = pred
        distribs[col] = dist

    overall_series = [v for v in preds.values() if not pd.isna(v)]
    overall_pred, overall_dist = predict_next(overall_series)

    # ===============================
    # TAMPILKAN HASIL
    # ===============================
    st.markdown("#### 🔢 Prediksi per Kolom")
    for col, pred in preds.items():
        st.write(f"**Kolom {col}** → Prediksi: `{pred}`")

    st.markdown("#### 📊 Probabilitas per Kolom (Top-5)")
    for col, dist in distribs.items():
        if not dist:
            continue
        st.write(f"**Kolom {col}**")
        top5 = top5_from_distribution(dist)
        df_top5 = pd.DataFrame(
            [(a, f"{p*100:.2f}%") for a, p in top5],
            columns=["Angka", "Probabilitas"]
        )
        st.table(df_top5)

    st.markdown("#### 🎯 Prediksi Keseluruhan (Gabungan Seluruh Kolom)")
    st.success(f"**{int(overall_pred)}** — nilai prediksi utama dari seluruh kolom")

    if overall_dist:
        top5_all = top5_from_distribution(overall_dist)
        st.markdown("##### 🧠 Top-5 Prediksi Keseluruhan:")
        df_top5_all = pd.DataFrame(
            [(a, f"{p*100:.2f}%") for a, p in top5_all],
            columns=["Angka", "Probabilitas"]
        )
        st.table(df_top5_all)

    st.markdown("---")
    return preds, overall_pred

# ===============================
# INPUT: AUTO LOAD ATAU UPLOAD
# ===============================
st.sidebar.header("📂 Pengaturan Data")
auto_load = st.sidebar.checkbox("Gunakan data dari repo (/data/)", value=True)

files_to_use = {}
for fname in FILES:
    if auto_load:
        file_path = os.path.join(DATA_PATH, fname)
        if os.path.exists(file_path):
            files_to_use[fname] = file_path
        else:
            st.warning(f"⚠️ File {fname} tidak ditemukan di {DATA_PATH}/")
    else:
        files_to_use[fname] = st.sidebar.file_uploader(f"Upload {fname}", type=["csv"], key=fname)

# ===============================
# PROSES SEMUA FILE
# ===============================
st.markdown("---")
for fname in FILES:
    file_source = files_to_use.get(fname)
    if file_source:
        process_file(file_source, fname.split(".")[0])

st.markdown("---")
st.caption("© 2025 — Kombinasi Angka Engine by Ferri Kusuma & GPT-5")
