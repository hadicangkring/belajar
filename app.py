import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from collections import Counter, defaultdict

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka", layout="wide")
st.title("🔮 Prediksi Kombinasi Angka — A, B, dan C")
st.caption("Analisis probabilistik & angka samaran — versi deploy dari repo")

DATA_PATH = "./data"  # folder di dalam repo (buat folder ini di GitHub)
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

def to_digits(num):
    """Pisahkan angka jadi ribuan, ratusan, puluhan, satuan"""
    s = str(num).zfill(4)
    return [int(s[-4]), int(s[-3]), int(s[-2]), int(s[-1])]

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

def predict_next(numbers):
    """Prediksi nilai berikutnya berbasis transisi frekuensi"""
    if len(numbers) < 3:
        return np.nan
    transitions = defaultdict(Counter)
    for i in range(len(numbers) - 1):
        transitions[numbers[i]][numbers[i + 1]] += 1
    last = numbers[-1]
    if last not in transitions:
        return int(np.median(numbers))
    next_val = transitions[last].most_common(1)[0][0]
    return int(next_val)

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
    for col in numeric_df.columns:
        series = numeric_df[col].dropna().astype(int).tolist()
        if len(series) == 0:
            continue
        expanded = []
        for n in series:
            expanded.extend(apply_alias(n))
        pred = predict_next(expanded)
        preds[col] = pred

    overall_pred = predict_next([v for v in preds.values() if not pd.isna(v)])

    st.markdown("#### 🔢 Prediksi per Kolom:")
    st.write(preds)

    st.markdown("#### 🎯 Prediksi Keseluruhan:")
    st.success(f"Prediksi keseluruhan (gabungan seluruh kolom): **{int(overall_pred)}**")

    return preds, overall_pred

# ===============================
# INPUT: AUTO LOAD ATAU UPLOAD
# ===============================
st.sidebar.header("📂 Pengaturan Data")
auto_load = st.sidebar.checkbox("Gunakan data dari repo (/data/)", value=True)

files_to_use = {}
for fname in FILES:
    label = fname.upper()
    if auto_load:
        file_path = os.path.join(DATA_PATH, fname)
        if os.path.exists(file_path):
            files_to_use[fname] = file_path
        else:
            st.warning(f"⚠️ File {fname} tidak ditemukan di {DATA_PATH}/")
    else:
        files_to_use[fname] = st.sidebar.file_uploader(f"Upload {fname}", type=["csv"], key=fname)

# ===============================
# PROSES FILE
# ===============================
st.markdown("---")
for fname in FILES:
    file_source = files_to_use.get(fname)
    if file_source:
        process_file(file_source, fname.split(".")[0])

st.markdown("---")
st.caption("© 2025 — Kombinasi Angka Engine by Ferri Kusuma & GPT-5")
