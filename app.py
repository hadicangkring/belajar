import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka", layout="wide")

st.title("🔮 Prediksi Kombinasi Angka — A, B, dan C")
st.caption("Analisis berbasis transisi probabilistik + angka samaran (kode rahasia)")

# === Peta Angka Samaran ===
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
    9: [6, 3]
}

# === Fungsi bantu ===
def clean_number(x):
    """Ambil angka dari string, kembalikan None jika bukan angka"""
    if pd.isna(x): return None
    s = re.sub(r'\D', '', str(x))
    if s == '': return None
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
    combos = [int(f"{a}{b}{c}{d}") for a in variants[0] for b in variants[1] for c in variants[2] for d in variants[3]]
    return combos

def predict_next(numbers):
    """Prediksi nilai berikutnya berbasis transisi frekuensi"""
    if len(numbers) < 3:
        return np.nan
    transitions = defaultdict(Counter)
    for i in range(len(numbers)-1):
        transitions[numbers[i]][numbers[i+1]] += 1
    last = numbers[-1]
    if last not in transitions:  # fallback jika tidak ada transisi
        return int(np.median(numbers))
    next_val = transitions[last].most_common(1)[0][0]
    return int(next_val)

def process_file(file, name):
    st.subheader(f"📁 File {name.upper()}")
    df = pd.read_csv(file)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

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

# === Upload/Load Data ===
col1, col2, col3 = st.columns(3)
with col1:
    file_a = st.file_uploader("Unggah file A.csv", type=["csv"], key="a")
with col2:
    file_b = st.file_uploader("Unggah file B.csv", type=["csv"], key="b")
with col3:
    file_c = st.file_uploader("Unggah file C.csv", type=["csv"], key="c")

if file_a:
    process_file(file_a, "a")

if file_b:
    process_file(file_b, "b")

if file_c:
    process_file(file_c, "c")

st.markdown("---")
st.caption("© 2025 — Kombinasi Angka Engine by Ferri & GPT-5")
