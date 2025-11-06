# app.py
# 📊 Prediksi Kombinasi Angka — versi final (Streamlit)
# Ferri Kusuma & GPT-5, 2025

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import itertools
import os

# === KONFIGURASI DASAR ===
st.set_page_config(page_title="Prediksi Kombinasi Angka", layout="wide")
st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")

# === FUNGSI PEMBANTU ===

# Hari dan pasaran otomatis
from datetime import datetime

def get_hari_pasaran():
    # Hari dan pasaran sekarang (otomatis)
    hari = datetime.now().strftime("%A").lower()
    hari_map = {
        "monday": "senin",
        "tuesday": "selasa",
        "wednesday": "rabu",
        "thursday": "kamis",
        "friday": "jumat",
        "saturday": "sabtu",
        "sunday": "minggu"
    }

    hari_id = hari_map.get(hari, "kamis")  # fallback kamis

    pasaran_list = ["legi", "pahing", "pon", "wage", "kliwon"]
    base_date = datetime(2024, 1, 1)
    delta_days = (datetime.now() - base_date).days
    pasaran = pasaran_list[delta_days % 5]

    hari_val_map = {"senin":4, "selasa":3, "rabu":7, "kamis":8, "jumat":6, "sabtu":9, "minggu":5}
    pasaran_val_map = {"legi":5,"pahing":9,"pon":7,"wage":4,"kliwon":8}

    return hari_id, pasaran, hari_val_map[hari_id], pasaran_val_map[pasaran]

# Angka samaran
samaran = {
  0:[1,8], 1:[0,7], 2:[5,6], 3:[8,9], 4:[7,5],
  5:[2,4], 6:[9,2], 7:[4,1], 8:[3,0], 9:[6,3]
}

def expand_digit(d):
    """Kembalikan set angka yang mewakili digit + samaran"""
    d = int(d)
    return {d, samaran[d][0], samaran[d][1]}

def prepare_number(x, is_six=False):
    """Ambil 4 digit terakhir jika 6 digit, isi nol jika kurang"""
    try:
        x = str(int(x))
        if is_six and len(x) >= 6:
            x = x[-4:]
        return x.zfill(4)
    except:
        return None

def hitung_frekuensi(df, is_six=False):
    """Hitung frekuensi digit per posisi"""
    counts = [dict() for _ in range(4)]
    for val in df.dropna().astype(str):
        n = prepare_number(val, is_six)
        if not n:
            continue
        for i, ch in enumerate(n):
            for ex in expand_digit(ch):
                counts[i][ex] = counts[i].get(ex, 0) + 1
    return counts

def apply_bobot(counts, hari_val, pasaran_val):
    """Tambahkan bobot berdasarkan hari/pasaran"""
    for i in range(4):
        for d in list(counts[i].keys()):
            if d == hari_val:
                counts[i][d] *= 1.1
            if d == pasaran_val:
                counts[i][d] *= 1.1
    return counts

def prediksi_top5(counts):
    """Kombinasikan top 3 per posisi lalu ranking"""
    posisi_top = [sorted(c.items(), key=lambda x: x[1], reverse=True)[:3] for c in counts]
    combos = []
    for ribu, ratu, puluh, satu in itertools.product(*posisi_top):
        prob = ribu[1]*ratu[1]*puluh[1]*satu[1]
        combos.append(("{}{}{}{}".format(ribu[0],ratu[0],puluh[0],satu[0]), prob))
    top5 = sorted(combos, key=lambda x: x[1], reverse=True)[:5]
    total = sum(p for _,p in top5)
    return [(c, round(p/total*100,2)) for c,p in top5]

def log_hasil(file, top5, hari, pasaran, neptu):
    """Simpan ke log CSV"""
    log_file = "prediksi_log.csv"
    now = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S")
    df_log = pd.DataFrame([
        {"timestamp": now, "file": file, "prediksi": c, "prob": p,
         "hari": hari, "pasaran": pasaran, "neptu": neptu}
        for c,p in top5
    ])
    if os.path.exists(log_file):
        old = pd.read_csv(log_file)
        df_log = pd.concat([old, df_log], ignore_index=True)
    df_log.to_csv(log_file, index=False)

def tampilkan_prediksi(nama_file, df, is_six=False):
    st.subheader(f"📘 File {nama_file.upper()}")
    hari, pasaran, hari_val, pasar_val = get_hari_pasaran()
    neptu = hari_val + pasar_val
    st.caption(f"Hari ini: **{hari.capitalize()} {pasaran.capitalize()} (Neptu {neptu})**")

    # Perhitungan
    counts = hitung_frekuensi(df.stack(), is_six)
    counts = apply_bobot(counts, hari_val, pasar_val)
    top5 = prediksi_top5(counts)

    # Tampilkan hasil
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("**Top 5 Prediksi:**")
        for c,p in top5:
            st.write(f"🔹 {c} → {p}%")
    with col2:
        st.metric("Prediksi Utama", top5[0][0], f"{top5[0][1]}%")

    if st.button(f"💾 Simpan Log {nama_file.upper()}"):
        log_hasil(nama_file, top5, hari, pasaran, neptu)
        st.success("Disimpan ke prediksi_log.csv")

# === TAMPILAN UTAMA ===
st.markdown("---")
st.subheader("🧮 Jalankan Prediksi")

colA, colB, colC = st.columns(3)

with colA:
    if os.path.exists("a.csv"):
        df_a = pd.read_csv("a.csv")
        tampilkan_prediksi("a.csv", df_a, is_six=True)
    else:
        st.warning("File a.csv belum ditemukan.")

with colB:
    if os.path.exists("b.csv"):
        df_b = pd.read_csv("b.csv")
        tampilkan_prediksi("b.csv", df_b, is_six=False)
    else:
        st.warning("File b.csv belum ditemukan.")

with colC:
    if os.path.exists("c.csv"):
        df_c = pd.read_csv("c.csv")
        tampilkan_prediksi("c.csv", df_c, is_six=False)
    else:
        st.warning("File c.csv belum ditemukan.")

st.markdown("---")

# Tombol prediksi semua sekaligus
if st.button("🚀 Prediksi Semua Sekaligus"):
    for nama, df, six in [("a.csv", df_a, True), ("b.csv", df_b, False), ("c.csv", df_c, False)]:
        if "df_" + nama[0] in locals():
            hari, pasaran, h_val, p_val = get_hari_pasaran()
            neptu = h_val + p_val
            counts = hitung_frekuensi(df.stack(), six)
            counts = apply_bobot(counts, h_val, p_val)
            top5 = prediksi_top5(counts)
            log_hasil(nama, top5, hari, pasaran, neptu)
    st.success("✅ Semua hasil sudah diprediksi dan disimpan ke log.")
