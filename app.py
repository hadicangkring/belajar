import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka — Streamlit Edition", layout="centered")

# === Fungsi bantu ===
def get_hari_pasaran():
    hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    pasaran_list = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
    now = datetime.now()
    hari = hari_list[now.weekday()]
    idx = (now.toordinal() + 3) % 5  # offset agar cocok kalender Jawa
    pasaran = pasaran_list[idx]
    hari_map = {"Senin": 4, "Selasa": 3, "Rabu": 7, "Kamis": 8, "Jumat": 6, "Sabtu": 9, "Minggu": 5}
    neptu = hari_map[hari] + {"Legi": 5, "Pahing": 9, "Pon": 7, "Wage": 4, "Kliwon": 8}[pasaran]
    return hari, pasaran, neptu

def hitung_frekuensi(df, is_six=False):
    counts = {0: {}, 1: {}, 2: {}, 3: {}}
    for row in df.itertuples(index=False):
        for i, ex in enumerate(row):
            if pd.isna(ex): continue
            counts[i][ex] = counts[i].get(ex, 0) + 1
    return counts

def tampilkan_prediksi(nama_file, df, is_six=False):
    hari, pasaran, neptu = get_hari_pasaran()
    st.markdown(f"### 📘 {nama_file}  \nHari ini: **{hari} {pasaran} (Neptu {neptu})**")

    # Cari angka terakhir sebelum prediksi
    last_row = df.dropna().tail(1)
    if not last_row.empty:
        last_number = ''.join(str(int(x)) for x in last_row.values.flatten() if str(x).isdigit())
        if last_number:
            st.markdown(f"*(angka terakhir sebelum prediksi: **{last_number}**)*")
        else:
            st.markdown(f"*(angka terakhir sebelum prediksi tidak diketahui)*")
    else:
        st.markdown(f"*(angka terakhir sebelum prediksi tidak diketahui)*")

    # Hitung frekuensi
    counts = hitung_frekuensi(df, is_six)

    # Buat tabel horizontal
    posisi = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
    data_freq, data_prob = [], []
    angka_dominan = []

    for i, pos in enumerate(posisi):
        frek = sorted(counts[i].items(), key=lambda x: x[1], reverse=True)[:5]
        if not frek: 
            data_freq.append(["-", "-", "-", "-", "-"])
            data_prob.append(["-", "-", "-", "-", "-"])
            angka_dominan.append("-")
            continue

        angka = [str(x[0]) for x in frek]
        jumlah = sum(x[1] for x in frek)
        prob = [round(100 * x[1] / jumlah, 1) for x in frek]

        data_freq.append(angka)
        data_prob.append(prob)

        # Ambil angka dominan (probabilitas tertinggi)
        angka_dominan.append(angka[0])

    # Susun dataframe horizontal
    df_view = pd.DataFrame({
        "Posisi": posisi,
        "Angka": [", ".join(a) for a in data_freq],
        "Persen (%)": [", ".join(str(p) for p in b) for b in data_prob],
    }).set_index("Posisi")

    # Styling tabel
    styled = df_view.style.set_properties(**{
        "text-align": "center",
        "background-color": "#E9F5FF",
        "border-color": "#A1C6EA",
    }).set_table_styles([
        {"selector": "th", "props": [("font-weight", "bold"), ("background-color", "#CFE7FF")]}
    ])

    st.dataframe(styled, use_container_width=True)

    # Tampilkan angka dominan
    st.markdown(f"**🔢 Angka Dominan:** {'–'.join(angka_dominan)}")

    # Simpan log otomatis
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/log_{nama_file.lower().replace('.csv','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"{nama_file}\nHari: {hari} {pasaran} (Neptu {neptu})\n")
        f.write(df_view.to_string())
        f.write(f"\n\nAngka Dominan: {'-'.join(angka_dominan)}")
    st.caption(f"📁 Log tersimpan di: `{log_file}`")

# === APLIKASI UTAMA ===
st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")
st.divider()

uploaded_files = st.file_uploader("📂 Unggah satu atau beberapa file CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        tampilkan_prediksi(file.name, df, is_six=True)
else:
    st.info("Unggah file CSV terlebih dahulu untuk menjalankan prediksi.")
