import streamlit as st
import pandas as pd
from datetime import datetime
import os
from collections import defaultdict, Counter

# === Konfigurasi dasar ===
st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka", layout="centered")
st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")
st.caption("Prediksi otomatis berbasis data historis 6 digit (mengambil 4 digit terakhir).")

# === Fungsi bantu ===
def get_tanggal():
    return datetime.now().strftime("%A %d-%m-%Y")

def ambil_angka_terakhir(df):
    """Ambil angka terakhir yang valid dari file"""
    for val in reversed(df.stack()):
        try:
            if pd.notna(val):
                return str(int(val)).zfill(6)
        except Exception:
            continue
    return "-"

def hitung_frekuensi(df):
    """Hitung frekuensi tiap posisi angka (6 digit tapi prediksi 4 digit terakhir)"""
    posisi = ["satuan", "puluhan", "ratusan", "ribuan"]
    counts = {p: Counter() for p in posisi}
    for val in df.stack():
        try:
            s = str(int(val)).zfill(6)[-4:]  # ambil 4 digit terakhir dari 6 digit
            for i, p in enumerate(posisi):
                counts[p][s[::-1][i]] += 1
        except Exception:
            continue
    return counts

def tabel_mendatar(frek):
    """Susun hasil frekuensi menjadi tabel mendatar top-5"""
    posisi = ["ribuan", "ratusan", "puluhan", "satuan"]
    data = {}
    for pos in posisi:
        angka_sorted = [k for k, v in frek[pos].most_common(5)]
        angka_sorted = (angka_sorted + ["-"] * 5)[:5]
        data[pos] = angka_sorted

    df = pd.DataFrame({
        "ribuan": data["ribuan"],
        "ratusan": data["ratusan"],
        "puluhan": data["puluhan"],
        "satuan": data["satuan"]
    }, index=[1, 2, 3, 4, 5])
    return df

def prediksi_dari_frek(frek):
    """Ambil angka dengan frekuensi tertinggi per posisi"""
    posisi = ["ribuan", "ratusan", "puluhan", "satuan"]
    hasil = "".join(frek[p].most_common(1)[0][0] if len(frek[p]) else "0" for p in posisi)
    return hasil

def angka_dominan(df):
    """Ambil 10 angka 4-digit yang paling sering muncul"""
    counter = Counter()
    for val in df.stack():
        try:
            s = str(int(val)).zfill(6)[-4:]
            counter[s] += 1
        except Exception:
            continue
    top10 = counter.most_common(10)
    return pd.DataFrame(top10, columns=["Kombinasi 4 Digit", "Frekuensi"])

# === Tampilan utama ===
st.header("🧮 Jalankan Prediksi")
st.write(f"📅 Hari ini: {get_tanggal()}")

def tampilkan_prediksi(nama_file, judul, warna_ikon):
    if os.path.exists(nama_file):
        df = pd.read_csv(nama_file, header=None)
        angka_terakhir = ambil_angka_terakhir(df)
        st.subheader(f"{warna_ikon} {judul}")
        st.caption(f"Angka terakhir sebelum prediksi adalah: **{angka_terakhir}**")

        frek = hitung_frekuensi(df)
        tabel = tabel_mendatar(frek)
        st.write("📊 **Frekuensi Angka per Posisi (Top-5)**")
        st.dataframe(tabel, use_container_width=True)

        hasil_prediksi = prediksi_dari_frek(frek)
        st.success(f"🎯 **Prediksi Kombinasi Teratas:** {hasil_prediksi}")

        st.write("🔥 **Angka Dominan (Top-10 Kombinasi 4 Digit)**")
        st.dataframe(angka_dominan(df), use_container_width=True)

        # === Simpan log ===
        os.makedirs("logs", exist_ok=True)
        tgl = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/log_{nama_file.replace('.csv','')}_{tgl}.txt"
        with open(log_file, "w") as f:
            f.write(f"File: {nama_file}\nTanggal: {get_tanggal()}\n")
            f.write(f"Angka terakhir: {angka_terakhir}\n")
            f.write(f"Prediksi: {hasil_prediksi}\n\n")
        st.caption(f"📁 Log tersimpan di: `{log_file}`")
    else:
        st.warning(f"File {nama_file} belum ditemukan.")

# === Jalankan untuk ketiga file ===
tampilkan_prediksi("a.csv", "File A", "📘")
tampilkan_prediksi("b.csv", "File B", "📗")
tampilkan_prediksi("c.csv", "File C", "📙")
