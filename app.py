# app.py
import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import os

# === KONFIGURASI DASAR ===
st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka — Streamlit Edition", layout="centered")
st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")
st.caption("Prediksi otomatis berbasis data historis 6 digit (mengambil 4 digit terakhir).")

# === DAFTAR FILE CSV ===
FILES = {
    "📘 File A": "a.csv",
    "📗 File B": "b.csv",
    "📙 File C": "c.csv"
}

# === Fungsi bantu ===
def baca_data(namafile):
    """Baca CSV dan ambil 6 digit terakhir, keluarkan juga 4 digit terakhir untuk prediksi."""
    df = pd.read_csv(namafile, header=None)
    df = df.dropna()
    df[0] = df[0].astype(str).str.replace(r'\D', '', regex=True)
    df = df[df[0].str.len() >= 4]
    df["6digit"] = df[0].str[-6:]
    df["4digit"] = df["6digit"].str[-4:]
    return df

def hitung_frekuensi(df):
    """Hitung frekuensi tiap digit berdasarkan posisi (ribu, ratus, puluh, satu)."""
    posisi = defaultdict(list)
    for val in df["4digit"]:
        if len(val) == 4:
            posisi["ribuan"].append(val[0])
            posisi["ratusan"].append(val[1])
            posisi["puluhan"].append(val[2])
            posisi["satuan"].append(val[3])

    hasil = {}
    for p in posisi:
        c = Counter(posisi[p])
        total = sum(c.values())
        frek = {k: v for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0]))}
        hasil[p] = {"angka": list(frek.keys()), "persen": [round((v/total)*100, 1) for v in frek.values()]}
    return hasil

def tabel_frekuensi(frek):
    """Buat tabel ringkas (angka dan persen) untuk setiap posisi."""
    tabel = pd.DataFrame({
        "ribuan": frek["ribuan"]["angka"],
        "ratusan": frek["ratusan"]["angka"],
        "puluhan": frek["puluhan"]["angka"],
        "satuan": frek["satuan"]["angka"]
    }).head(5)
    tabel_persen = pd.DataFrame({
        "ribuan": frek["ribuan"]["persen"],
        "ratusan": frek["ratusan"]["persen"],
        "puluhan": frek["puluhan"]["persen"],
        "satuan": frek["satuan"]["persen"]
    }).head(5)
    tabel_persen = tabel_persen.applymap(lambda x: f"{x}%" if pd.notnull(x) else "-")
    tabel.index = [f"angka {i+1}" for i in range(len(tabel))]
    tabel_persen.index = [f"persen {i+1}" for i in range(len(tabel_persen))]
    return tabel, tabel_persen

def kombinasi_terbaik(frek):
    """Ambil 5 kombinasi teratas dengan probabilitas tertinggi."""
    kombinasi = []
    for r in frek["ribuan"]["angka"][:5]:
        for s in frek["ratusan"]["angka"][:5]:
            for p in frek["puluhan"]["angka"][:5]:
                for u in frek["satuan"]["angka"][:5]:
                    prob = (
                        frek["ribuan"]["persen"][frek["ribuan"]["angka"].index(r)] *
                        frek["ratusan"]["persen"][frek["ratusan"]["angka"].index(s)] *
                        frek["puluhan"]["persen"][frek["puluhan"]["angka"].index(p)] *
                        frek["satuan"]["persen"][frek["satuan"]["angka"].index(u)]
                    )
                    kombinasi.append((f"{r}{s}{p}{u}", prob))
    kombinasi = sorted(kombinasi, key=lambda x: -x[1])[:5]
    return pd.DataFrame(kombinasi, columns=["Kombinasi", "Bobot (%)"])

def angka_dominan(df):
    """Ambil 10 digit paling sering muncul dari seluruh 6 digit."""
    semua = "".join(df["6digit"].dropna().astype(str))
    hitung = Counter(semua)
    dominan = sorted(hitung.items(), key=lambda x: (-x[1], x[0]))[:10]
    return pd.DataFrame(dominan, columns=["Angka", "Frekuensi"])

# === FUNGSI TAMPILKAN PREDIKSI ===
def tampilkan_prediksi(namafile, df):
    st.markdown(f"### {namafile}")
    if df.empty:
        st.warning("Data kosong atau tidak valid.")
        return

    # Ambil angka terakhir
    terakhir = df["6digit"].iloc[-1]
    st.write(f"Angka terakhir sebelum prediksi adalah: **{terakhir}**")

    frek = hitung_frekuensi(df)
    tabel, tabel_persen = tabel_frekuensi(frek)

    st.subheader("📊 Frekuensi Angka per Posisi")
    st.dataframe(tabel, use_container_width=True)
    st.dataframe(tabel_persen, use_container_width=True)

    st.subheader("🎯 Prediksi Kombinasi Teratas")
    st.dataframe(kombinasi_terbaik(frek), use_container_width=True)

    st.subheader("🔥 Angka Dominan (Top-10)")
    st.dataframe(angka_dominan(df), use_container_width=True)

# === MAIN ===
hari = datetime.now().strftime("%A %d-%m-%Y")
st.info(f"📅 Hari ini: {hari}")

# === PROSES SETIAP FILE ===
for nama, file in FILES.items():
    if os.path.exists(file):
        df = baca_data(file)
        tampilkan_prediksi(nama, df)
    else:
        st.error(f"File {file} tidak ditemukan.")
