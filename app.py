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

FILES = {
    "📘 File A": "a.csv",
    "📗 File B": "b.csv",
    "📙 File C": "c.csv"
}

# === Fungsi bantu ===
def baca_data(namafile):
    df = pd.read_csv(namafile, header=None)
    df = df.dropna()
    df[0] = df[0].astype(str).str.replace(r'\D', '', regex=True)
    df = df[df[0].str.len() >= 4]
    df["6digit"] = df[0].str[-6:]
    df["4digit"] = df["6digit"].str[-4:]
    return df

def hitung_frekuensi(df):
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
        hasil[p] = {
            "angka": list(frek.keys())[:5],
            "persen": [round((v/total)*100, 1) for v in list(frek.values())[:5]]
        }
    return hasil

def tabel_mendatar(frek):
    """Tabel mendatar seperti contoh: baris 'angka' dan 'persen'."""
    df = pd.DataFrame({
        "ribuan": frek["ribuan"]["angka"],
        "ratusan": frek["ratusan"]["angka"],
        "puluhan": frek["puluhan"]["angka"],
        "satuan": frek["satuan"]["angka"]
    }).T
    df.columns = [f"{i+1}" for i in range(df.shape[1])]
    df.loc["persen"] = [
        ", ".join([f"{x:.1f}" for x in frek["ribuan"]["persen"]]),
        ", ".join([f"{x:.1f}" for x in frek["ratusan"]["persen"]]),
        ", ".join([f"{x:.1f}" for x in frek["puluhan"]["persen"]]),
        ", ".join([f"{x:.1f}" for x in frek["satuan"]["persen"]]),
    ]
    return df

def kombinasi_terbaik(frek):
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
                    kombinasi.append((f"{r}{s}{p}{u}", round(prob / 10000, 4)))
    kombinasi = sorted(kombinasi, key=lambda x: -x[1])[:5]
    return pd.DataFrame(kombinasi, columns=["Kombinasi", "Bobot (%)"])

def angka_dominan(df):
    semua = "".join(df["6digit"].dropna().astype(str))
    hitung = Counter(semua)
    dominan = sorted(hitung.items(), key=lambda x: (-x[1], x[0]))[:10]
    return pd.DataFrame(dominan, columns=["Angka", "Frekuensi"])

# === Tampilan Prediksi ===
def tampilkan_prediksi(namafile, df):
    st.markdown(f"### {namafile}")
    if df.empty:
        st.warning("Data kosong atau tidak valid.")
        return

    terakhir = df["6digit"].iloc[-1]
    st.write(f"Angka terakhir sebelum prediksi adalah: **{terakhir}**")

    frek = hitung_frekuensi(df)
    tabel = tabel_mendatar(frek)

    st.subheader("📊 Frekuensi & Probabilitas (Mendatar)")
    st.dataframe(tabel, use_container_width=True)

    st.subheader("🎯 Prediksi Kombinasi Teratas")
    st.dataframe(kombinasi_terbaik(frek), use_container_width=True)

    st.subheader("🔥 Angka Dominan (Top-10)")
    st.dataframe(angka_dominan(df), use_container_width=True)

# === MAIN ===
hari = datetime.now().strftime("%A %d-%m-%Y")
st.info(f"📅 Hari ini: {hari}")

for nama, file in FILES.items():
    if os.path.exists(file):
        df = baca_data(file)
        tampilkan_prediksi(nama, df)
    else:
        st.error(f"File {file} tidak ditemukan.")
