# app.py
import streamlit as st
import pandas as pd
from collections import defaultdict, Counter

st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka — Streamlit Edition", layout="centered")
st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")
st.caption("Prediksi otomatis berbasis data historis 6 digit (mengambil 4 digit terakhir).")

st.write("📅 Hari ini:", pd.Timestamp.now().strftime("%A %d-%m-%Y"))
st.divider()

# === Fungsi bantu ===
def pad4(x: int) -> str:
    """Pastikan jadi 4 digit"""
    return str(int(x)).zfill(4)

def ambil_angka_terakhir(df):
    """Ambil angka terakhir di baris paling bawah kolom paling kanan"""
    val = df.iloc[-1, -1]
    try:
        return pad4(val)
    except:
        return "0000"

def siapkan_data(df):
    """Ambil 4 digit terakhir dari tiap angka"""
    data = []
    for col in df.columns:
        for val in df[col]:
            try:
                v = pad4(str(val)[-4:])
                data.append(v)
            except:
                continue
    return data

# === Model Markov Ordo 2 ===
def markov_ordo2(data, alpha=0.5):
    """Bangun model Markov ordo 2"""
    transisi = defaultdict(Counter)
    for i in range(len(data) - 2):
        prev2 = (data[i], data[i + 1])
        next_val = data[i + 2]
        transisi[prev2][next_val] += 1
    return transisi

def prediksi_markov(transisi, last2, alpha=0.5, top_n=5):
    """Prediksi kombinasi 4 digit dan 2 digit"""
    counter = transisi.get(last2, Counter())
    total = sum(counter.values()) + alpha * len(counter)
    hasil = {k: (v + alpha) / total for k, v in counter.items()}
    urut = sorted(hasil.items(), key=lambda x: x[1], reverse=True)[:top_n]
    df4 = pd.DataFrame([{"Kombinasi 4 Digit": k, "Bobot": f"{p:.3f}"} for k, p in urut])
    df2 = pd.DataFrame([{"2 Digit": k[-2:], "Bobot": f"{p:.3f}"} for k, p in urut])
    return df4, df2

# === Tampilan Tabel ===
def tabel_mendatar(df):
    df = df.reset_index(drop=True)
    df.index = [f"Top {i+1}" for i in range(len(df))]
    return df.T

# === Tampilan Prediksi ===
def tampilkan_prediksi(file_name, judul, icon, alpha):
    st.subheader(f"{icon} {judul}")

    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        st.error(f"Gagal membaca {file_name}: {e}")
        return

    data = siapkan_data(df)
    if len(data) < 3:
        st.warning("Data terlalu sedikit untuk membentuk model.")
        return

    angka_terakhir = ambil_angka_terakhir(df)
    last2 = (data[-2], data[-1])

    st.write(f"Angka terakhir sebelum prediksi adalah: **{angka_terakhir}**")

    transisi = markov_ordo2(data, alpha)
    df4, df2 = prediksi_markov(transisi, last2, alpha)

    st.markdown("**Prediksi 4 Digit (Top 5):**")
    st.dataframe(tabel_mendatar(df4), use_container_width=True)

    st.markdown("**Prediksi 2 Digit (Top 5):**")
    st.dataframe(tabel_mendatar(df2), use_container_width=True)

# === Kontrol UI ===
st.sidebar.header("⚙️ Pengaturan Prediksi")
alpha = st.sidebar.slider("Nilai Alpha (Smoothing)", 0.0, 1.0, 0.5, 0.05)

if st.button("🔮 Jalankan Prediksi"):
    tampilkan_prediksi("a.csv", "File A", "📘", alpha)
    st.divider()
    tampilkan_prediksi("b.csv", "File B", "📗", alpha)
    st.divider()
    tampilkan_prediksi("c.csv", "File C", "📙", alpha)

    # === Gabungan ===
    st.divider()
    st.subheader("🧩 Gabungan Semua Data")
    try:
        df_a = pd.read_csv("a.csv")
        df_b = pd.read_csv("b.csv")
        df_c = pd.read_csv("c.csv")
        df_all = pd.concat([df_a, df_b, df_c], ignore_index=True)
        data_all = siapkan_data(df_all)
        if len(data_all) >= 3:
            angka_terakhir_all = ambil_angka_terakhir(df_all)
            last2_all = (data_all[-2], data_all[-1])
            st.write(f"Angka terakhir sebelum prediksi gabungan: **{angka_terakhir_all}**")
            transisi_all = markov_ordo2(data_all, alpha)
            df4_all, df2_all = prediksi_markov(transisi_all, last2_all, alpha)
            st.markdown("**Prediksi 4 Digit (Top 5 Gabungan):**")
            st.dataframe(tabel_mendatar(df4_all), use_container_width=True)
            st.markdown("**Prediksi 2 Digit (Top 5 Gabungan):**")
            st.dataframe(tabel_mendatar(df2_all), use_container_width=True)
        else:
            st.warning("Data gabungan masih terlalu sedikit.")
    except Exception as e:
        st.error(f"Gagal membaca data gabungan: {e}")
