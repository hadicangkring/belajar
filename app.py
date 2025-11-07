# app.py
import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

# === Fungsi bantu ===
def ambil_4digit_dan_2digit(nilai):
    """Ambil 4 digit terakhir dan 2 digit terakhir dari kolom angka"""
    nilai_str = str(int(nilai)).zfill(6)  # pastikan panjang 6 digit
    return nilai_str[-4:], nilai_str[-2:]

def siapkan_data(df):
    """Pastikan data bersih dan siap diproses"""
    df = df.dropna()
    df = df[df.iloc[:, 0].astype(str).str.isdigit()]
    df["angka"] = df.iloc[:, 0].astype(int)
    df["4digit"], df["2digit"] = zip(*df["angka"].apply(ambil_4digit_dan_2digit))
    return df

def hitung_prediksi(data, kolom, alpha=0.5):
    """Hitung prediksi probabilistik dengan exponential smoothing"""
    counter = Counter()
    bobot = 1.0
    total = 0.0

    for nilai in reversed(data[kolom].tolist()):
        counter[nilai] += bobot
        total += bobot
        bobot *= alpha

    hasil = {k: v / total for k, v in counter.items()}
    return sorted(hasil.items(), key=lambda x: x[1], reverse=True)[:10]

def tampilkan_prediksi(file, label, icon, alpha):
    """Tampilkan hasil prediksi dari file CSV"""
    try:
        df = pd.read_csv(file)
        data = siapkan_data(df)

        st.subheader(f"{icon} {label}")
        st.caption(f"Total data: {len(data)} baris")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 – 4 Digit Terakhir**")
            pred_4 = hitung_prediksi(data, "4digit", alpha)
            for angka, prob in pred_4:
                st.write(f"{angka} — {prob*100:.2f}%")

        with col2:
            st.markdown("**Top 10 – 2 Digit Terakhir**")
            pred_2 = hitung_prediksi(data, "2digit", alpha)
            for angka, prob in pred_2:
                st.write(f"{angka} — {prob*100:.2f}%")

        # Angka terakhir (paling kanan dari data terakhir)
        angka_terakhir = str(data.iloc[-1]["angka"]).zfill(6)[-1]
        st.info(f"🔚 **Angka terakhir (kanan): {angka_terakhir}**")

    except Exception as e:
        st.error(f"Gagal memproses {file}: {e}")

# === UI Streamlit ===
st.set_page_config(page_title="Prediksi Kombinasi Angka", layout="centered")

st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")
st.write("Prediksi otomatis berbasis data historis 6 digit (mengambil 4 digit terakhir dan 2 digit terakhir).")

st.markdown(f"📅 Hari ini: **{datetime.now().strftime('%A %d-%m-%Y')}**")

# Kontrol Alpha
st.sidebar.header("⚙️ Pengaturan Prediksi")
alpha = st.sidebar.slider("Nilai Alpha (Smoothing)", 0.00, 1.00, 0.5, 0.05)

# Jalankan prediksi untuk dua file
tampilkan_prediksi("a.csv", "File A", "📘", alpha)
tampilkan_prediksi("b.csv", "File B", "📗", alpha)
