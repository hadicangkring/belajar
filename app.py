import streamlit as st
import pandas as pd
from datetime import datetime
import os
from collections import defaultdict
import itertools
import numpy as np

# === Konfigurasi dasar ===
st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka", layout="centered")
st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")
st.caption("Prediksi otomatis berbasis data historis 6 digit (mengambil 4 digit terakhir).")

# === Fungsi bantu kalender Jawa sederhana ===
def get_hari_pasaran():
    hari_map = {"Senin":4, "Selasa":3, "Rabu":7, "Kamis":8, "Jumat":6, "Sabtu":9, "Minggu":5}
    pasaran_list = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
    today = datetime.now()
    hari = today.strftime("%A")
    hari = hari.replace("Monday","Senin").replace("Tuesday","Selasa").replace("Wednesday","Rabu") \
               .replace("Thursday","Kamis").replace("Friday","Jumat").replace("Saturday","Sabtu").replace("Sunday","Minggu")
    pasaran = pasaran_list[today.toordinal() % 5]
    neptu = hari_map[hari] + {"Legi":5,"Pahing":9,"Pon":7,"Wage":4,"Kliwon":8}[pasaran]
    return hari, pasaran, neptu

# === Hitung frekuensi 4 digit terakhir ===
def hitung_frekuensi(df):
    counts = [defaultdict(int) for _ in range(4)]  # [satuan, puluhan, ratusan, ribuan]
    for val in df.stack():
        try:
            s = str(int(val))[-4:]  # ambil 4 digit terakhir
            for i, d in enumerate(s[::-1]):  # dari satuan ke ribuan
                counts[i][d] += 1
        except Exception:
            continue
    return counts

# === Hitung probabilitas (%)
def probabilitas(counts):
    probs = []
    for pos in counts:
        total = sum(pos.values())
        probs.append({k: (v/total*100 if total else 0) for k,v in pos.items()})
    return probs

# === Kombinasi Top-5 Prediksi Gabungan ===
def top5_kombinasi(probs):
    top_per_pos = [sorted(p.items(), key=lambda x:x[1], reverse=True)[:3] for p in probs]
    kombinasi = []
    for rib, rat, pul, sat in itertools.product(top_per_pos[3], top_per_pos[2], top_per_pos[1], top_per_pos[0]):
        prob_total = rib[1] * rat[1] * pul[1] * sat[1]
        kombinasi.append(("".join([rib[0], rat[0], pul[0], sat[0]]), prob_total))
    kombinasi.sort(key=lambda x: x[1], reverse=True)
    return kombinasi[:5]

# === Ambil angka terakhir valid dari dataframe ===
def ambil_angka_terakhir(df):
    # Flatten semua isi dataframe jadi 1 list
    semua_nilai = df.replace('', np.nan).stack().dropna().tolist()
    if not semua_nilai:
        return "Tidak diketahui"
    try:
        # Ambil nilai terakhir yang valid
        val = str(semua_nilai[-1]).strip()
        # Ambil 4 digit terakhir, tetap tampilkan nol di depan jika ada
        val = ''.join([c for c in val if c.isdigit()])
        if len(val) >= 4:
            return val[-4:]
        elif len(val) > 0:
            return val.zfill(4)
        else:
            return "Tidak diketahui"
    except Exception:
        return "Tidak diketahui"

# === Tampilkan hasil prediksi ===
def tampilkan_prediksi(nama_file, df):
    hari, pasaran, neptu = get_hari_pasaran()
    counts = hitung_frekuensi(df)
    probs = probabilitas(counts)
    posisi = ["Satuan","Puluhan","Ratusan","Ribuan"]

    angka_terakhir = ambil_angka_terakhir(df)
    st.subheader(f"{nama_file}  (angka terakhir sebelum prediksi: **{angka_terakhir}**)")
    st.write(f"📅 Hari ini: **{hari} {pasaran} (Neptu {neptu})**")

    # === Tabel ringkas probabilitas ===
    st.write("### 🔍 Frekuensi & Probabilitas (Tabel Ringkas)")
    data_ringkas = {"Posisi": [], "Angka": [], "Persen (%)": []}

    for i, (count, prob) in enumerate(zip(counts, probs)):
        sorted_prob = sorted(prob.items(), key=lambda x:x[1], reverse=True)[:5]
        angka_list = ",".join([a for a,_ in sorted_prob])
        persen_list = ",".join([f"{p:.0f}" for _,p in sorted_prob])
        data_ringkas["Posisi"].append(posisi[i])
        data_ringkas["Angka"].append(angka_list)
        data_ringkas["Persen (%)"].append(persen_list)

    df_ringkas = pd.DataFrame(data_ringkas).set_index("Posisi").T
    st.dataframe(df_ringkas, use_container_width=True)

    # === Prediksi akhir ===
    prediksi = "".join(max(p.items(), key=lambda x:x[1])[0] for p in probs[::-1])
    st.success(f"🎯 Prediksi 4 Digit Terkuat: **{prediksi}**")

    # === Tambahan: Top-5 Kombinasi Prediksi Gabungan ===
    st.write("### 🔢 Top-5 Kombinasi Prediksi Gabungan")
    top5 = top5_kombinasi(probs)
    df_top5 = pd.DataFrame(top5, columns=["Kombinasi", "Skor Perkalian Probabilitas"])
    st.dataframe(df_top5, hide_index=True, use_container_width=True)

    # === Simpan log ===
    os.makedirs("logs", exist_ok=True)
    tgl = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/log_{nama_file.replace('.csv','')}_{tgl}.txt"
    with open(log_file, "w") as f:
        f.write(f"Prediksi file {nama_file}\n")
        f.write(f"Hari: {hari} {pasaran} (Neptu {neptu})\n")
        f.write(f"Angka terakhir sebelum prediksi: {angka_terakhir}\n")
        f.write(f"Prediksi 4 digit: {prediksi}\n\n")
        for i, (count, prob) in enumerate(zip(counts, probs)):
            f.write(f"[{posisi[i]}]\n")
            for k,v in sorted(prob.items(), key=lambda x:x[1], reverse=True):
                f.write(f" {k}: {v:.2f}%\n")
            f.write("\n")
        f.write("Top-5 Kombinasi:\n")
        for k,v in top5:
            f.write(f" {k}: {v:.2f}\n")
    st.caption(f"📁 Log tersimpan di: `{log_file}`")

# === Jalankan Prediksi untuk 3 file ===
st.header("🧮 Jalankan Prediksi")

for nama_file, keterangan in [("a.csv","📘 File A.CSV"),
                              ("b.csv","📗 File B.CSV"),
                              ("c.csv","📙 File C.CSV")]:
    if os.path.exists(nama_file):
        df = pd.read_csv(nama_file, header=None)
        tampilkan_prediksi(keterangan, df)
    else:
        st.warning(f"File {nama_file} belum ditemukan.")
