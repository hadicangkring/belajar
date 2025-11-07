import streamlit as st
import pandas as pd
from datetime import datetime
import os
from collections import defaultdict

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

# === Tampilkan hasil prediksi ===
def tampilkan_prediksi(nama_file, df):
    hari, pasaran, neptu = get_hari_pasaran()
    st.write(f"📅 Hari ini: **{hari} {pasaran} (Neptu {neptu})**")

    # Ambil angka terakhir sebelum prediksi
    last_number = None
    for val in reversed(df.stack()):
        try:
            last_number = str(int(val))[-4:]
            break
        except Exception:
            continue
    if last_number:
        st.subheader(f"{nama_file} (angka terakhir sebelum prediksi: {last_number})")
    else:
        st.subheader(f"{nama_file} (angka terakhir tidak diketahui)")

    counts = hitung_frekuensi(df)
    probs = probabilitas(counts)
    posisi = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]

    # === Buat tabel horizontal
    data = []
    angka_dominan = []
    for i in range(3, -1, -1):  # urutan ribuan ke satuan
        pos = posisi[3 - i]
        sorted_prob = sorted(probs[i].items(), key=lambda x:x[1], reverse=True)
        top5 = sorted_prob[:5]
        if top5:
            angka_dominan.append(top5[0][0])
        row = {
            "Posisi": pos,
            "Digit": ", ".join(k for k,_ in top5),
            "Probabilitas (%)": ", ".join(f"{v:.1f}" for _,v in top5)
        }
        data.append(row)

    df_show = pd.DataFrame(data)
    st.dataframe(df_show.style.set_properties(**{
        "text-align": "center",
        "background-color": "#EAF4FF",
    }).set_table_styles([{"selector":"th", "props":[("font-weight","bold"),("background-color","#D6E9FF")]}]),
    use_container_width=True, hide_index=True)

    # === Prediksi akhir ===
    prediksi = "".join(angka_dominan)
    st.success(f"🎯 Prediksi 4 Digit Terkuat: **{prediksi}**")
    st.markdown(f"**🔢 Angka Dominan:** {'–'.join(angka_dominan)}")

    # === Simpan log ===
    os.makedirs("logs", exist_ok=True)
    tgl = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/log_{nama_file.replace('.csv','')}_{tgl}.txt"
    with open(log_file, "w") as f:
        f.write(f"Prediksi file {nama_file}\n")
        f.write(f"Hari: {hari} {pasaran} (Neptu {neptu})\n")
        f.write(f"Angka terakhir sebelum prediksi: {last_number}\n")
        f.write(f"Prediksi 4 digit: {prediksi}\n\n")
        for r in data:
            f.write(f"{r['Posisi']}: {r['Digit']} ({r['Probabilitas (%)']}%)\n")
        f.write(f"\nAngka Dominan: {'-'.join(angka_dominan)}")
    st.caption(f"📁 Log tersimpan di: `{log_file}`")


# === Jalankan untuk tiga file tetap ===
st.header("🧮 Jalankan Prediksi")
for nama_file, keterangan in [("a.csv","📘 File A.CSV"),
                              ("b.csv","📗 File B.CSV"),
                              ("c.csv","📙 File C.CSV")]:
    if os.path.exists(nama_file):
        df = pd.read_csv(nama_file, header=None)
        tampilkan_prediksi(keterangan, df)
    else:
        st.warning(f"File {nama_file} belum ditemukan.")
