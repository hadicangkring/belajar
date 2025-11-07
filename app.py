import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# === KONFIGURASI DASAR ===
st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka — Streamlit Edition", layout="wide")
st.title("🔢 Prediksi Kombinasi Angka — Streamlit Edition")
st.caption("📊 Versi dengan tabel frekuensi & probabilitas ringkas horizontal")

# === PETA HARI & PASARAN ===
hari_map = {
    "senin": 4, "selasa": 3, "rabu": 7, "kamis": 8, "jumat": 6, "sabtu": 9, "minggu": 5
}
pasaran_map = {
    "legi": 5, "pahing": 9, "pon": 7, "wage": 4, "kliwon": 8
}

# === FUNGSI HARI & PASARAN OTOMATIS ===
def get_hari_pasaran():
    now = datetime.now()
    hari = now.strftime("%A").lower()
    hari = {"monday": "senin", "tuesday": "selasa", "wednesday": "rabu",
            "thursday": "kamis", "friday": "jumat", "saturday": "sabtu", "sunday": "minggu"}[hari]
    idx = (now.toordinal() % 5)
    pasaran = list(pasaran_map.keys())[idx]
    return hari, pasaran, hari_map[hari], pasaran_map[pasaran]


# === FUNGSI UTAMA ===
def pad4(x):
    """Pastikan angka jadi 4 digit (ambil 4 digit terakhir)."""
    try:
        s = str(int(str(x).strip()))
        return s[-4:].zfill(4)
    except:
        return None


def hitung_frekuensi(series, is_six=False):
    """Hitung frekuensi tiap posisi digit."""
    counts = {0: {}, 1: {}, 2: {}, 3: {}}
    for val in series:
        try:
            s = str(int(str(val).strip()))
            if is_six:
                s = s[-6:].zfill(6)
            else:
                s = s[-4:].zfill(4)
            for i in range(4):
                ex = s[-4:][i]
                counts[i][ex] = counts[i].get(ex, 0) + 1
        except:
            continue
    return counts


def tampilkan_prediksi(nama_file, df, is_six=False):
    st.markdown(f"### 📘 File **{nama_file.upper()}**")

    hari, pasaran, hari_val, pasar_val = get_hari_pasaran()
    st.write(f"Hari ini: **{hari.capitalize()} {pasaran.capitalize()}** (Neptu {hari_val + pasar_val})")

    # Hapus sel kosong dan non-numeric
    df = df.applymap(lambda x: x if str(x).strip().isdigit() else np.nan).dropna(how="all")

    # Ambil kolom terakhir
    last_col = df.iloc[:, -1].dropna().astype(str)
    counts = hitung_frekuensi(last_col, is_six=is_six)

    hasil_prediksi = {}
    hasil_persen = {}

    for i, nama in enumerate(["ribuan", "ratusan", "puluhan", "satuan"]):
        frek = counts[i]
        total = sum(frek.values()) or 1
        sorted_items = sorted(frek.items(), key=lambda x: x[1], reverse=True)
        angka = [k for k, _ in sorted_items[:5]]
        persen = [round(v / total * 100, 2) for _, v in sorted_items[:5]]
        hasil_prediksi[nama] = angka
        hasil_persen[nama] = persen

    # Tampilkan tabel ringkas
    tabel_data = {
        "Posisi": ["Angka", "Persen (%)"],
        "Ribuan": [", ".join(hasil_prediksi["ribuan"]),
                   ", ".join(map(str, hasil_persen["ribuan"]))],
        "Ratusan": [", ".join(hasil_prediksi["ratusan"]),
                    ", ".join(map(str, hasil_persen["ratusan"]))],
        "Puluhan": [", ".join(hasil_prediksi["puluhan"]),
                    ", ".join(map(str, hasil_persen["puluhan"]))],
        "Satuan": [", ".join(hasil_prediksi["satuan"]),
                   ", ".join(map(str, hasil_persen["satuan"]))],
    }

    st.table(pd.DataFrame(tabel_data).set_index("Posisi"))

    # Kombinasi top 5
    top1 = "".join(v[0] if v else "0" for v in hasil_prediksi.values())
    st.success(f"🎯 Prediksi utama: **{top1}**")

    all_combos = []
    from itertools import product
    for combo in product(*hasil_prediksi.values()):
        prob = np.prod([
            hasil_persen["ribuan"][hasil_prediksi["ribuan"].index(combo[0])] if combo[0] in hasil_prediksi["ribuan"] else 1,
            hasil_persen["ratusan"][hasil_prediksi["ratusan"].index(combo[1])] if combo[1] in hasil_prediksi["ratusan"] else 1,
            hasil_persen["puluhan"][hasil_prediksi["puluhan"].index(combo[2])] if combo[2] in hasil_prediksi["puluhan"] else 1,
            hasil_persen["satuan"][hasil_prediksi["satuan"].index(combo[3])] if combo[3] in hasil_prediksi["satuan"] else 1,
        ])
        all_combos.append(("".join(combo), prob))
    top5 = sorted(all_combos, key=lambda x: x[1], reverse=True)[:5]

    st.markdown("#### 🧮 Top-5 Kombinasi Prediksi:")
    top_df = pd.DataFrame(top5, columns=["Kombinasi", "Probabilitas (%)"])
    top_df["Probabilitas (%)"] = top_df["Probabilitas (%)"].apply(lambda x: round(x, 2))
    st.dataframe(top_df, hide_index=True, use_container_width=True)

    # Simpan log otomatis
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/log_{nama_file.replace('.csv','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_path, "w") as f:
        f.write(f"Prediksi {nama_file} - {datetime.now()}\n")
        f.write(f"Hari: {hari} {pasaran} (Neptu {hari_val+pasar_val})\n\n")
        f.write(top_df.to_string(index=False))
    st.info(f"📁 Log tersimpan otomatis di: `{log_path}`")


# === LOAD FILE ===
for nama_file, is_six in [("a.csv", True), ("b.csv", False), ("c.csv", False)]:
    if os.path.exists(nama_file):
        df = pd.read_csv(nama_file, header=None)
        tampilkan_prediksi(nama_file, df, is_six=is_six)
    else:
        st.warning(f"File **{nama_file}** belum ditemukan.")
