# app.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import os
import math

st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka (Markov2)", layout="centered")
st.title("🔢 Prediksi Kombinasi Angka — Markov Ordo 2 (Samaran + Hari/Pasaran)")
st.caption("Model Markov orde-2; data: a.csv, b.csv, c.csv. Output: Top-K 6-digit (juga 4-digit terakhir).")

# --------------------
# Konfigurasi & mapping
# --------------------
ALIAS = {
    0: [1, 8],
    1: [0, 7],
    2: [5, 6],
    3: [8, 9],
    4: [7, 5],
    5: [2, 4],
    6: [9, 2],
    7: [4, 1],
    8: [3, 0],
    9: [6, 3],
}

HARI_MAP = {"senin":4,"selasa":3,"rabu":7,"kamis":8,"jumat":6,"sabtu":9,"minggu":5}
PASARAN_LIST = ["legi","pahing","pon","wage","kliwon"]
PASARAN_VAL = {"legi":5,"pahing":9,"pon":7,"wage":4,"kliwon":8}

FILES = [("a.csv","📘 File A"), ("b.csv","📗 File B"), ("c.csv","📙 File C")]

# --------------------
# Utility functions
# --------------------
def get_hari_pasaran():
    now = datetime.now()
    eng = now.strftime("%A").lower()
    eng_map = {"monday":"senin","tuesday":"selasa","wednesday":"rabu",
               "thursday":"kamis","friday":"jumat","saturday":"sabtu","sunday":"minggu"}
    hari = eng_map.get(eng, "kamis")
    pasaran = PASARAN_LIST[now.toordinal() % 5]
    return hari, pasaran, HARI_MAP[hari], PASARAN_VAL[pasaran]

def read_and_normalize(path):
    """Baca CSV fleksibel: cari nilai valid di tiap baris (kolom pertama non-empty),
       lalu normalisasi jadi 6-digit string (zfill). Kembalikan DataFrame dengan kolom 6digit,4digit.
    """
    df_raw = pd.read_csv(path, header=None, dtype=str)
    # bersihkan non-digit per cell
    df_clean = df_raw.applymap(lambda x: "" if pd.isna(x) else "".join(ch for ch in str(x) if ch.isdigit()))
    # ambil first non-empty cell per baris
    vals = []
    for _, row in df_clean.iterrows():
        first = ""
        for cell in row:
            if cell != "":
                first = cell
                break
        if first != "":
            vals.append(first)
    if not vals:
        return pd.DataFrame(columns=["6digit","4digit"])
    s = pd.Series(vals).astype(str)
    s6 = s.str[-6:].str.zfill(6)
    return pd.DataFrame({"6digit": s6, "4digit": s6.str[-4:]})

def ambil_angka_terakhir_baris_terbawah(df):
    """Ambil nilai di baris paling bawah yang valid (first non-empty cell),
       dikembalikan sebagai 6-digit string atau '-' jika tidak ada.
    """
    if df.empty:
        return "-"
    # df is DataFrame with '6digit' column
    try:
        val = df["6digit"].iloc[-1]
        if pd.isna(val) or val == "":
            return "-"
        return str(val).zfill(6)
    except Exception:
        return "-"

# --------------------
# Markov2 building & probs
# --------------------
def build_markov2_counts(series6):
    """series6: list of 6-digit strings"""
    counts = defaultdict(Counter)
    for s in series6:
        if not isinstance(s,str) or len(s) < 3:
            continue
        s6 = str(s).zfill(6)
        digits = list(s6)
        for i in range(len(digits)-2):
            a,b,c = digits[i], digits[i+1], digits[i+2]
            counts[(a,b)][c] += 1
    return counts

def cond_probs_from_counts(counts, alpha=1.0):
    """Laplace smoothing; return dict: (a,b) -> {d:prob} for d in '0'..'9'"""
    probs = {}
    for key, counter in counts.items():
        total = sum(counter.values()) + alpha*10
        probs[key] = {}
        for d in map(str, range(10)):
            probs[key][d] = (counter.get(d,0) + alpha) / total
    return probs

# fallback overall unigram probs from data if pair unseen
def unigram_probs_from_counts(counts):
    total_counts = Counter()
    for counter in counts.values():
        total_counts.update(counter)
    total = sum(total_counts.values()) or 1
    probs = {str(d): (total_counts.get(str(d),0)/total) for d in range(10)}
    # if all zeros, fallback uniform
    if sum(probs.values()) == 0:
        probs = {str(d): 1.0/10.0 for d in range(10)}
    return probs

# --------------------
# multiplier for candidate
# --------------------
def multiplier_for_candidate(prev_pair, candidate, hari_val, pasaran_val, use_samaran=True, use_hari=True, use_pasaran=True):
    a = int(prev_pair[0]); b = int(prev_pair[1]); c = int(candidate)
    m = 1.0
    if use_samaran:
        if c in ALIAS.get(a, []):
            m *= 1.15
        if c in ALIAS.get(b, []):
            m *= 1.12
    if use_hari and c == hari_val:
        m *= 1.10
    if use_pasaran and c == pasaran_val:
        m *= 1.10
    return m

# --------------------
# Beam search generator
# --------------------
def generate_top_k_markov2(start_pair, cond_probs, unigram_probs, hari_val, pasaran_val,
                           steps=4, beam_width=10, top_k=5,
                           use_samaran=True, use_hari=True, use_pasaran=True):
    # beams: list of (seq_str, logscore)
    beams = [( "".join(start_pair), 0.0 )]
    for step in range(steps):
        new_beams = []
        for seq, logscore in beams:
            a,b = seq[-2], seq[-1]
            key = (a,b)
            if key in cond_probs:
                cand_probs = cond_probs[key]
            else:
                cand_probs = unigram_probs
            # compute weighted scores
            scored = []
            for c, p in cand_probs.items():
                if p <= 0:
                    continue
                m = multiplier_for_candidate(key, c, hari_val, pasaran_val,
                                             use_samaran=use_samaran,
                                             use_hari=use_hari,
                                             use_pasaran=use_pasaran)
                score = p * m
                if score <= 0:
                    continue
                scored.append((c, score))
            # normalize local scores so beams from different parents comparable
            total = sum(s for _,s in scored) or 1.0
            for c, s_prob in scored:
                new_log = logscore + math.log(s_prob / total)
                new_beams.append((seq + c, new_log))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    # finalize
    beams.sort(key=lambda x: x[1], reverse=True)
    top = beams[:top_k]
    if not top:
        return []
    exps = [math.exp(b[1]) for b in top]
    s = sum(exps) or 1.0
    results = [(top[i][0], exps[i]/s) for i in range(len(top))]
    return results

# --------------------
# helpers UI tables (top5 per position, top10 combos)
# --------------------
def compute_position_top5(series6):
    counters = {"ribuan":Counter(), "ratusan":Counter(), "puluhan":Counter(), "satuan":Counter()}
    for s in series6:
        try:
            s6 = str(s).zfill(6)[-4:]
            counters["ribuan"][s6[0]] += 1
            counters["ratusan"][s6[1]] += 1
            counters["puluhan"][s6[2]] += 1
            counters["satuan"][s6[3]] += 1
        except Exception:
            continue
    top5 = {}
    for pos in ["ribuan","ratusan","puluhan","satuan"]:
        most = [k for k,_ in counters[pos].most_common(5)]
        most = (most + ["-"]*5)[:5]
        top5[pos] = most
    return top5

def top10_combinations(series6):
    ctr = Counter()
    for s in series6:
        try:
            ctr[str(s).zfill(6)[-4:]] += 1
        except Exception:
            continue
    return pd.DataFrame(ctr.most_common(10), columns=["Kombinasi 4 Digit","Frekuensi"])

# --------------------
# Streamlit UI: controls
# --------------------
st.sidebar.header("Model controls")
use_markov = st.sidebar.checkbox("Gunakan Markov Ordo-2", value=True)
use_samaran = st.sidebar.checkbox("Gunakan Angka Samaran", value=True)
use_hari = st.sidebar.checkbox("Gunakan Hari (neptu) sebagai faktor", value=True)
use_pasaran = st.sidebar.checkbox("Gunakan Pasaran (neptu) sebagai faktor", value=True)
beam_width = st.sidebar.slider("Beam width (lebar pencarian)", min_value=3, max_value=50, value=12, step=1)
top_k = st.sidebar.slider("Jumlah hasil (Top K)", 1, 10, 5)
alpha = st.sidebar.number_input("Laplace smoothing alpha", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

hari_name, pasaran_name, hari_val, pasaran_val = get_hari_pasaran()
st.write(f"📅 Hari ini: **{hari_name.capitalize()} {pasaran_name.capitalize()} (Neptu {hari_val+pasaran_val})**")
st.write("---")

# --------------------
# Process per file and show results
# --------------------
def process_and_show(path, title):
    st.subheader(title)
    if not os.path.exists(path):
        st.warning(f"File {path} tidak ditemukan.")
        return None, None
    df_norm = read_and_normalize(path)
    if df_norm.empty:
        st.warning("Tidak ada data valid pada file.")
        return None, None

    last6 = ambil_angka_terakhir_baris_terbawah(df_norm)
    st.caption(f"Angka terakhir sebelum prediksi (baris terbawah): **{last6}**")

    series6 = df_norm["6digit"].tolist()
    # Markov counts / probs
    counts = build_markov2_counts(series6)
    cond_probs = cond_probs_from_counts(counts, alpha=float(alpha))
    unigram = unigram_probs_from_counts(counts)

    # UI table top5 per position (mendatar)
    pos_top5 = compute_position_top5(series6)
    table_df = pd.DataFrame({
        "ribuan": pos_top5["ribuan"],
        "ratusan": pos_top5["ratusan"],
        "puluhan": pos_top5["puluhan"],
        "satuan": pos_top5["satuan"]
    }, index=[1,2,3,4,5])
    st.write("📊 Frekuensi Angka per Posisi (Top-5)")
    st.dataframe(table_df, use_container_width=True)

    # Predictions
    if use_markov:
        # start pair = last two digits of last6 (d5,d6)
        if last6 == "-" or len(last6) < 2:
            st.info("Tidak cukup data untuk menentukan pasangan awal (d5,d6).")
            preds = []
        else:
            start_pair = [last6[-2], last6[-1]]
            preds = generate_top_k_markov2(start_pair, cond_probs, unigram,
                                           hari_val, pasaran_val,
                                           steps=4, beam_width=int(beam_width), top_k=int(top_k),
                                           use_samaran=use_samaran, use_hari=use_hari, use_pasaran=use_pasaran)
    else:
        # fallback simple freq-based 6-digit construction:
        # pick most common per pos and append to last two digits
        # we'll create 4 new digits choosing most common per position (ribuan..satuan)
        most = {p: Counter() for p in ["ribuan","ratusan","puluhan","satuan"]}
        for s in series6:
            s4 = str(s).zfill(6)[-4:]
            most["ribuan"][s4[0]] += 1
            most["ratusan"][s4[1]] += 1
            most["puluhan"][s4[2]] += 1
            most["satuan"][s4[3]] += 1
        choice = lambda ctr: ctr.most_common(1)[0][0] if ctr else "0"
        new4 = [ choice(most["ribuan"]), choice(most["ratusan"]), choice(most["puluhan"]), choice(most["satuan"]) ]
        if last6 == "-" or len(last6) < 2:
            preds = []
        else:
            base = last6[-2] + last6[-1]
            seq = base + "".join(new4)
            preds = [(seq, 1.0)]
    # display predictions
    st.write("🧠 Prediksi (Top-{}) 6-digit  — model: {}".format(top_k, "Markov2" if use_markov else "Frekuensi"))
    if not preds:
        st.info("Tidak ada prediksi (data/awal tidak cukup).")
    else:
        df_preds = pd.DataFrame([{"rank": i+1, "prediksi_6d": seq, "prediksi_4d": seq[-4:], "score": round(score,4)}
                                 for i,(seq,score) in enumerate(preds)])
        st.table(df_preds.set_index("rank"))

    # dominant 4-digit combos
    st.write("🔥 Angka Dominan (Top-10 Kombinasi 4 Digit)")
    st.dataframe(top10_combinations(series6), use_container_width=True)

    # log
    os.makedirs("logs", exist_ok=True)
    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"logs/markov2_{os.path.basename(path).replace('.csv','')}_{tstamp}.txt"
    with open(logfile, "w", encoding="utf-8") as f:
        f.write(f"File: {path}\nTanggal: {datetime.now()}\n")
        f.write(f"Angka terakhir (baris terbawah): {last6}\n")
        f.write("Prediksi top-k (6-digit, score):\n")
        for seq,score in preds:
            f.write(f"{seq}  {score:.6f}\n")
    st.caption(f"📁 Log tersimpan di: `{logfile}`")
    return series6, preds

# --------------------
# Run each file
# --------------------
st.header("Hasil per file")
all_series = []
for path, title in FILES:
    series6, preds = process_and_show(path, title)
    if series6:
        all_series.extend(series6)
st.write("---")

# --------------------
# Combined model (gabungan A+B+C)
# --------------------
st.header("📦 Prediksi Gabungan (A + B + C)")
if all_series:
    # build counts & probs from combined all_series
    counts_all = build_markov2_counts(all_series)
    cond_probs_all = cond_probs_from_counts(counts_all, alpha=float(alpha))
    unigram_all = unigram_probs_from_counts(counts_all)
    # determine last6 for combined: we take last valid row bottom from each file's df? simpler: use last element of all_series
    last6_combined = all_series[-1] if all_series else "-"
    st.caption(f"Angka terakhir gabungan (ambil terakhir dari gabungan): **{last6_combined}**")
    if last6_combined == "-" or len(last6_combined) < 2:
        st.info("Tidak cukup data untuk prediksi gabungan.")
    else:
        start_pair = [last6_combined[-2], last6_combined[-1]]
        preds_comb = generate_top_k_markov2(start_pair, cond_probs_all, unigram_all,
                                           hari_val, pasaran_val,
                                           steps=4, beam_width=int(beam_width), top_k=int(top_k),
                                           use_samaran=use_samaran, use_hari=use_hari, use_pasaran=use_pasaran)
        if not preds_comb:
            st.info("Tidak ada prediksi gabungan.")
        else:
            df_preds_c = pd.DataFrame([{"rank":i+1,"prediksi_6d":seq,"prediksi_4d":seq[-4:],"score":round(score,4)}
                                       for i,(seq,score) in enumerate(preds_comb)])
            st.table(df_preds_c.set_index("rank"))
else:
    st.info("Gabungan kosong (tidak ada data valid di file A/B/C).")

# --------------------
# Footer / quick tips
# --------------------
st.write("---")
st.write("Tips: coba ubah `Beam width` & toggle `Use Samaran / Use Hari / Use Pasaran` di sidebar untuk melihat bagaimana hasil prediksi berubah.")
