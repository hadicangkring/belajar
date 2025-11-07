# app.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import os
import itertools
import math

st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka (Markov2)", layout="centered")
st.title("🔢 Prediksi Kombinasi Angka — Markov Ordo 2 (with Samaran & Neptu)")
st.caption("Model: Markov orde-2 + angka samaran + hari & pasaran Jawa. Data: a.csv, b.csv, c.csv")

# --- Angka samaran (mapping yang kamu berikan) ---
ALIAS = {
    0: [1,8],
    1: [0,7],
    2: [5,6],
    3: [8,9],
    4: [7,5],
    5: [2,4],
    6: [9,2],
    7: [4,1],
    8: [3,0],
    9: [6,3],
}

# --- Hari & pasaran mapping (numerik) ---
HARI_MAP = {"senin":4,"selasa":3,"rabu":7,"kamis":8,"jumat":6,"sabtu":9,"minggu":5}
PASARAN_LIST = ["legi","pahing","pon","wage","kliwon"]
PASARAN_VAL = {"legi":5,"pahing":9,"pon":7,"wage":4,"kliwon":8}

# --- util: dapatkan hari & pasaran saat ini ---
def get_hari_pasaran():
    now = datetime.now()
    hari_eng = now.strftime("%A").lower()
    eng_to_id = {"monday":"senin","tuesday":"selasa","wednesday":"rabu",
                 "thursday":"kamis","friday":"jumat","saturday":"sabtu","sunday":"minggu"}
    hari_id = eng_to_id.get(hari_eng, "kamis")
    pasaran = PASARAN_LIST[now.toordinal() % 5]
    return hari_id, pasaran, HARI_MAP[hari_id], PASARAN_VAL[pasaran]

# --- baca & normalisasi: setiap sel jadi 6-digit string (zfill) ---
def read_and_normalize(path):
    df = pd.read_csv(path, header=None, dtype=str)
    # keep everything as string, remove non-digit chars
    df = df.applymap(lambda x: ("" if pd.isna(x) else "".join(ch for ch in str(x) if ch.isdigit())))
    # drop entirely-empty rows
    df = df[df.apply(lambda row: any(cell != "" for cell in row), axis=1)].copy()
    # create 6-digit normalized column per row: concatenate cells? 
    # We treat each non-empty cell as one "value" (previous app used single-column files). 
    # To keep compatibility with earlier structure (single column per file), assume first non-empty cell per row is value
    # So create series values = first non-empty cell in each row
    vals = []
    for _, row in df.iterrows():
        first = ""
        for cell in row:
            if cell != "":
                first = cell
                break
        vals.append(first)
    s = pd.Series(vals)
    # only keep those with at least 1 digit
    s = s[s.str.len() > 0].reset_index(drop=True)
    # normalize to 6 digits (if shorter, zfill -> leading zeros)
    s6 = s.str[-6:].str.zfill(6)
    # store dataframe with both 6 and 4 digit variants
    res = pd.DataFrame({"6digit": s6, "4digit": s6.str[-4:]})
    return res

# --- build Markov-2 transition counts from a series of 6-digit strings ---
# We'll treat each 6-digit as sequence of digits d1..d6 and extract transitions:
# (d1,d2)->d3, (d2,d3)->d4, (d3,d4)->d5, (d4,d5)->d6
def build_markov2_counts(series6):
    counts = defaultdict(lambda: Counter())
    # counts[(a,b)][c] = occurrences
    for s in series6:
        if not isinstance(s, str) or len(s) < 4:
            continue
        # ensure 6-digit
        s6 = s.zfill(6)[-6:]
        digits = list(s6)
        # iterate triples
        for i in range(len(digits)-2):
            a, b, c = digits[i], digits[i+1], digits[i+2]
            counts[(a,b)][c] += 1
    return counts

# --- get conditional probabilities P(c | a,b) with simple smoothing ---
def cond_probs_from_counts(counts, alpha=1.0):
    probs = {}
    for key, counter in counts.items():
        total = sum(counter.values()) + alpha * 10  # add-one smoothing scaled
        probs[key] = {}
        for d in map(str, range(10)):
            cnt = counter.get(d, 0) + alpha
            probs[key][d] = cnt / total
    return probs

# --- scoring multiplier f based on samaran, hari, pasaran ---
def multiplier_for_candidate(prev_pair, candidate, hari_val, pasaran_val):
    # prev_pair: tuple of two chars ('a','b')
    a, b = int(prev_pair[0]), int(prev_pair[1])
    c = int(candidate)
    m = 1.0
    # boost if candidate is alias of a or b
    if c in ALIAS.get(a, []):
        m *= 1.15
    if c in ALIAS.get(b, []):
        m *= 1.12
    # boost if equal to hari_val or pasaran_val
    if c == hari_val:
        m *= 1.10
    if c == pasaran_val:
        m *= 1.10
    return m

# --- beam search generator using conditional probs and multiplier ---
def generate_top_k_markov2(start_pair, cond_probs, hari_val, pasaran_val, steps=4, beam_width=10, top_k=5):
    # start_pair: tuple/list of 2 chars e.g. ['5','1'] -> we will append steps digits to make length 2+steps
    BeamItem = tuple  # (sequence_str, log_score)
    beams = [( "".join(start_pair), 0.0 )]  # log-score (use log to avoid underflow)
    for step in range(steps):
        new_beams = []
        for seq, logscore in beams:
            a, b = seq[-2], seq[-1]
            key = (a,b)
            # get candidate probs for this key, fallback to uniform if unseen
            if key in cond_probs:
                cand_probs = cond_probs[key]
            else:
                # fallback: uniform over 0..9
                cand_probs = {str(d): 1.0/10.0 for d in range(10)}
            # evaluate each candidate with multiplier
            scored = []
            for c, p in cand_probs.items():
                m = multiplier_for_candidate(key, c, hari_val, pasaran_val)
                score = p * m
                # avoid zero
                if score <= 0:
                    continue
                scored.append((c, score))
            # normalize scored to probabilities (so comparables across different seq origins)
            total = sum(s for _, s in scored) or 1.0
            for c, s_prob in scored:
                new_log = logscore + math.log(s_prob / total)
                new_beams.append((seq + c, new_log))
        # keep top beam_width
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
        if not beams:
            break
    # after steps, beams contain sequences of length start+steps
    beams.sort(key=lambda x: x[1], reverse=True)
    topk = beams[:top_k]
    # convert log-scores back to normalized probability-like scores (soft)
    # we'll exponentiate and normalize
    exps = [math.exp(item[1]) for item in topk]
    total_exps = sum(exps) or 1.0
    results = [(item[0], exps[i]/total_exps) for i,item in enumerate(topk)]
    return results

# --- helper: build freqs/top5-per-position (mendatar) like before ---
def compute_position_top5(series6):
    # compute counts for 4-digit last positions
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
    # top5 lists
    top5 = {}
    for pos in ["ribuan","ratusan","puluhan","satuan"]:
        most = [k for k,_ in counters[pos].most_common(5)]
        most = (most + ["-"]*5)[:5]
        top5[pos] = most
    return top5

# --- read files and run model per file ---
FILES = [("a.csv","📘 File A"), ("b.csv","📗 File B"), ("c.csv","📙 File C")]

hari_name, pasaran_name, hari_val, pasaran_val = get_hari_pasaran()
st.write(f"📅 Hari ini: **{hari_name.capitalize()} {pasaran_name.capitalize()} (Neptu {hari_val+pasaran_val})**")

def process_file(path, title):
    st.subheader(title)
    if not os.path.exists(path):
        st.warning(f"File {path} tidak ditemukan.")
        return
    df_norm = read_and_normalize(path)  # columns: 6digit, 4digit
    if df_norm.empty:
        st.warning("Tidak ada data valid.")
        return

    # show last value (6-digit)
    last6 = df_norm["6digit"].iloc[-1]
    st.caption(f"Angka terakhir sebelum prediksi adalah: **{last6}**")

    # Build Markov-2 counts and conditional probs
    counts = build_markov2_counts(df_norm["6digit"].tolist())
    cond_probs = cond_probs_from_counts(counts, alpha=1.0)

    # compute per-position top5 (for UI table mendatar)
    pos_top5 = compute_position_top5(df_norm["6digit"].tolist())
    # prepare dataframe mendatar (Top-5 digits per column)
    table_df = pd.DataFrame({
        "ribuan": pos_top5["ribuan"],
        "ratusan": pos_top5["ratusan"],
        "puluhan": pos_top5["puluhan"],
        "satuan": pos_top5["satuan"]
    }, index=[1,2,3,4,5])
    st.write("📊 Frekuensi Angka per Posisi (Top-5)")
    st.dataframe(table_df, use_container_width=True)

    # --- GENERATE Top-5 sequences 6-digit by Markov2 ---
    # start pair = last two digits of last6
    start_pair = [last6[-6:-4], last6[-4:-2]]  # careful: last6 is 6-digit string; take positions
    # But if last6 is length 6 string: digits d1..d6, we want last two digits: d5,d6
    # adjust: simpler:
    start_pair = [ last6[-2], last6[-1] ]
    # generate
    results = generate_top_k_markov2(start_pair, cond_probs,
                                     hari_val, pasaran_val,
                                     steps=4, beam_width=20, top_k=5)
    st.write("🧠 Prediksi Markov Ordo-2 (Top-5 - 6 digit)")
    # show also last 4 digits of each predicted
    df_res = pd.DataFrame([
        {"rank": i+1, "prediksi_6d": seq, "prediksi_4d": seq[-4:], "score": f"{score:.3f}"}
        for i,(seq,score) in enumerate(results)
    ])
    if df_res.empty:
        st.info("Tidak cukup data transisi untuk menghasilkan prediksi Markov2.")
    else:
        st.table(df_res[["rank","prediksi_6d","prediksi_4d","score"]].set_index("rank"))

    # show top-10 dominant 4-digit combos (as before)
    st.write("🔥 Angka Dominan (Top-10 Kombinasi 4 Digit)")
    dom = Counter()
    for s in df_norm["6digit"].tolist():
        dom[s[-4:]] += 1
    dom_df = pd.DataFrame(dom.most_common(10), columns=["Kombinasi 4 Digit","Frekuensi"])
    st.dataframe(dom_df, use_container_width=True)

    # save simple log
    os.makedirs("logs", exist_ok=True)
    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"logs/markov2_{os.path.basename(path).replace('.csv','')}_{tstamp}.txt"
    with open(logfile, "w") as f:
        f.write(f"File: {path}\n")
        f.write(f"Tanggal: {datetime.now()}\n")
        f.write(f"Angka terakhir: {last6}\n\n")
        f.write("Top-5 Markov2 predictions (6-digit,score):\n")
        for seq, score in results:
            f.write(f"{seq}  {score:.4f}\n")
    st.caption(f"📁 Log tersimpan: `{logfile}`")

# run for each file
st.header("🧮 Hasil untuk file")
for path, title in FILES:
    process_file(path, title)
