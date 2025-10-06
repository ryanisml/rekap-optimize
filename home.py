import streamlit as st
import pandas as pd
import traceback
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ----------------- helper functions -----------------
def make_unique_columns(cols):
    seen = {}
    new_cols = []
    for i, c in enumerate(cols):
        name = str(c).strip()
        if name.lower() in ('nan', 'none', ''):
            name = f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    return new_cols

def infer_header_and_df(df_raw):
    max_rows_to_check = min(5, df_raw.shape[0])
    best_idx = 0
    best_score = -1
    for i in range(max_rows_to_check):
        row = df_raw.iloc[i].astype(str).fillna('').str.strip()
        non_empty = (row != '') & (~row.str.replace('.', '', 1).str.isnumeric())
        score = non_empty.sum()
        if score > best_score:
            best_score = score
            best_idx = i
    header_row = df_raw.iloc[best_idx].tolist()
    cols = make_unique_columns(header_row)
    df = df_raw.iloc[best_idx+1:].reset_index(drop=True)
    df.columns = cols
    df = df.dropna(axis=1, how='all')
    df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_name_column(df):
    lowered = {c: str(c).lower() for c in df.columns}
    candidates = [c for c, lc in lowered.items() if any(k in lc for k in ['kec', 'kecamatan', 'kel', 'kelurahan', 'puskesmas', 'nama'])]
    if candidates:
        return candidates[0]
    for c in df.columns:
        sample = df[c].dropna().astype(str)
        if sample.shape[0] > 0 and not sample.iloc[0].replace('.', '', 1).isdigit():
            return c
    return df.columns[0]

def find_indicator_columns(df):
    lower_cols = {c: str(c).lower() for c in df.columns}
    stunting_col = None
    gizi_buruk_col = None
    for c, lc in lower_cols.items():
        if 'stunt' in lc or 'stunting' in lc:
            stunting_col = c
        if 'gizi buruk' in lc or ('gizi' in lc and 'buruk' in lc) or ('giziburuk' in lc) or ('kurang' in lc and 'gizi' in lc):
            gizi_buruk_col = c
    numeric_candidates = []
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors='coerce')
        if vals.notna().sum() > 0:
            numeric_candidates.append((c, int(vals.notna().sum())))
    numeric_candidates.sort(key=lambda x: -x[1])
    numeric_cols_sorted = [c for c, _ in numeric_candidates]
    if stunting_col is None and numeric_cols_sorted:
        stunting_col = numeric_cols_sorted[0]
    if gizi_buruk_col is None:
        if len(numeric_cols_sorted) > 1:
            gizi_buruk_col = numeric_cols_sorted[1]
        elif numeric_cols_sorted:
            gizi_buruk_col = numeric_cols_sorted[0]
    return {'stunting': stunting_col, 'gizi_buruk': gizi_buruk_col, 'numeric_cols': numeric_cols_sorted}

def to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil_Clustering')
    buf.seek(0)
    return buf

def categorize_by_percentiles(series):
    p33 = series.quantile(0.33)
    p66 = series.quantile(0.66)
    def cat(x):
        if pd.isna(x):
            return "Unknown"
        try:
            x = float(x)
        except:
            return "Unknown"
        if x <= p33:
            return "Rendah"
        elif x <= p66:
            return "Sedang"
        else:
            return "Tinggi"
    return cat, (p33, p66)

def make_cluster_explanation(cluster_id, mean_stunt, mean_gizi, st_cat, gizi_cat):
    parts = []
    parts.append(f"Cluster {cluster_id}: Rata-rata stunting = {mean_stunt:.2f}, rata-rata gizi buruk = {mean_gizi:.2f}.")
    if st_cat == "Tinggi" and gizi_cat == "Tinggi":
        parts.append("Interpretasi: STUNTING dan GIZI BURUK tinggi — PRIORITAS INTERVENSI.")
        parts.append("Rekomendasi: Program gizi segera, penyuluhan, screening balita.")
    elif st_cat == "Tinggi" and gizi_cat in ("Sedang", "Rendah"):
        parts.append("Interpretasi: Stunting tinggi — intervensi gizi dan penyelidikan determinan.")
        parts.append("Rekomendasi: Suplementasi, pemantauan tumbuh kembang.")
    elif st_cat in ("Sedang", "Rendah") and gizi_cat == "Tinggi":
        parts.append("Interpretasi: Gizi buruk relatif tinggi — fokus penanganan gizi akut.")
        parts.append("Rekomendasi: Penanganan kasus gizi buruk dan edukasi pemberian makanan.")
    else:
        parts.append("Interpretasi: Tingkat stunting & gizi buruk tergolong rendah/menengah — pertahankan program pencegahan.")
        parts.append("Rekomendasi: Promosi kesehatan, pemeliharaan layanan dasar.")
    return " ".join(parts)

# ----------------- main app -----------------
def main():
    st.title("K-Means Clustering - Stunting")
    st.write("Upload file Excel berisi data wilayah untuk analisis clustering.")

    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx/.xls)", type=["xlsx", "xls"])
    if uploaded_file is None:
        return

    try:
        data = uploaded_file.read()
        xls = pd.ExcelFile(BytesIO(data))
        sheet_names = xls.sheet_names
        default_sheet = 'ENTRY KEC' if 'ENTRY KEC' in sheet_names else sheet_names[0]
        sheet_choice = st.selectbox("Pilih sheet untuk analisis", options=sheet_names, index=sheet_names.index(default_sheet))

        df_raw = pd.read_excel(BytesIO(data), sheet_name=sheet_choice, header=None, engine='openpyxl')
        df = infer_header_and_df(df_raw)
        st.subheader("Preview data (sebagian)")
        st.dataframe(df.head())

        name_col = detect_name_column(df)
        indicators = find_indicator_columns(df)
        st.caption(f"Teridentifikasi kolom nama wilayah: **{name_col}**")
        st.caption(f"Teridentifikasi kolom stunting: **{indicators['stunting']}**, gizi buruk: **{indicators['gizi_buruk']}**")

        numeric_cols = indicators['numeric_cols']
        if not numeric_cols:
            st.error("Tidak ditemukan kolom numerik untuk clustering.")
            return

        X_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        X_df = X_df.fillna(X_df.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)

        n_samples = X_df.shape[0]
        if n_samples < 2:
            st.error("Jumlah data terlalu sedikit untuk clustering.")
            return

        k_min = 2
        k_max = max(2, min(10, n_samples - 1))

        st.sidebar.subheader("Pengaturan Cluster")
        use_auto_k = st.sidebar.checkbox("Gunakan auto-selection k (Silhouette)", value=True)
        k_manual = st.sidebar.slider("Jumlah cluster (manual)", min_value=k_min, max_value=k_max, value=3, step=1)

        inertias, silhouettes = [], []
        K_range = range(k_min, k_max + 1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            try:
                silhouettes.append(silhouette_score(X_scaled, labels))
            except:
                silhouettes.append(float('nan'))

        silhouette_series = pd.Series(silhouettes, index=K_range)
        if silhouette_series.dropna().shape[0] > 0:
            auto_k = int(silhouette_series.idxmax())
            auto_reason = f"Silhouette max = {silhouette_series.max():.3f}"
        else:
            inertia_diff = pd.Series(inertias, index=K_range).diff().abs()
            auto_k = int(inertia_diff.idxmax()) if inertia_diff.dropna().shape[0] > 0 else K_range[0]
            auto_reason = "Fallback inertia diff"

        st.sidebar.markdown(f"**Rekomendasi k:** {auto_k}  \n_{auto_reason}_")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(K_range, inertias, marker='o')
        ax[0].set_title("Elbow (Inertia)")
        ax[1].plot(K_range, silhouettes, marker='o')
        ax[1].set_title("Silhouette Score")
        plt.tight_layout()
        st.pyplot(fig)

        k_final = auto_k if use_auto_k else k_manual
        st.info(f"Menggunakan {k_final} cluster (K-Means)")

        kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
        labels_final = kmeans_final.fit_predict(X_scaled)
        df_result = df.copy()
        df_result["Cluster"] = labels_final.astype(int)

        # ---- Ringkasan ----
        st_col = indicators['stunting']
        gizi_col = indicators['gizi_buruk']
        summary_cols = [c for c in [st_col, gizi_col] if c in df_result.columns]
        if not summary_cols:
            summary_cols = numeric_cols[:2]

        df_result[summary_cols] = df_result[summary_cols].apply(pd.to_numeric, errors='coerce')
        cluster_summary = df_result.groupby('Cluster')[summary_cols].mean().round(2)
        cluster_summary["Count"] = df_result["Cluster"].value_counts().sort_index()

        dominant = {}
        for c in sorted(df_result["Cluster"].unique()):
            names = df_result.loc[df_result["Cluster"] == c, name_col]
            names = names[names.notna() & (names.astype(str).str.strip() != "")]
            if len(names) > 0:
                top = names.value_counts().head(3)
                dominant[c] = ", ".join([f"{n} ({cnt})" for n, cnt in top.items()])
            else:
                dominant[c] = "Tidak ada nama wilayah"
        cluster_summary["Dominant_Wilayah_top3"] = cluster_summary.index.map(lambda x: dominant.get(x, ""))

        st.subheader("Ringkasan Cluster")
        st.dataframe(cluster_summary)

        # ---- Penjelasan Otomatis ----
        all_st_vals = df_result[summary_cols[0]].dropna() if len(summary_cols) >= 1 else pd.Series(dtype=float)
        all_gizi_vals = df_result[summary_cols[1]].dropna() if len(summary_cols) >= 2 else all_st_vals
        st_cat_func, _ = categorize_by_percentiles(all_st_vals)
        gizi_cat_func, _ = categorize_by_percentiles(all_gizi_vals)

        st.subheader("Penjelasan Otomatis per Cluster")
        for c in cluster_summary.index:
            mean_st = float(cluster_summary.loc[c, summary_cols[0]]) if summary_cols[0] in cluster_summary else float('nan')
            mean_gizi = float(cluster_summary.loc[c, summary_cols[1]]) if len(summary_cols) > 1 and summary_cols[1] in cluster_summary else float('nan')
            st_cat = st_cat_func(mean_st)
            gizi_cat = gizi_cat_func(mean_gizi)
            explanation = make_cluster_explanation(c, mean_st, mean_gizi, st_cat, gizi_cat)
            st.markdown(f"**Cluster {c}** — {explanation}")

        # ---- Unduh ----
        excel_buf_summary = to_excel_bytes(cluster_summary.reset_index())
        st.download_button("Unduh ringkasan cluster", data=excel_buf_summary, file_name="ringkasan_cluster_kmeans.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        excel_buf_all = to_excel_bytes(df_result)
        st.download_button("Unduh hasil lengkap", data=excel_buf_all, file_name="hasil_kmeans_lengkap.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error("Terjadi kesalahan:")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
