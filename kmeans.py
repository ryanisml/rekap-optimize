import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

@st.cache_data(show_spinner=False)
def read_file(file, sheet_name=None):
    if file is None:
        return None, None
    name = getattr(file, "name", "").lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            return df, None
        else:
            xls = pd.ExcelFile(file)
            sheets = xls.sheet_names
            if sheet_name is None:
                preferred = [s for s in sheets if "entry" in s.lower() or "data" in s.lower()]
                sheet = preferred[0] if preferred else sheets[0]
            else:
                sheet = sheet_name
            df = pd.read_excel(xls, sheet_name=sheet)
            return df, sheets
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None, None


def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)


def get_default_features(df):
    if df is None:
        return []
    candidates = [
        "Overweight","Stunting","Normal"
    ]
    return [c for c in candidates if c in df.columns and is_numeric_series(df[c])]


def to_excel_bytes(df_dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=name[:31])
    buf.seek(0)
    return buf


def compute_silhouette_scores(X, k_max=10, random_state=42):
    scores = []
    k_range = range(2, min(k_max, len(X)) + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(X)
        try:
            sc = silhouette_score(X, labels)
        except Exception:
            sc = np.nan
        scores.append((k, sc))
    return scores


def show():
    st.title("🔬 K-Means Clustering")
    st.caption("Analisis klaster untuk data stunting/gizi — pilih fitur, tentukan jumlah klaster (atau biarkan auto), dan unduh hasil.")

    with st.expander("📥 Upload Data", expanded=True):
        file = st.file_uploader("Unggah berkas (.csv / .xlsx)", type=["csv","xlsx"], accept_multiple_files=False)
        sheet_name = None
        df, sheets = read_file(file, sheet_name)
        if file is not None and sheets:
            sheet_name = st.selectbox("Pilih sheet Excel", options=sheets, index=0)
            df, _ = read_file(file, sheet_name)

    if df is None:
        st.info("Unggah data untuk mulai analisis.")
        st.stop()

    id_cols = [c for c in df.columns if not is_numeric_series(df[c])]
    suggest_id = None
    for c in ["Puskesmas","PUSKESMAS","Nama Puskesmas","Puskesmas Name","Kode"]:
        if c in id_cols:
            suggest_id = c
            break

    with st.expander("⚙️ Pilih fitur & pengaturan", expanded=True):
        numeric_cols = [c for c in df.columns if is_numeric_series(df[c])]
        default_feats = get_default_features(df) or numeric_cols[: min(6, len(numeric_cols))]
        features = st.multiselect("Pilih fitur numerik untuk clustering", options=numeric_cols, default=default_feats)
        id_col = st.selectbox("Kolom identitas (opsional)", options=[None] + id_cols,
                              index=(0 if suggest_id is None else ([None] + id_cols).index(suggest_id)))
        scale = st.checkbox("Standarisasi fitur (StandardScaler)", value=True)
        use_auto_k = st.checkbox("Pilih k otomatis menggunakan Silhouette (rentang 2-10)", value=True)
        k_manual = st.number_input("Jumlah cluster (k) jika tidak otomatis", min_value=2, max_value=20, value=2, step=1)
        random_state = st.number_input("Random state", value=42, step=1)

        work = df.copy()
        work = work.replace([np.inf, -np.inf], np.nan)
        if features:
            sel = work[features].astype(float).fillna(0.0)
        else:
            st.error("Pilih minimal satu fitur numerik.")
            st.stop()

    X = sel.values
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Determine k and compute silhouette scores for visualization
    sils = compute_silhouette_scores(X_scaled, k_max=10, random_state=int(random_state))
    sils_dict = {k: s for k, s in sils}

    if use_auto_k:
        sils_clean = [(k, s) for k, s in sils if not np.isnan(s)]
        if sils_clean:
            best_k = max(sils_clean, key=lambda x: x[1])[0]
        else:
            best_k = int(k_manual)
        k = int(best_k)
        st.success(f"Memilih k = {k} berdasarkan silhouette.")
    else:
        k = int(k_manual)

    # Elbow plot (WCSS)
    st.subheader("🔍 Diagram Elbow (Within-Cluster Sum of Squares)")
    # compute WCSS (inertia) for k=1..10 (or up to len(X))
    k_max_elbow = min(10, len(X))
    k_range_elbow = list(range(1, k_max_elbow + 1))
    wcss = []
    for kk in k_range_elbow:
        km_temp = KMeans(n_clusters=kk, random_state=int(random_state))
        km_temp.fit(X_scaled)
        try:
            wcss.append(km_temp.inertia_)
        except Exception:
            wcss.append(np.nan)
    fig_elbow = plt.figure(figsize=(6,3.5))
    plt.plot(k_range_elbow, wcss, marker='o')
    plt.xticks(k_range_elbow)
    plt.xlabel('k (jumlah cluster)')
    plt.ylabel('WCSS (inertia)')
    plt.title('Elbow Plot: WCSS vs k')
    plt.grid(True)
    # mark chosen k if within range
    if k in k_range_elbow:
        idx_el = k_range_elbow.index(k)
        plt.scatter([k], [wcss[idx_el]], s=120, facecolors='none', edgecolors='black', linewidths=2, label=f"Chosen k = {k}")
        plt.legend()
    st.pyplot(fig_elbow, use_container_width=True)

    # Show silhouette plot
    # Davies-Bouldin index plot
    st.subheader("📈 Davies-Bouldin Index untuk k = 2..10")
    k_range_db = [k for k,_ in sils]
    dbs = []
    for kk in k_range_db:
        try:
            km_tmp = KMeans(n_clusters=kk, random_state=int(random_state))
            labs = km_tmp.fit_predict(X_scaled)
            db = davies_bouldin_score(X_scaled, labs)
        except Exception:
            db = np.nan
        dbs.append(db)
    fig_db = plt.figure(figsize=(6,3.5))
    plt.plot(k_range_db, dbs, marker='o')
    plt.xticks(k_range_db)
    plt.xlabel("k (jumlah cluster)")
    plt.ylabel("Davies-Bouldin index (lebih rendah lebih baik)")
    plt.title("Davies-Bouldin Index per k")
    plt.grid(True)
    if k in k_range_db:
        idx_db = k_range_db.index(k)
        plt.scatter([k], [dbs[idx_db]], s=120, facecolors='none', edgecolors='black', linewidths=2, label=f"Chosen k = {k}")
        plt.legend()
    st.pyplot(fig_db, use_container_width=True)

    st.subheader("📊 Silhouette Scores untuk k = 2..10")
    ks = [k for k, _ in sils]
    scores = [s for _, s in sils]
    fig_sil = plt.figure(figsize=(6,3.5))
    plt.plot(ks, scores, marker='o')
    plt.xticks(ks)
    plt.xlabel("k (jumlah cluster)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score per k")
    plt.grid(True)
    # highlight chosen k
    if k in ks:
        idx = ks.index(k)
        plt.scatter([ks[idx]], [scores[idx]], s=120, facecolors='none', edgecolors='black', linewidths=2, label=f"Chosen k = {k}")
        plt.legend()
    st.pyplot(fig_sil, use_container_width=True)

    with st.spinner("Menjalankan K-Means dan PCA..."):
        try:
            km = KMeans(n_clusters=k, random_state=int(random_state))
            labels = km.fit_predict(X_scaled)
            centers = km.cluster_centers_
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)
            st.success("Selesai.")
        except Exception as e:
            st.error(f"Gagal menjalankan K-Means atau PCA: {e}")
            st.stop()

    out = df.copy()
    out["Cluster"] = labels
    if id_col:
        show_cols = [id_col, "Cluster"] + features
    else:
        show_cols = ["Cluster"] + features

    st.subheader("📄 Hasil Klaster")
    st.dataframe(out[show_cols].sort_values("Cluster").reset_index(drop=True), use_container_width=True)

    st.subheader("🧮 Ringkasan per Cluster (rata-rata fitur)")
    summary = out.groupby("Cluster")[features].mean().reset_index()
    st.dataframe(summary, use_container_width=True)

    # PCA visualization
    st.subheader("📈 Visualisasi 2D dengan PCA")
    fig = plt.figure(figsize=(6,4))
    unique_labels = np.unique(labels)
    for c in unique_labels:
        mask = labels == c
        plt.scatter(pcs[mask,0], pcs[mask,1], label=f"Cluster {c}", alpha=0.8)
    plt.xlabel(f"PC1 (var={pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (var={pca.explained_variance_ratio_[1]:.2f})")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig, use_container_width=True)

    # Download
    st.subheader("💾 Unduh Hasil")
    excel_bytes = to_excel_bytes({
        "Clustered_Data": out,
        "Cluster_Summary": summary
    })
    st.download_button(label="Unduh Excel (data & ringkasan)", data=excel_bytes,
                       file_name="kmeans_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    csv_buf = out.to_csv(index=False).encode("utf-8")
    st.download_button(label="Unduh CSV (data berlabel)", data=csv_buf,
                      file_name="kmeans_labeled.csv", mime="text/csv", use_container_width=True)


if __name__ == "__main__":
    st.write("Modul K-Means siap digunakan dalam aplikasi Streamlit utama.")
