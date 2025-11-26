import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# 🌊 CUSTOM THEME — OCEAN BLUE
# ================================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #cce7ff, #e8f6ff);
            color: #00334e;
        }
        h1, h2, h3 {
            color: #003a5c !important;
            font-weight: 900 !important;
        }
        .big-card {
            padding: 18px;
            border-radius: 18px;
            background: #ffffffaa;
            border-left: 6px solid #0077b6;
            margin-bottom: 20px;
        }
        .dataframe th {
            background-color: #0077b6 !important;
            color: white !important;
        }
        .stButton button {
            background-color: #0096c7;
            color: white;
            border-radius: 10px;
            padding: 8px 20px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #00b4d8;
            color: #003049;
        }
    </style>
""", unsafe_allow_html=True)

# ================================
# DATA DEFAULT
# ================================
def load_default_data():
    tanggal_range = pd.date_range(start="2018-01-01", end="2022-12-31", freq="D")
    np.random.seed(42)
    df = pd.DataFrame({
        "Tanggal": tanggal_range,
        "Tn": np.random.uniform(20, 25, len(tanggal_range)),
        "Tx": np.random.uniform(28, 34, len(tanggal_range)),
        "Tavg": np.random.uniform(24, 29, len(tanggal_range)),
        "kelembaban": np.random.uniform(60, 95, len(tanggal_range)),
        "curah_hujan": np.random.choice([0, 2, 5, 10, 20, 50], len(tanggal_range)),
        "matahari": np.random.uniform(1, 10, len(tanggal_range)),
        "FF_X": np.random.uniform(1, 6, len(tanggal_range)),
        "DDD_X": np.random.uniform(0, 360, len(tanggal_range)),
    })
    return df

# ================================
# HEADER
# ================================
st.title("🌊 Iklim Bangka Belitung — Prediksi Modern dengan Machine Learning")
st.markdown("""
### Selamat datang di Dashboard Prediksi Iklim Bangka Belitung!  
Pantau dan prediksi perubahan iklim hingga tahun 2075 dengan tampilan estetik biru laut 🐚🌤️  
""")

# ================================
# UPLOAD DATA
# ================================
st.subheader("📁 Upload Data Cuaca Anda")
uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        st.success("🌴 Data berhasil dimuat! Menggunakan data asli Anda.")
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        st.stop()
else:
    df = load_default_data()
    st.info("📌 Tidak ada file diupload — menggunakan data cuaca Bangka Belitung (default).")

# ================================
# PROSES DATA
# ================================
if "Tanggal" not in df.columns:
    st.error("Kolom 'Tanggal' tidak ditemukan. Pastikan file memiliki kolom 'Tanggal'.")
    st.stop()

df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
df = df.dropna(subset=['Tanggal']).copy()
df['Tahun'] = df['Tanggal'].dt.year
df['Bulan'] = df['Tanggal'].dt.month

possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan", "matahari", "FF_X", "DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

if len(available_vars) == 0:
    st.error("Tidak ada variabel cuaca valid (Tn/Tx/Tavg/kelembaban/curah_hujan/matahari/FF_X/DDD_X).")
    st.stop()

akademis_label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban Udara (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Penyinaran Matahari (jam)",
    "FF_X": "Kecepatan Angin Maksimum (m/s)",
    "DDD_X": "Arah Angin Kecepatan Maksimum (°)"
}

agg_dict = {v: 'mean' for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

monthly_df = df.groupby(['Tahun', 'Bulan']).agg(agg_dict).reset_index()

st.subheader("📊 Rekapan Data Bulanan")
st.dataframe(monthly_df)

# ================================
# TRAIN MODEL
# ================================
# Pastikan ada cukup data untuk melatih
if monthly_df.shape[0] < 12:
    st.warning("Data bulanan terlalu sedikit untuk pelatihan yang andal (butuh >= 12 baris).")
models = {}
metrics = {}

for var in available_vars:
    y = monthly_df[var]
    # Skip jika terlalu sedikit
    if len(y) < 10:
        st.warning(f"Data untuk variabel {var} terlalu sedikit, dilewati pelatihan.")
        continue
    X_train, X_test, y_train, y_test = train_test_split(
        monthly_df[['Tahun', 'Bulan']], y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    models[var] = model
    metrics[var] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred)
    }

# ================================
# EVALUASI
# ================================
st.subheader("📈 Evaluasi Model")
if len(metrics) == 0:
    st.info("Tidak ada model terlatih (mungkin data terlalu sedikit).")
else:
    for var, m in metrics.items():
        st.markdown(f"""
        <div class="big-card">
            <b>{akademis_label[var]}</b><br>
            RMSE: {m['rmse']:.3f} — R²: {m['r2']:.3f}
        </div>
        """, unsafe_allow_html=True)

# ================================
# PREDIKSI MANUAL
# ================================
st.subheader("🔮 Prediksi Manual 1 Bulan")
tahun_input = st.number_input("Tahun Prediksi:", min_value=2025, max_value=2100, value=2035)
bulan_input = st.selectbox("Bulan Prediksi:", list(range(1,13)))

input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])

st.write("### 🌤️ Hasil Prediksi:")
for var in available_vars:
    if var in models:
        pred_val = models[var].predict(input_data)[0]
        st.success(f"{akademis_label[var]}: **{pred_val:.2f}**")
    else:
        st.info(f"Model untuk {akademis_label[var]} tidak tersedia (data kurang).")

# ================================
# PREDIKSI 2025–2075
# ================================
st.subheader("📆 Prediksi Iklim 2025–2075 (50 Tahun)")

future_data = pd.DataFrame(
    [(y, m) for y in range(2025, 2076) for m in range(1, 13)],
    columns=["Tahun", "Bulan"]
)

for var in available_vars:
    if var in models:
        future_data[f"Pred_{var}"] = models[var].predict(future_data[['Tahun', 'Bulan']])
    else:
        # isi NaN jika model tidak ada
        future_data[f"Pred_{var}"] = np.nan

st.dataframe(future_data.head(15))

# ================================
# GRAFIK TREN UTAMA
# ================================
st.subheader("🌊 Grafik Tren Iklim Bangka Belitung")

monthly_df['Sumber'] = 'Historis'
future_data['Sumber'] = 'Prediksi'

all_data = []
for var in available_vars:
    # historis
    hist = monthly_df[['Tahun','Bulan',var,'Sumber']].rename(columns={var:'Nilai'}).copy()
    hist['Variabel'] = akademis_label[var]

    # prediksi: pastikan kolom Pred_{var} ada
    pred_col = f"Pred_{var}"
    if pred_col in future_data.columns:
        fut = future_data[['Tahun','Bulan', pred_col, 'Sumber']].rename(columns={pred_col:'Nilai'}).copy()
        fut['Variabel'] = akademis_label[var]
        all_data.append(pd.concat([hist, fut]))
    else:
        all_data.append(hist)

if len(all_data) == 0:
    st.warning("Data untuk grafik tidak tersedia.")
else:
    merged = pd.concat(all_data, ignore_index=True)
    merged['Tanggal'] = pd.to_datetime(merged['Tahun'].astype(str) + "-" + merged['Bulan'].astype(str) + "-01")

    selected_var = st.selectbox("Pilih Variabel Cuaca:", [akademis_label[v] for v in available_vars])

    # konversi label ke key
    var_key = [k for k,v in akademis_label.items() if v == selected_var][0]

    fig = px.line(
        merged[merged["Variabel"] == akademis_label[var_key]],
        x="Tanggal",
        y="Nilai",
        color="Sumber",
        title=f"🌤️ Tren {selected_var} di Bangka Belitung",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================================
    # 📊 GRAFIK TAHUNAN — TAMBAHAN BARU
    # ================================
    st.subheader("📘 Grafik Ringkasan Tahunan")

    # annual historis
    annual_hist = monthly_df.groupby("Tahun")[available_vars].mean().reset_index()
    annual_hist["Sumber"] = "Historis"

    # annual prediksi (gunakan Pred_ kolom jika tersedia)
    pred_cols = [f"Pred_{v}" for v in available_vars if f"Pred_{v}" in future_data.columns]
    if len(pred_cols) > 0:
        annual_pred = future_data.groupby("Tahun")[pred_cols].mean().reset_index()
        # rename Pred_* -> original var names for plotting
        rename_map = {f"Pred_{v}": v for v in available_vars if f"Pred_{v}" in future_data.columns}
        annual_pred = annual_pred.rename(columns=rename_map)
        annual_pred["Sumber"] = "Prediksi"
        annual_all = pd.concat([annual_hist, annual_pred], ignore_index=True, sort=False)
    else:
        annual_all = annual_hist.copy()

    selected_var2 = st.selectbox("Pilih Variabel Tahunan:", [akademis_label[v] for v in available_vars], key="annual_var")
    var_key2 = [k for k,v in akademis_label.items() if v == selected_var2][0]

    if var_key2 in annual_all.columns:
        fig2 = px.line(
            annual_all,
            x="Tahun",
            y=var_key2,
            color="Sumber",
            title=f"📘 Tren Tahunan {selected_var2} (Rata-rata per Tahun)",
            markers=True
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Data tahunan prediksi tidak lengkap untuk variabel ini.")

# ================================
# 🧠 RINGKASAN ANALISIS
# ================================
st.subheader("🧠 Ringkasan Perubahan Iklim 2025–2075")

summary_text = ""
for var in available_vars:
    pred_col = f"Pred_{var}"
    if pred_col in future_data.columns:
        # ambil nilai pertama dan terakhir yang bukan NaN
        valid_vals = future_data[pred_col].dropna().values
        if valid_vals.size == 0:
            continue
        awal = valid_vals[0]
        akhir = valid_vals[-1]
        perubahan = akhir - awal
        tren = "meningkat" if perubahan > 0 else "menurun" if perubahan < 0 else "stabil"
        summary_text += f"- **{akademis_label[var]}** diprediksi **{tren}** sebesar **{perubahan:.2f}** dalam 50 tahun.<br>"
    else:
        summary_text += f"- **{akademis_label[var]}**: prediksi tidak tersedia (model tidak terlatih).<br>"

if summary_text == "":
    st.info("Ringkasan tidak tersedia (tidak ada prediksi).")
else:
    st.markdown(f"""
    <div class="big-card">
    <b>✨ Ringkasan Tren 50 Tahun</b><br><br>
    {summary_text}
    </div>
    """, unsafe_allow_html=True)

# ================================
# 🎯 KEGUNAAN DASHBOARD
# ================================
st.subheader("🎯 Kegunaan Dashboard Prediksi Iklim")

st.markdown("""
<div class="big-card">
<b>🔹 1. Monitoring Iklim</b><br>
Mengamati tren iklim historis dan membandingkannya dengan prediksi hingga tahun 2075.<br><br>

<b>🔹 2. Perencanaan Kebijakan</b><br>
Membantu pemerintah daerah merancang mitigasi dan adaptasi perubahan iklim.<br><br>

<b>🔹 3. Analisis Risiko Cuaca Ekstrem</b><br>
Melihat potensi kenaikan suhu, curah hujan, dan perubahan atmosfer lainnya.<br><br>

<b>🔹 4. Edukasi dan Penelitian</b><br>
Membantu mahasiswa, guru, dan peneliti memahami perubahan iklim dengan visual interaktif.<br><br>

<b>🔹 5. Machine Learning Insight</b><br>
Menggunakan Random Forest untuk menghasilkan prediksi berbasis data ilmiah.
</div>
""", unsafe_allow_html=True)

# ================================
# DOWNLOAD FILE
# ================================
csv = future_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Prediksi 2025–2075 (CSV)",
    data=csv,
    file_name="prediksi_iklim_babel_2025_2075.csv",
    mime="text/csv"
)
