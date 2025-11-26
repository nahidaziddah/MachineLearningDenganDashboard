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
        /* Background gradient */
        .stApp {
            background: linear-gradient(180deg, #cce7ff, #e8f6ff);
            color: #00334e;
        }

        /* Titles */
        h1, h2, h3 {
            color: #003a5c !important;
            font-weight: 900 !important;
        }

        /* Card style */
        .big-card {
            padding: 18px;
            border-radius: 18px;
            background: #ffffffaa;
            border-left: 6px solid #0077b6;
            margin-bottom: 20px;
        }

        /* Table styling */
        .dataframe th {
            background-color: #0077b6 !important;
            color: white !important;
        }

        /* Buttons */
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
# 0. DATA DEFAULT
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
    df = pd.read_excel(uploaded_file, sheet_name=0)
    st.success("🌴 Data berhasil dimuat! Menggunakan data asli Anda.")
else:
    df = load_default_data()
    st.info("📌 Tidak ada file diupload — menggunakan **data bawaan Bangka Belitung (default)**.")

# ================================
# PROSES DATA
# ================================
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Tahun'] = df['Tanggal'].dt.year
df['Bulan'] = df['Tanggal'].dt.month

possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan", "matahari", "FF_X", "DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

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
X = monthly_df[['Tahun', 'Bulan']]
models = {}
metrics = {}

for var in available_vars:
    y = monthly_df[var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    pred_val = models[var].predict(input_data)[0]
    st.success(f"{akademis_label[var]}: **{pred_val:.2f}**")

# ================================
# PREDIKSI 2025–2075
# ================================
st.subheader("📆 Prediksi Iklim 2025–2075 (50 Tahun)")

future_data = pd.DataFrame(
    [(y, m) for y in range(2025, 2076) for m in range(1, 13)],
    columns=["Tahun", "Bulan"]
)

for var in available_vars:
    future_data[f"Pred_{var}"] = models[var].predict(future_data[['Tahun', 'Bulan']])

st.dataframe(future_data.head(15))

# ================================
# GRAFIK
# ================================
st.subheader("🌊 Grafik Tren Iklim Bangka Belitung")

monthly_df['Sumber'] = 'Historis'
future_data['Sumber'] = 'Prediksi'

all_data = []

for var in available_vars:
    hist = monthly_df[['Tahun','Bulan',var,'Sumber']].rename(columns={var:'Nilai'})
    hist['Variabel'] = akademis_label[var]

    fut = future_data[['Tahun','Bulan',f"Pred_{var}",'Sumber']].rename(columns={f"Pred_{var}":'Nilai'})
    fut['Variabel'] = akademis_label[var]

    all_data.append(pd.concat([hist, fut]))

merged = pd.concat(all_data)
merged['Tanggal'] = pd.to_datetime(merged['Tahun'].astype(str) + "-" + merged['Bulan'].astype(str) + "-01")

selected_var = st.selectbox("Pilih Variabel Cuaca:", [akademis_label[v] for v in available_vars])

fig = px.line(
    merged[merged["Variabel"] == selected_var],
    x="Tanggal",
    y="Nilai",
    color="Sumber",
    title=f"🌤️ Tren {selected_var} di Bangka Belitung",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# ================================
# DOWNLOAD DATA
# ================================
csv = future_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Prediksi 2025–2075 (CSV)",
    data=csv,
    file_name="prediksi_iklim_babel_2025_2075.csv",
    mime="text/csv"
)
