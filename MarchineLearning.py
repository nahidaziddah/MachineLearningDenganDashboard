import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==================================================================
# 0. DATA DEFAULT: Data Bangka Belitung (dummy tetapi realistis)
# ==================================================================
def load_default_data():
    tanggal_range = pd.date_range(start="2015-01-01", end="2024-12-31", freq="D")

    np.random.seed(42)
    df = pd.DataFrame({
        "Tanggal": tanggal_range,
        "Tn": np.random.uniform(22, 25, len(tanggal_range)),
        "Tx": np.random.uniform(30, 34, len(tanggal_range)),
        "Tavg": np.random.uniform(26, 29, len(tanggal_range)),
        "kelembaban": np.random.uniform(75, 95, len(tanggal_range)),
        "curah_hujan": np.random.choice([0, 2, 5, 10, 20, 50, 80], len(tanggal_range)),
        "matahari": np.random.uniform(2, 10, len(tanggal_range)),
        "FF_X": np.random.uniform(1, 5, len(tanggal_range)),
        "DDD_X": np.random.uniform(0, 360, len(tanggal_range))
    })
    return df

# ==================================================================
# STREAMLIT UI
# ==================================================================
st.title("🌤️ Prediksi Iklim Provinsi Bangka Belitung")
st.write("Upload data iklim sendiri atau gunakan **data default Bangka Belitung**.")

# Upload File
uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

# ==================================================================
# LOAD DATA
# ==================================================================
try:
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("✔ Data berhasil dimuat dari file yang Anda upload!")
    else:
        df = load_default_data()
        st.info("📌 Tidak ada file diupload — menggunakan **data default Bangka Belitung**.")
except:
    st.error("❌ Format file tidak sesuai. Pastikan file Excel berisi kolom Tanggal dan variabel cuaca.")
    st.stop()

# ==================================================================
# CEK & PROSES DATA
# ==================================================================
if "Tanggal" not in df.columns:
    st.error("❌ Tidak ditemukan kolom 'Tanggal' pada file yang Anda upload.")
    st.stop()

df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
df = df.dropna(subset=["Tanggal"])

df['Tahun'] = df['Tanggal'].dt.year
df['Bulan'] = df['Tanggal'].dt.month

# Variabel yang digunakan
possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan", "matahari", "FF_X", "DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

if len(available_vars) == 0:
    st.error("❌ Tidak ada variabel cuaca yang valid ditemukan di file Anda.")
    st.stop()

# Label akademis
akademis_label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban Udara (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Penyinaran Matahari (jam)",
    "FF_X": "Kecepatan Angin Maksimum (m/s)",
    "DDD_X": "Arah Angin saat Kecepatan Maksimum (°)"
}

# ==================================================================
# AGREGASI BULANAN
# ==================================================================
agg_dict = {v: 'mean' for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"  # curah hujan dijumlahkan per bulan

cuaca_df = df[['Tahun', 'Bulan'] + available_vars]
monthly_df = cuaca_df.groupby(['Tahun', 'Bulan']).agg(agg_dict).reset_index()

st.subheader("📊 Rekapan Data Bulanan")
st.dataframe(monthly_df)

# ==================================================================
# TRAIN MODEL
# ==================================================================
X = monthly_df[['Tahun', 'Bulan']]
models = {}
metrics = {}

for var in available_vars:
    y = monthly_df[var]

    # Mencegah error jika data terlalu sedikit
    if len(y) < 10:
        st.warning(f"⚠️ Data untuk variabel {var} terlalu sedikit untuk melatih model.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    models[var] = model
    metrics[var] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred)
    }

# ==================================================================
# TAMPILKAN EVALUASI MODEL
# ==================================================================
st.subheader("📈 Evaluasi Model")
for var, m in metrics.items():
    st.write(f"**{akademis_label[var]}** → RMSE: {m['rmse']:.3f} | R²: {m['r2']:.3f}")

# ==================================================================
# PREDIKSI MANUAL
# ==================================================================
st.subheader("🔮 Prediksi Manual (1 Bulan)")
tahun_input = st.number_input("Masukkan Tahun Prediksi", min_value=2025, max_value=2100, value=2035)
bulan_input = st.selectbox("Pilih Bulan", list(range(1, 13)))

input_data = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])

st.write("### Hasil Prediksi:")
for var in available_vars:
    if var in models:
        pred_val = models[var].predict(input_data)[0]
        st.success(f"{akademis_label[var]}: **{pred_val:.2f}**")

# ==================================================================
# PREDIKSI 2025–2075
# ==================================================================
st.subheader("📆 Prediksi Otomatis 2025–2075")
future_years = list(range(2025, 2076))
future_months = list(range(1, 13))

future_data = pd.DataFrame(
    [(year, month) for year in future_years for month in future_months],
    columns=['Tahun', 'Bulan']
)

for var in available_vars:
    if var in models:
        future_data[f"Pred_{var}"] = models[var].predict(future_data[['Tahun', 'Bulan']])

st.dataframe(future_data.head(12))

# ==================================================================
# GRAFIK HISTORIS VS PREDIKSI
# ==================================================================
monthly_df['Sumber'] = 'Data Historis'
future_data['Sumber'] = 'Prediksi'

merge_list = []
for var in available_vars:
    if f"Pred_{var}" not in future_data.columns:
        continue

    hist = monthly_df[['Tahun', 'Bulan', var, 'Sumber']].rename(columns={var: 'Nilai'})
    hist['Variabel'] = akademis_label[var]

    fut = future_data[['Tahun', 'Bulan', f"Pred_{var}", 'Sumber']].rename(columns={f"Pred_{var}": 'Nilai'})
    fut['Variabel'] = akademis_label[var]

    merge_list.append(pd.concat([hist, fut]))

future_data_merged = pd.concat(merge_list)
future_data_merged['Tanggal'] = pd.to_datetime(
    future_data_merged['Tahun'].astype(str) + "-" +
    future_data_merged['Bulan'].astype(str) + "-01"
)

st.subheader("📈 Grafik Tren Variabel Cuaca")
selected_var = st.selectbox("Pilih Variabel Cuaca", [akademis_label[v] for v in available_vars])

fig = px.line(
    future_data_merged[future_data_merged['Variabel'] == selected_var],
    x='Tanggal',
    y='Nilai',
    color='Sumber',
    title=f"Tren {selected_var} Bulanan"
)
st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# DOWNLOAD CSV
# ==================================================================
csv = future_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download CSV Prediksi 2025–2075",
    data=csv,
    file_name='prediksi_iklim_babel_2025_2075.csv',
    mime='text/csv'
)

