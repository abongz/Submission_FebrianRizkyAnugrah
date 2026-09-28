from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Bike Sharing Dashboard", page_icon="🚲", layout="wide")

WEATHER = {
    1: "Clear / Few clouds",
    2: "Mist / Cloudy",
    3: "Light Snow / Rain",
    4: "Heavy Rain / Snow / Fog",
}

@st.cache_data

def load_data():
    day = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "day.csv")
    hour = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "hour.csv")
    day["dteday"] = pd.to_datetime(day["dteday"])
    hour["dteday"] = pd.to_datetime(hour["dteday"])
    return day, hour


day, hour = load_data()

st.title("🚲 Bike Sharing Dashboard")
st.caption("Capital Bikeshare — Washington D.C., 2011–2012")
st.info(
    "**Pertanyaan 1:** pola rental menurut bulan, jam, dan working day. "
    "**Pertanyaan 2:** hubungan cuaca/lingkungan dengan rental. "
    "**Analisis lanjutan non-ML:** segmentasi demand dengan binning/manual grouping."
)

with st.sidebar:
    st.header("Filter")
    years = st.multiselect(
        "Tahun", sorted(day.yr.unique()), default=sorted(day.yr.unique()),
        format_func=lambda x: "2011" if x == 0 else "2012"
    )
    months = st.multiselect(
        "Bulan", sorted(day.mnth.unique()), default=sorted(day.mnth.unique())
    )
    weather_options = st.multiselect(
        "Kondisi cuaca", sorted(day.weathersit.unique()),
        default=sorted(day.weathersit.unique()),
        format_func=lambda x: WEATHER[x]
    )

fd = day[
    day.yr.isin(years) & day.mnth.isin(months) & day.weathersit.isin(weather_options)
].copy()
fh = hour[
    hour.yr.isin(years) & hour.mnth.isin(months) & hour.weathersit.isin(weather_options)
].copy()

if fd.empty or fh.empty:
    st.warning("Tidak ada data untuk filter tersebut.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rental", f"{fd.cnt.sum():,.0f}")
c2.metric("Rata-rata/Hari", f"{fd.cnt.mean():,.0f}")
imax = fd.cnt.idxmax()
c3.metric("Rental Tertinggi", f"{fd.loc[imax, 'cnt']:,.0f}")
c4.metric("Jumlah Hari", f"{len(fd):,}")

st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Pola & Cuaca", "📦 Segmentasi Non-ML", "🚨 Analisis Tambahan"])

with tab1:
    st.subheader("📈 Tren Rental Harian")
    trend = fd.groupby("dteday", as_index=False).cnt.sum()
    st.plotly_chart(
        px.line(trend, x="dteday", y="cnt", labels={"dteday": "Tanggal", "cnt": "Jumlah Rental"}),
        use_container_width=True,
    )

    a, b = st.columns(2)
    with a:
        st.subheader("⏰ Pola Rental per Jam")
        hp = fh.groupby("hr", as_index=False).cnt.mean()
        st.plotly_chart(
            px.line(hp, x="hr", y="cnt", markers=True,
                    labels={"hr": "Jam", "cnt": "Rata-rata Rental"}),
            use_container_width=True,
        )
    with b:
        st.subheader("💼 Working Day vs Non-working Day")
        wd = fd.groupby("workingday", as_index=False).cnt.mean()
        wd["kategori"] = wd.workingday.map({0: "Non-working day", 1: "Working day"})
        st.plotly_chart(
            px.bar(wd, x="kategori", y="cnt",
                   labels={"kategori": "", "cnt": "Rata-rata Rental"}),
            use_container_width=True,
        )

    a, b = st.columns(2)
    with a:
        st.subheader("🌦️ Rental berdasarkan Cuaca")
        wp = fd.groupby("weathersit", as_index=False).cnt.mean()
        wp["weather"] = wp.weathersit.map(WEATHER)
        st.plotly_chart(
            px.bar(wp, x="weather", y="cnt",
                   labels={"weather": "Kondisi Cuaca", "cnt": "Rata-rata Rental"}),
            use_container_width=True,
        )
    with b:
        st.subheader("🌡️ Temperatur vs Rental")
        st.plotly_chart(
            px.scatter(fd, x="temp", y="cnt", hover_data=["dteday"],
                       labels={"temp": "Normalized Temperature", "cnt": "Jumlah Rental"}),
            use_container_width=True,
        )
    st.caption("Catatan: temp, atemp, hum, dan windspeed merupakan variabel normalized dari dataset.")

with tab2:
    st.subheader("📦 Segmentasi Tingkat Demand")
    q1, q2, q3 = fd.cnt.quantile([0.25, 0.50, 0.75])
    bins = [-np.inf, q1, q2, q3, np.inf]
    labels = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
    fd["demand_level"] = pd.cut(fd.cnt, bins=bins, labels=labels, include_lowest=True)

    summary = (
        fd.groupby("demand_level", observed=False).cnt
        .agg(Jumlah_Hari="count", Minimum="min", Median="median", Rata_rata="mean", Maksimum="max")
        .reset_index()
    )
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(summary, x="demand_level", y="Rata_rata",
               labels={"demand_level": "Level Demand", "Rata_rata": "Rata-rata Rental"}),
        use_container_width=True,
    )

    st.subheader("⏱️ Segmentasi Kelompok Jam")
    hour_bins = [-1, 5, 11, 17, 23]
    hour_labels = ["Dini hari (00–05)", "Pagi (06–11)", "Siang (12–17)", "Malam (18–23)"]
    fh["hour_group"] = pd.cut(fh.hr, bins=hour_bins, labels=hour_labels)
    seg = (
        fh.groupby(["hour_group", "workingday"], observed=False).cnt
        .agg(Rata_rata="mean", Median="median", Observasi="count")
        .reset_index()
    )
    seg["jenis_hari"] = seg.workingday.map({0: "Non-working day", 1: "Working day"})
    st.plotly_chart(
        px.bar(seg, x="hour_group", y="Rata_rata", color="jenis_hari", barmode="group",
               labels={"hour_group": "Kelompok Jam", "Rata_rata": "Rata-rata Rental", "jenis_hari": "Jenis Hari"}),
        use_container_width=True,
    )
    st.dataframe(seg.round(2), use_container_width=True, hide_index=True)
    st.caption("Segmentasi menggunakan binning/manual grouping dan tidak menggunakan algoritma machine learning.")

with tab3:
    st.subheader("🤖 Performance Regression")

    def prep(df, hourly=False):
        d = df.copy()
        d["date_ordinal"] = (d.dteday - day.dteday.min()).dt.days
        d["month_sin"] = np.sin(2 * np.pi * d.mnth / 12)
        d["month_cos"] = np.cos(2 * np.pi * d.mnth / 12)
        if hourly:
            d["hr_sin"] = np.sin(2 * np.pi * d.hr / 24)
            d["hr_cos"] = np.cos(2 * np.pi * d.hr / 24)
        return d.drop(columns=["dteday", "cnt", "casual", "registered", "instant", "demand_level", "hour_group"], errors="ignore")

    def evaluate(df, hourly=False):
        X = prep(df, hourly)
        y = df.cnt
        cut = int(len(df) * 0.8)
        cats = [c for c in ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"] if c in X]
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)], remainder="passthrough")
        pipe = Pipeline([
            ("pre", pre),
            ("model", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.03, loss="huber", random_state=42)),
        ])
        pipe.fit(X.iloc[:cut], y.iloc[:cut])
        pred = pipe.predict(X.iloc[cut:])
        return mean_absolute_error(y.iloc[cut:], pred), mean_squared_error(y.iloc[cut:], pred) ** .5, r2_score(y.iloc[cut:], pred)

    day_metrics = evaluate(day)
    hour_metrics = evaluate(hour, True)
    metrics = pd.DataFrame([
        {"Dataset": "Harian", "MAE": day_metrics[0], "RMSE": day_metrics[1], "R2": day_metrics[2]},
        {"Dataset": "Per Jam", "MAE": hour_metrics[0], "RMSE": hour_metrics[1], "R2": hour_metrics[2]},
    ]).round(3)
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🚨 Event & Anomaly Detection")
    st.caption("Anomali dicari pada 20% periode terakhir agar residual berasal dari data uji, bukan data yang digunakan untuk melatih model.")
    X = prep(day)
    cut = int(len(day) * .8)
    cats = [c for c in ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"] if c in X]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)], remainder="passthrough")
    model = Pipeline([
        ("pre", pre),
        ("model", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.03, loss="huber", random_state=42)),
    ])
    model.fit(X.iloc[:cut], day.cnt.iloc[:cut])
    an = day.iloc[cut:][["dteday", "cnt", "weathersit", "holiday", "workingday"]].copy()
    an["pred"] = model.predict(X.iloc[cut:])
    an["resid"] = an.cnt - an.pred
    med = an.resid.median()
    mad = np.median(np.abs(an.resid - med))
    an["robust_z"] = .6745 * (an.resid - med) / (mad if mad else 1)
    an["status"] = np.where(an.robust_z.abs() > 3.5, "Anomaly", "Normal")
    st.plotly_chart(
        px.scatter(an, x="dteday", y="cnt", color="status", hover_data=["pred", "resid", "robust_z"],
                   labels={"dteday": "Tanggal", "cnt": "Rental"}),
        use_container_width=True,
    )
    anom = an[an.status == "Anomaly"].sort_values("robust_z").copy()
    anom["weather"] = anom.weathersit.map(WEATHER)
    st.dataframe(
        anom[["dteday", "cnt", "pred", "resid", "robust_z", "weather", "holiday", "workingday"]]
        .round({"cnt": 0, "pred": 0, "resid": 0, "robust_z": 2}).head(20),
        use_container_width=True, hide_index=True,
    )
    st.info("Ambang anomali: |robust z| > 3,5. Kandidat anomali perlu divalidasi dengan informasi eksternal sebelum penyebab/event ditetapkan.")
    st.caption("`casual` dan `registered` tidak digunakan sebagai prediktor karena merupakan komponen langsung dari `cnt`.")
