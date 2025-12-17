import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
from datetime import datetime
import joblib # type: ignore

st.set_page_config(
    page_title="Sistem Pemantauan Kualitas Udara - SPKU",
    layout="wide"
)

@st.cache_resource
def load_model():
        return joblib.load("kualitas_model.joblib")
   

@st.cache_resource
def load_encoder():
        return joblib.load("kualitas_encoder.joblib")


@st.cache_data
def load_dataset():
        df = pd.read_csv("SPKU_final.csv")
        
        df.replace('---', pd.NA, inplace=True)
        df = df.replace({pd.NA: np.nan})
        
        num_cols = ['pm10', 'so2', 'co', 'o3', 'no2', 'max']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        return df

model = load_model()
encoder = load_encoder()
df = load_dataset()

if model is None or encoder is None or df is None:
    st.stop()

def get_category_info(category):
    info = {
        'BAIK': {
            'icon': '✅',
            'color': 'green',
            'desc': 'Kualitas udara sangat baik, tidak memberikan efek negatif.',
            'action': 'Sangat baik untuk aktivitas luar ruangan.'
        },
        'SEDANG': {
            'icon': '⚠️',
            'color': 'orange',
            'desc': 'Kualitas udara masih dapat diterima.',
            'action': 'Kelompok sensitif sebaiknya mengurangi aktivitas luar ruangan yang berkepanjangan.'
        },
        'TIDAK SEHAT': {
            'icon': '🚨',
            'color': 'red',
            'desc': 'Kualitas udara bersifat merugikan pada manusia, hewan, dan tumbuhan.',
            'action': 'Kelompok sensitif perlu menghindari aktivitas luar ruangan.'
        },
        'SANGAT TIDAK SEHAT': {
            'icon': '☠️',
            'color': 'red',
            'desc': 'Kualitas udara sangat berbahaya bagi kesehatan.',
            'action': 'Semua orang sebaiknya menghindari aktivitas luar ruangan.'
        }
    }
    return info.get(category, info['SEDANG'])

def predict_air_quality(pm10, so2, co, o3, no2):
    X_new = np.array([[pm10, so2, co, o3, no2]])
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]
    category = encoder.inverse_transform([prediction])[0]
    
    max_value = max(pm10, so2, co, o3, no2)
    pollutants = ['PM10', 'SO2', 'CO', 'O3', 'NO2']
    values = [pm10, so2, co, o3, no2]
    critical_pollutant = pollutants[values.index(max_value)]
    
    return category, probabilities, prediction, max_value, critical_pollutant

st.title(" Sistem Pemantauan & Prediksi Kualitas Udara")
st.markdown("Stasiun Pemantauan Kualitas Udara (SPKU) DKI Jakarta")
st.markdown("---")

with st.sidebar:
    
    menu = st.radio(
        "Pilih Menu:",
        [
            "🏠 Dashboard",
            "🔮 Prediksi Kualitas Udara",
            "📊 Analisis Stasiun",
            "📈 Visualisasi Data",
            "📋 Data Explorer",
            "ℹ️ Info Model"
        ]
    )
    
    st.markdown("---")
    
    st.subheader("📊 Statistik Dataset")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Jumlah Stasiun", df['stasiun'].nunique())
    st.metric("Periode Data", df['periode_data'].nunique())
    

if menu == "🏠 Dashboard":
    st.header(" Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'categori' not in df.columns:
        st.error("Kolom 'categori' tidak ditemukan dalam dataset!")
        st.stop()
    
    category_counts = df['categori'].value_counts()
    total_records = len(df)
    
    with col1:
        baik = category_counts.get('BAIK', 0)
        st.metric("📗 BAIK", f"{baik:,}", delta=f"{baik/total_records*100:.1f}%")
    
    with col2:
        sedang = category_counts.get('SEDANG', 0)
        st.metric("📙 SEDANG", f"{sedang:,}", delta=f"{sedang/total_records*100:.1f}%")
    
    with col3:
        tidak_sehat = category_counts.get('TIDAK SEHAT', 0)
        st.metric("📕 TIDAK SEHAT", f"{tidak_sehat:,}", delta=f"{tidak_sehat/total_records*100:.1f}%", delta_color="inverse")
    
    with col4:
        sangat = category_counts.get('SANGAT TIDAK SEHAT', 0)
        st.metric("📕 SANGAT TIDAK SEHAT", f"{sangat:,}", delta=f"{sangat/total_records*100:.1f}%", delta_color="inverse")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Kategori Kualitas Udara")
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color=category_counts.index,
            color_discrete_map={
                'BAIK': '#10b981',
                'SEDANG': '#fbbf24',
                'TIDAK SEHAT': '#f97316',
                'SANGAT TIDAK SEHAT': '#ef4444'
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Grafik Stasiun Terburuk")
        bad_stations = df[df['categori'].isin(['TIDAK SEHAT', 'SANGAT TIDAK SEHAT'])]
        station_counts = bad_stations['stasiun'].value_counts().head(10)
        
        fig = px.bar(
            x=station_counts.values,
            y=station_counts.index,
            orientation='h',
            labels={'x': 'Jumlah Kejadian', 'y': 'Stasiun'},
            color=station_counts.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Polutan Kritis Dominan")
        critical_counts = df['critical'].value_counts()
        
        fig = px.bar(
            x=critical_counts.index,
            y=critical_counts.values,
            labels={'x': 'Polutan', 'y': 'Frekuensi'},
            color=critical_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Rata-rata Konsentrasi Polutan")
        pollutants = ['pm10', 'so2', 'co', 'o3', 'no2']
        avg_values = df[pollutants].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pollutants,
            y=avg_values.values,
            marker_color=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444'],
            text=[f'{v:.2f}' for v in avg_values.values],
            textposition='outside'
        ))
        fig.update_layout(xaxis_title="Polutan", yaxis_title="Konsentrasi (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(" Korelasi Antar Polutan")
    corr_matrix = df[pollutants].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "🔮 Prediksi Kualitas Udara":
    st.header(" Simulasi Prediksi Kualitas Udara")    
    st.subheader(" Input Data Polutan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(" Particulate Matter & Gas")
        pm10 = st.number_input(
            "PM10 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state.get('pm10', 45.0),
            step=1.0
        )
        
        so2 = st.number_input(
            "SO2 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state.get('so2', 32.0),
            step=1.0
        )
        
        co = st.number_input(
            "CO (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state.get('co', 11.0),
            step=1.0
        )
    
    with col2:
        st.markdown(" Ozon & Nitrogen")
        o3 = st.number_input(
            "O3 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state.get('o3', 37.0),
            step=1.0
        )
        
        no2 = st.number_input(
            "NO2 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=st.session_state.get('no2', 7.0),
            step=1.0
        )
    
    st.markdown("---")

    if st.button(" PREDIKSI KUALITAS UDARA", type="primary", use_container_width=True):
        with st.spinner(" Memproses prediksi..."):
            category, probabilities, prediction, max_value, critical_pollutant = predict_air_quality(
                pm10, so2, co, o3, no2
            )
            
            st.success("Prediksi berhasil!")
            
            st.markdown("---")

            st.subheader("HASIL PREDIKSI")
            
            info = get_category_info(category)
            
            if info['color'] == 'green':
                st.success(f"{info['icon']} **KATEGORI: {category}**")
            elif info['color'] == 'orange':
                st.warning(f"{info['icon']} **KATEGORI: {category}**")
            else:
                st.error(f"{info['icon']} **KATEGORI: {category}**")
         
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nilai Maksimum", f"{max_value:.2f} µg/m³")
            
            with col2:
                st.metric("Polutan Kritis", critical_pollutant)

elif menu == "📊 Analisis Stasiun":
    st.header(" Analisis per Stasiun")
    
    selected_station = st.selectbox(
        " Pilih Stasiun:",
        options=sorted(df['stasiun'].unique())
    )
    
    station_data = df[df['stasiun'] == selected_station]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(station_data))
    
    with col2:
        dominant = station_data['categori'].mode()[0]
        st.metric("Kategori Dominan", dominant)
    
    with col3:
        critical = station_data['critical'].mode()[0]
        st.metric("Polutan Kritis", critical)
    
    with col4:
        avg_max = station_data['max'].mean()
        st.metric("Rata-rata Max", f"{avg_max:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Distribusi Kategori")
        cat_dist = station_data['categori'].value_counts()
        
        fig = px.pie(
            values=cat_dist.values,
            names=cat_dist.index,
            color=cat_dist.index,
            color_discrete_map={
                'BAIK': '#10b981',
                'SEDANG': '#fbbf24',
                'TIDAK SEHAT': '#f97316',
                'SANGAT TIDAK SEHAT': '#ef4444'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Polutan Kritis")
        crit_dist = station_data['critical'].value_counts()
        
        fig = px.bar(
            x=crit_dist.index,
            y=crit_dist.values,
            labels={'x': 'Polutan', 'y': 'Frekuensi'},
            color=crit_dist.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(" Distribusi Konsentrasi Polutan")
    
    pollutants = ['pm10', 'so2', 'co', 'o3', 'no2']
    
    fig = go.Figure()
    for pollutant in pollutants:
        fig.add_trace(go.Box(
            y=station_data[pollutant],
            name=pollutant.upper()
        ))
    
    fig.update_layout(yaxis_title="Konsentrasi (µg/m³)")
    st.plotly_chart(fig, use_container_width=True)
    
    if 'tanggal' in station_data.columns:
        st.subheader(" Tren Waktu")
        station_data['tanggal'] = pd.to_datetime(station_data['tanggal'])
        station_sorted = station_data.sort_values('tanggal')
        
        fig = px.line(
            station_sorted,
            x='tanggal',
            y='max',
            color='categori',
            markers=True,
            color_discrete_map={
                'BAIK': '#10b981',
                'SEDANG': '#fbbf24',
                'TIDAK SEHAT': '#f97316',
                'SANGAT TIDAK SEHAT': '#ef4444'
            }
        )
        fig.update_layout(xaxis_title="Tanggal", yaxis_title="Nilai Max (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

elif menu == "📈 Visualisasi Data":
    st.header(" Visualisasi Data")
    
    viz_type = st.selectbox(
        "Pilih Jenis Visualisasi:",
        [
            "3D Scatter Plot",
            "Heatmap Kategori per Stasiun",
            "Violin Plot",
            "Sunburst Chart",
            "Time Series Multi-Polutan"
        ]
    )
    
    if viz_type == "3D Scatter Plot":
        st.subheader(" 3D Scatter: PM10 vs SO2 vs CO")
        
        sample_size = min(500, len(df))
        sample_df = df.sample(sample_size)
        
        fig = px.scatter_3d(
            sample_df,
            x='pm10',
            y='so2',
            z='co',
            color='categori',
            color_discrete_map={
                'BAIK': '#10b981',
                'SEDANG': '#fbbf24',
                'TIDAK SEHAT': '#f97316',
                'SANGAT TIDAK SEHAT': '#ef4444'
            },
            size='max',
            hover_data=['stasiun', 'critical']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Heatmap Kategori per Stasiun":
        st.subheader(" Heatmap: Kategori per Stasiun")
        
        pivot = df.groupby(['stasiun', 'categori']).size().unstack(fill_value=0)
        
        fig = px.imshow(
            pivot,
            labels=dict(x="Kategori", y="Stasiun", color="Frekuensi"),
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Violin Plot":
        st.subheader(" Violin Plot")
        
        pollutant = st.selectbox("Pilih Polutan:", ['pm10', 'so2', 'co', 'o3', 'no2'])
        
        fig = px.violin(
            df,
            x='categori',
            y=pollutant,
            color='categori',
            color_discrete_map={
                'BAIK': '#10b981',
                'SEDANG': '#fbbf24',
                'TIDAK SEHAT': '#f97316',
                'SANGAT TIDAK SEHAT': '#ef4444'
            },
            box=True,
            points='all'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Sunburst Chart":
        st.subheader(" Sunburst: Hierarki Data")
        
        sample = df.groupby(['categori', 'stasiun', 'critical']).size().reset_index(name='count')
        
        fig = px.sunburst(
            sample,
            path=['categori', 'stasiun', 'critical'],
            values='count',
            color='categori',
            color_discrete_map={
                'BAIK': '#10b981',
                'SEDANG': '#fbbf24',
                'TIDAK SEHAT': '#f97316',
                'SANGAT TIDAK SEHAT': '#ef4444'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Time Series Multi-Polutan":
        st.subheader(" Time Series: Semua Polutan")
        
        if 'tanggal' in df.columns:
            df['tanggal'] = pd.to_datetime(df['tanggal'])
            time_data = df.groupby('tanggal')[['pm10', 'so2', 'co', 'o3', 'no2']].mean().reset_index()
            
            fig = make_subplots(
                rows=5, cols=1,
                subplot_titles=('PM10', 'SO2', 'CO', 'O3', 'NO2')
            )
            
            pollutants = ['pm10', 'so2', 'co', 'o3', 'no2']
            colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
            
            for i, (pol, color) in enumerate(zip(pollutants, colors), 1):
                fig.add_trace(
                    go.Scatter(x=time_data['tanggal'], y=time_data[pol], name=pol.upper(), line=dict(color=color)),
                    row=i, col=1
                )
            
            fig.update_layout(height=1000, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Data tanggal tidak tersedia")

elif menu == "📋 Data Explorer":
    st.header(" Data Explorer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stations = st.multiselect("Stasiun:", sorted(df['stasiun'].unique()))
    
    with col2:
        categories = st.multiselect("Kategori:", sorted(df['categori'].unique()))
    
    with col3:
        criticals = st.multiselect("Polutan Kritis:", sorted(df['critical'].unique()))
    
    filtered = df.copy()
    
    if stations:
        filtered = filtered[filtered['stasiun'].isin(stations)]
    if categories:
        filtered = filtered[filtered['categori'].isin(categories)]
    if criticals:
        filtered = filtered[filtered['critical'].isin(criticals)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Filtered Records", f"{len(filtered):,}")
    with col2:
        st.metric("Total Records", f"{len(df):,}")
    with col3:
        pct = (len(filtered) / len(df) * 100) if len(df) > 0 else 0
        st.metric("Percentage", f"{pct:.1f}%")
    
    st.markdown("---")
    
    search = st.text_input("🔎 Search dalam data:")
    
    if search:
        mask = filtered.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        filtered = filtered[mask]
   
    st.subheader(f" Data Table ({len(filtered):,} records)")
    st.dataframe(filtered, use_container_width=True, height=500)
    
    with st.expander(" Statistik Data"):
        st.dataframe(filtered[['pm10', 'so2', 'co', 'o3', 'no2']].describe())
    
    col1, col2 = st.columns(2)
    
elif menu == "ℹ️ Info Model":
    st.header("ℹ️ Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Detail Model")
        st.info("""
        **Algoritma:** Random Forest Classifier
        
        **Hyperparameter:**
        - N Estimators: 100
        - Criterion: Gini
        - Random State: 42
        
        **Input Features:**
        - PM10 (Particulate Matter ≤10µm)
        - SO2 (Sulfur Dioksida)
        - CO (Karbon Monoksida)
        - O3 (Ozon)
        - NO2 (Nitrogen Dioksida)
        
        **Output:**
        - Kategori Kualitas Udara (4 kelas)
        """)
        
        st.subheader(" Label Encoding")
        
        label_map = {}
        for cls in encoder.classes_:
            encoded = encoder.transform([cls])[0]
            label_map[cls] = encoded
        
        label_df = pd.DataFrame(list(label_map.items()), columns=['Kategori', 'Encoded'])
        st.dataframe(label_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader(" Informasi Dataset")
        
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Jumlah Fitur", 5)
        st.metric("Jumlah Kelas", len(encoder.classes_))
        st.metric("Jumlah Stasiun", df['stasiun'].nunique())
        st.metric("Periode Data", df['periode_data'].nunique())
        
        st.subheader(" Distribusi Kelas")
        
        class_dist = df['categori'].value_counts()
        
        fig = px.bar(
            x=class_dist.index,
            y=class_dist.values,
            labels={'x': 'Kategori', 'y': 'Jumlah'},
            color=class_dist.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Preprocessing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Data Cleaning:**
        - Replace '---' dengan NaN
        - Numeric conversion
        - Median imputation
        """)
    
    with col2:
        st.info("""
        **Encoding:**
        - Label Encoding untuk target
        - Kategori → 0, 1, 2, 3
        - Reversible transformation
        """)
    
    with col3:
        st.info("""
        **Train-Test Split:**
        - Training: 80%
        - Testing: 20%
        - Stratified sampling
        """)
    
    st.subheader(" Tentang Dataset")
    
    st.success("""
    **Standar Pencemar Kualitas Udara (SPKU) DKI Jakarta**
    
    Dataset ini berisi hasil monitoring kualitas udara di berbagai stasiun pemantauan 
    di wilayah DKI Jakarta. Parameter yang diukur meliputi PM10, SO2, CO, O3, dan NO2.
    
    **Kategori Kualitas Udara:**
    
     **BAIK (0-50):** Kualitas udara sangat baik, tidak ada dampak kesehatan
    
     **SEDANG (51-100):** Kualitas udara masih dapat diterima
    
     **TIDAK SEHAT (101-199):** Kelompok sensitif mungkin mengalami efek kesehatan
    
     **SANGAT TIDAK SEHAT (200+):** Semua orang dapat mengalami efek kesehatan serius
    """)
    
st.markdown("---")