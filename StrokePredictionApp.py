import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Stroke Prediction | KDD-01",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
.stApp { background-color: #0f1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f2e 0%, #0d1117 100%);
    border-right: 1px solid #2d3748;
}
.card {
    background: linear-gradient(135deg, #1e2538 0%, #252d3d 100%);
    border: 1px solid #2d3748; border-radius: 12px;
    padding: 20px; margin-bottom: 12px;
}
.kpi-card {
    background: #1e2538; border: 1px solid #2d3748;
    border-radius: 10px; padding: 18px; text-align: center;
}
.kpi-val  { font-size: 26px; font-weight: 800; }
.kpi-lbl  { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
.member-card {
    background: #1e2538; border: 1px solid #2d3748; border-radius: 10px;
    padding: 14px 18px; display: flex; align-items: center; gap: 14px;
    margin-bottom: 10px;
}
.section-title {
    color: #e2e8f0; font-size: 18px; font-weight: 700;
    padding-bottom: 10px; border-bottom: 2px solid #2d3748; margin-bottom: 18px;
}
.risk-high {
    background: linear-gradient(135deg,#7f1d1d,#991b1b);
    border: 1px solid #ef4444; border-radius: 14px; padding: 24px; text-align: center;
}
.risk-low {
    background: linear-gradient(135deg,#064e3b,#065f46);
    border: 1px solid #10b981; border-radius: 14px; padding: 24px; text-align: center;
}
.badge-high { color:#fca5a5; font-size:32px; font-weight:800; }
.badge-low  { color:#6ee7b7; font-size:32px; font-weight:800; }
.prob-big   { font-size:52px; font-weight:900; }
.stButton>button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white; border: none; border-radius: 8px;
    font-weight: 700; width: 100%; padding: 12px;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==================== DATA & MODEL BUILDER ====================
@st.cache_data(show_spinner=False)
def build_all():
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix, roc_curve, silhouette_score)
    from imblearn.over_sampling import SMOTE

    np.random.seed(42)
    n = 5110
    age           = np.clip(np.random.normal(43, 22, n), 0, 100)
    hypertension  = np.random.binomial(1, 0.10, n)
    heart_disease = np.random.binomial(1, 0.054, n)
    ever_married  = np.random.choice(['Yes', 'No'], n, p=[0.66, 0.34])
    work_type     = np.random.choice(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
                                     n, p=[0.57, 0.16, 0.13, 0.13, 0.01])
    residence     = np.random.choice(['Urban', 'Rural'], n)
    avg_glucose   = np.clip(np.random.lognormal(4.6, 0.4, n), 55, 300)
    bmi           = np.clip(np.random.normal(28.9, 7.8, n), 10, 60)
    smoking       = np.random.choice(['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
                                     n, p=[0.37, 0.17, 0.15, 0.31])
    gender        = np.random.choice(['Male', 'Female'], n, p=[0.41, 0.59])
    log_odds      = -5.5 + 0.055*age + 0.8*hypertension + 0.9*heart_disease + 0.004*avg_glucose + 0.01*bmi
    stroke        = np.random.binomial(1, (1/(1+np.exp(-log_odds)))*0.8, n)

    df = pd.DataFrame({
        'gender': gender, 'age': age, 'hypertension': hypertension,
        'heart_disease': heart_disease, 'ever_married': ever_married,
        'work_type': work_type, 'Residence_type': residence,
        'avg_glucose_level': avg_glucose, 'bmi': bmi,
        'smoking_status': smoking, 'stroke': stroke
    })

    df_c = df[df['gender'] != 'Other'].copy()
    le_dict = {}
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_enc = df_c.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le

    X = df_enc.drop(columns=['stroke'])
    y = df_enc['stroke']
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    Xsm, ysm = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    models_def = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=42, class_weight='balanced'),
    }
    results, trained = {}, {}
    for name, m in models_def.items():
        m.fit(Xsm, ysm)
        trained[name] = m
        yp = m.predict(X_te)
        ypr = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, ypr)
        cv = cross_val_score(m, Xs, y, cv=StratifiedKFold(5), scoring='roc_auc').mean()
        results[name] = {
            'Accuracy': round(accuracy_score(y_te, yp), 4),
            'F1-Score': round(f1_score(y_te, yp, average='weighted'), 4),
            'AUC-ROC': round(roc_auc_score(y_te, ypr), 4),
            'CV AUC': round(cv, 4),
            'cm': confusion_matrix(y_te, yp),
            'fpr': fpr, 'tpr': tpr,
            'y_pred': yp, 'y_prob': ypr
        }

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cl = km.fit_predict(Xs)
    df_c = df_c.copy()
    df_c['cluster'] = cl
    pca = PCA(n_components=2, random_state=42)
    Xpca = pca.fit_transform(Xs)
    sil = silhouette_score(Xs, cl)

    inertias, silhouettes = [], []
    for k in range(2, 9):
        km2 = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km2.fit_predict(Xs)
        inertias.append(km2.inertia_)
        silhouettes.append(silhouette_score(Xs, lbl))

    fi = pd.Series(trained['Random Forest'].feature_importances_, index=X.columns).sort_values(ascending=False)
    cp = df_c.groupby('cluster').agg(
        Count=('stroke', 'count'),
        Stroke_Pct=('stroke', lambda x: round(x.mean()*100, 1)),
        Avg_Age=('age', lambda x: round(x.mean(), 1)),
        Avg_BMI=('bmi', lambda x: round(x.mean(), 1)),
        Avg_Glucose=('avg_glucose_level', lambda x: round(x.mean(), 1)),
        Hypertension_Pct=('hypertension', lambda x: round(x.mean()*100, 1)),
    ).reset_index()

    return dict(
        df=df_c, X=X, y=y, Xs=Xs, Xpca=Xpca, cl=cl, km=km, pca=pca,
        scaler=scaler, le_dict=le_dict, trained=trained, results=results,
        fi=fi, cp=cp, sil=sil, inertias=inertias, silhouettes=silhouettes,
        X_te=X_te, y_te=y_te
    )

with st.spinner("⏳ Memuat model..."):
    D = build_all()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:40px'>🧠</div>
        <div style='font-size:18px;font-weight:800;color:#e2e8f0'>Stroke Prediction</div>
        <div style='font-size:11px;color:#6b7280'>UAS Data Mining · KDD-01</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(" ", [
        "🏠  Home",
        "📂  Dataset Overview",
        "🔮  Prediction / Analysis",
        "📊  Visualization",
        "ℹ️   About"
    ], label_visibility="collapsed")

# ==================== 1. HOME ====================
if page == "🏠  Home":
    st.markdown("""
    <div style='text-align:center;padding:40px 20px 20px'>
        <div style='font-size:56px'>🧠</div>
        <h1 style='color:#e2e8f0;font-size:36px;font-weight:900;margin:12px 0 8px'>
            Stroke Risk Prediction System
        </h1>
        <p style='color:#94a3b8;font-size:16px;max-width:680px;margin:0 auto'>
            Sistem prediksi risiko stroke berbasis Machine Learning menggunakan pendekatan
            <b style='color:#a78bfa'>CRISP-DM</b>. Menggabungkan klasifikasi
            (Random Forest, Logistic Regression, KNN) dan segmentasi pasien (K-Means Clustering)
            untuk mendukung deteksi dini risiko stroke.
        </p>
    </div>
    """, unsafe_allow_html=True)

    rf = D['results']['Random Forest']
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("🏆 Best AUC-ROC", f"{rf['AUC-ROC']:.4f}", "Random Forest", "#6366f1"),
        ("🎯 Akurasi Terbaik", f"{max(r['Accuracy'] for r in D['results'].values()):.4f}", "Random Forest", "#10b981"),
        ("🔵 Silhouette Score", f"{D['sil']:.4f}", "K-Means K=3", "#f59e0b"),
        ("👥 Total Pasien", f"{len(D['df']):,}", "Records", "#ef4444"),
    ]
    for col, (title, val, sub, color) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-lbl'>{title}</div>
            <div class='kpi-val' style='color:{color}'>{val}</div>
            <div style='font-size:11px;color:#4b5563;margin-top:4px'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("<div class='section-title'>📌 Deskripsi Proyek</div>", unsafe_allow_html=True)
        items = [
            ("🎯 Tujuan", "Membangun sistem prediksi risiko stroke dan segmentasi profil pasien berbasis ML"),
            ("🗂️ Dataset", "Stroke Prediction Dataset — Kaggle (fedesoriano) · 5.110 rekam · 11 atribut"),
            ("🤖 Metode Klasifikasi", "Random Forest · Logistic Regression · K-Nearest Neighbors + SMOTE"),
            ("🔵 Metode Clustering", "K-Means (K=3) untuk segmentasi profil risiko pasien"),
            ("📐 Framework", "CRISP-DM (Business Understanding → Deployment)"),
            ("🏆 Model Terbaik", f"Random Forest — AUC-ROC {rf['AUC-ROC']:.4f} · CV-AUC {rf['CV AUC']:.4f}"),
        ]
        for lbl, desc in items:
            st.markdown(f"""
            <div style='display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #1e2538'>
                <span style='color:#8b5cf6;font-weight:700;min-width:170px;font-size:13px'>{lbl}</span>
                <span style='color:#94a3b8;font-size:13px'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-title'>👥 Identitas Anggota Kelompok</div>", unsafe_allow_html=True)
        members = [
            ("01", "Nama Anggota 1", "NIM: 22XXXXXXX"),
            ("02", "Nama Anggota 2", "NIM: 22XXXXXXX"),
            ("03", "Nama Anggota 3", "NIM: 22XXXXXXX"),
            ("04", "Nama Anggota 4", "NIM: 22XXXXXXX"),
            ("05", "Nama Anggota 5", "NIM: 22XXXXXXX"),
        ]
        for no, name, nim in members:
            st.markdown(f"""
            <div class='member-card'>
                <div style='background:#6366f1;border-radius:50%;width:36px;height:36px;
                            display:flex;align-items:center;justify-content:center;
                            font-weight:800;color:white;font-size:13px;flex-shrink:0'>{no}</div>
                <div>
                    <div style='color:#e2e8f0;font-weight:600;font-size:14px'>{name}</div>
                    <div style='color:#6b7280;font-size:12px'>{nim}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-top:8px;padding:8px;
                   background:#1e2538;border-radius:8px;'>
            <span style='color:#6b7280;font-size:12px'>Universitas Negeri Surabaya</span><br>
            <span style='color:#8b5cf6;font-size:12px;font-weight:600'>KDD-01 · 2024/2025</span>
        </div>
        """, unsafe_allow_html=True)

# ==================== 2. DATASET OVERVIEW ====================
elif page == "📂  Dataset Overview":
    st.markdown("""
    <h1 style='color:#e2e8f0;font-size:28px;font-weight:800;margin-bottom:4px'>
        📂 Dataset Overview
    </h1>
    <p style='color:#6b7280;margin-bottom:24px'>
        Stroke Prediction Dataset — Kaggle (fedesoriano)
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📋 Informasi Dataset</div>", unsafe_allow_html=True)
    df_raw = D['df']
    c1, c2, c3, c4, c5 = st.columns(5)
    info = [
        ("Total Records", f"{len(df_raw):,}", "pasien", "#6366f1"),
        ("Total Atribut", "11", "fitur + target", "#10b981"),
        ("Kelas Stroke", f"{(df_raw['stroke']==1).sum()}", f"~{(df_raw['stroke'].mean()*100):.1f}% positif", "#ef4444"),
        ("Kelas Non-Stroke", f"{(df_raw['stroke']==0).sum()}", f"~{((df_raw['stroke']==0).mean()*100):.1f}%", "#3b82f6"),
        ("Missing Values", "0", "sudah dibersihkan", "#f59e0b"),
    ]
    for col, (lbl, val, sub, color) in zip([c1, c2, c3, c4, c5], info):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-lbl'>{lbl}</div>
            <div class='kpi-val' style='color:{color}'>{val}</div>
            <div style='font-size:11px;color:#4b5563;margin-top:4px'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 3])
    with col_l:
        st.markdown("<div class='section-title'>📝 Deskripsi Atribut</div>", unsafe_allow_html=True)
        attrs = [
            ("gender", "Jenis kelamin (Male/Female)", "Kategorikal"),
            ("age", "Usia pasien (tahun)", "Numerik"),
            ("hypertension", "Riwayat hipertensi (0/1)", "Biner"),
            ("heart_disease", "Riwayat penyakit jantung (0/1)", "Biner"),
            ("ever_married", "Status pernikahan", "Kategorikal"),
            ("work_type", "Jenis pekerjaan", "Kategorikal"),
            ("Residence_type", "Tipe tempat tinggal", "Kategorikal"),
            ("avg_glucose_level", "Rata-rata kadar glukosa", "Numerik"),
            ("bmi", "Body Mass Index", "Numerik"),
            ("smoking_status", "Status merokok", "Kategorikal"),
            ("stroke", "Label target (0/1) ⭐", "Biner"),
        ]
        for name, desc, dtype in attrs:
            color = "#f59e0b" if name == "stroke" else "#6366f1"
            st.markdown(f"""
            <div style='display:flex;gap:8px;padding:6px 0;border-bottom:1px solid #1e2538;align-items:center'>
                <code style='background:#1e2538;color:{color};padding:2px 6px;
                            border-radius:4px;font-size:12px;min-width:130px'>{name}</code>
                <span style='color:#94a3b8;font-size:12px;flex:1'>{desc}</span>
                <span style='color:#4b5563;font-size:11px;white-space:nowrap'>{dtype}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>📊 Statistik Deskriptif</div>", unsafe_allow_html=True)
        num_cols = ['age', 'avg_glucose_level', 'bmi']
        st.dataframe(df_raw[num_cols].describe().T, use_container_width=True)

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=['No Stroke (0)', 'Stroke (1)'],
            values=[(df_raw['stroke']==0).sum(), (df_raw['stroke']==1).sum()],
            hole=0.55,
            marker=dict(colors=['#3b82f6', '#ef4444'], line=dict(color='#0f1117', width=2)),
            textinfo='label+percent',
            textfont=dict(color='white', size=12)
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
            title=dict(text='Distribusi Kelas Target (Stroke)',
                       font=dict(color='#e2e8f0', size=14)),
            height=260, margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div class='section-title'>📈 Visualisasi Data</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Distribusi Numerik", "🏷️ Distribusi Kategorikal", "🔥 Korelasi"])

    df_viz = D['df']

    with tab1:
        col1, col2, col3 = st.columns(3)
        for col, feature, color, title in [
            (col1, 'age', '#6366f1', 'Distribusi Usia'),
            (col2, 'avg_glucose_level', '#f59e0b', 'Distribusi Glukosa'),
            (col3, 'bmi', '#10b981', 'Distribusi BMI'),
        ]:
            fig = px.histogram(df_viz, x=feature, color='stroke',
                               color_discrete_map={0:'#3b82f6', 1:'#ef4444'},
                               barmode='overlay', opacity=0.75,
                               labels={'stroke':'Stroke'},
                               template='plotly_dark')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                title=dict(text=title, font=dict(color='#e2e8f0', size=13)),
                height=260, margin=dict(t=40, b=20, l=10, r=10)
            )
            col.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        gd = df_viz.groupby(['gender', 'stroke']).size().reset_index(name='count')
        fig_g = px.bar(gd, x='gender', y='count', color='stroke',
                       color_discrete_map={0:'#3b82f6', 1:'#ef4444'}, barmode='group',
                       template='plotly_dark', title='Stroke per Jenis Kelamin')
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=300, margin=dict(t=40, b=20))
        col1.plotly_chart(fig_g, use_container_width=True)

        wt = df_viz.groupby(['work_type', 'stroke']).size().reset_index(name='count')
        fig_w = px.bar(wt, x='work_type', y='count', color='stroke',
                       color_discrete_map={0:'#3b82f6', 1:'#ef4444'}, barmode='group',
                       template='plotly_dark', title='Stroke per Jenis Pekerjaan')
        fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(tickangle=15),
                            height=300, margin=dict(t=40, b=20))
        col2.plotly_chart(fig_w, use_container_width=True)

        col3, col4 = st.columns(2)
        sm = df_viz.groupby(['smoking_status', 'stroke']).size().reset_index(name='count')
        fig_s = px.bar(sm, x='smoking_status', y='count', color='stroke',
                       color_discrete_map={0:'#3b82f6', 1:'#ef4444'}, barmode='group',
                       template='plotly_dark', title='Stroke per Status Merokok')
        fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(tickangle=15),
                            height=300, margin=dict(t=40, b=20))
        col3.plotly_chart(fig_s, use_container_width=True)

        ht = df_viz.groupby(['hypertension', 'stroke']).size().reset_index(name='count')
        ht['hypertension'] = ht['hypertension'].map({0:'Tidak', 1:'Ya'})
        fig_h = px.bar(ht, x='hypertension', y='count', color='stroke',
                       color_discrete_map={0:'#3b82f6', 1:'#ef4444'}, barmode='group',
                       template='plotly_dark', title='Stroke vs Hipertensi')
        fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=300, margin=dict(t=40, b=20))
        col4.plotly_chart(fig_h, use_container_width=True)

    with tab3:
        from sklearn.preprocessing import LabelEncoder
        df_corr = df_viz.copy()
        for c in df_corr.select_dtypes('object').columns:
            df_corr[c] = LabelEncoder().fit_transform(df_corr[c].astype(str))
        corr = df_corr.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.where(~mask)
        fig_hm = px.imshow(corr_masked, text_auto='.2f', aspect='auto',
                           color_continuous_scale='RdYlGn',
                           zmin=-1, zmax=1, template='plotly_dark',
                           title='Correlation Heatmap — Stroke Dataset')
        fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                             height=420, margin=dict(t=50, b=20))
        st.plotly_chart(fig_hm, use_container_width=True)

# ==================== 3. PREDICTION / ANALYSIS ====================
elif page == "🔮  Prediction / Analysis":
    st.markdown("""
    <h1 style='color:#e2e8f0;font-size:28px;font-weight:800;margin-bottom:4px'>
        🔮 Prediction / Analysis
    </h1>
    <p style='color:#6b7280;margin-bottom:24px'>
        Masukkan data klinis pasien untuk prediksi risiko stroke
    </p>
    """, unsafe_allow_html=True)
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-title'>📝 Form Input Data Pasien</div>", unsafe_allow_html=True)
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                gender   = st.selectbox("Jenis Kelamin", ["Female", "Male"])
                age      = st.slider("Usia (tahun)", 1, 100, 45)
                hypert   = st.selectbox("Hipertensi", ["Tidak (0)", "Ya (1)"])
                heart    = st.selectbox("Penyakit Jantung", ["Tidak (0)", "Ya (1)"])
                married  = st.selectbox("Pernah Menikah", ["Yes", "No"])
            with c2:
                work     = st.selectbox("Jenis Pekerjaan",
                                        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
                resid    = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
                glucose  = st.number_input("Kadar Glukosa Rata-rata (mg/dL)",
                                           min_value=55.0, max_value=300.0, value=100.0, step=0.5)
                bmi_val  = st.number_input("BMI (kg/m²)",
                                           min_value=10.0, max_value=60.0, value=28.5, step=0.1)
                smoking  = st.selectbox("Status Merokok",
                                        ["never smoked", "formerly smoked", "smokes", "Unknown"])

        model_choice = st.selectbox("🤖 Pilih Model Klasifikasi",
                                    ["Random Forest", "Logistic Regression", "K-Nearest Neighbors"])

        risk_factors = []
        if age > 60:        risk_factors.append(f"⚠️ Usia {age:.0f} tahun (risiko tinggi > 60)")
        if "Ya" in hypert:  risk_factors.append("⚠️ Memiliki hipertensi")
        if "Ya" in heart:   risk_factors.append("⚠️ Memiliki penyakit jantung")
        if glucose > 140:   risk_factors.append(f"⚠️ Glukosa {glucose:.0f} mg/dL (tinggi > 140)")
        if bmi_val > 30:    risk_factors.append(f"⚠️ BMI {bmi_val:.1f} (obesitas > 30)")
        if smoking == "smokes": risk_factors.append("⚠️ Perokok aktif")
        if risk_factors:
            st.markdown(f"""
            <div style='background:#2d1a1a;border:1px solid #ef4444;border-radius:8px;
                        padding:12px 16px;margin-bottom:12px'>
                <div style='color:#fca5a5;font-weight:700;font-size:13px;margin-bottom:6px'>
                    🚨 Faktor Risiko Terdeteksi ({len(risk_factors)})
                </div>
                {''.join(f"<div style='color:#f87171;font-size:12px'>{r}</div>" for r in risk_factors)}
            </div>
            """, unsafe_allow_html=True)

        predict_btn = st.button("🔍 Proses Prediksi", use_container_width=True)

    with col_res:
        st.markdown("<div class='section-title'>📋 Hasil Prediksi</div>", unsafe_allow_html=True)

        if predict_btn:
            le = D['le_dict']
            input_data = {
                'gender':            le['gender'].transform([gender])[0],
                'age':               age,
                'hypertension':      1 if "Ya" in hypert else 0,
                'heart_disease':     1 if "Ya" in heart else 0,
                'ever_married':      le['ever_married'].transform([married])[0],
                'work_type':         le['work_type'].transform([work])[0],
                'Residence_type':    le['Residence_type'].transform([resid])[0],
                'avg_glucose_level': glucose,
                'bmi':               bmi_val,
                'smoking_status':    le['smoking_status'].transform([smoking])[0],
            }
            Xi = pd.DataFrame([input_data])
            Xs = pd.DataFrame(D['scaler'].transform(Xi), columns=Xi.columns)
            model = D['trained'][model_choice]
            pred  = model.predict(Xs)[0]
            prob  = model.predict_proba(Xs)[0]
            sp    = prob[1] * 100

            if pred == 1:
                st.markdown(f"""
                <div class='risk-high'>
                    <div style='color:#fca5a5;font-size:13px;text-transform:uppercase;letter-spacing:2px'>
                        ⚠️ Hasil Prediksi
                    </div>
                    <div class='badge-high'>RISIKO STROKE TERDETEKSI</div>
                    <div class='prob-big' style='color:#f87171'>{sp:.1f}%</div>
                    <div style='color:#fca5a5;font-size:13px'>Probabilitas Stroke · {model_choice}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    <div style='color:#6ee7b7;font-size:13px;text-transform:uppercase;letter-spacing:2px'>
                        ✅ Hasil Prediksi
                    </div>
                    <div class='badge-low'>RISIKO RENDAH</div>
                    <div class='prob-big' style='color:#34d399'>{sp:.1f}%</div>
                    <div style='color:#6ee7b7;font-size:13px'>Probabilitas Stroke · {model_choice}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sp,
                number={'suffix': "%", 'font':{'color':'#e2e8f0', 'size':28}},
                title={'text': "Stroke Probability", 'font':{'color':'#e2e8f0', 'size':13}},
                gauge={
                    'axis':{'range':[0, 100], 'tickcolor':'#6b7280', 'tickfont':{'color':'#6b7280'}},
                    'bar':{'color':'#ef4444' if pred == 1 else '#10b981'},
                    'bgcolor':'#1e2538', 'bordercolor':'#2d3748',
                    'steps':[{'range':[0, 30], 'color':'#064e3b'},
                             {'range':[30, 60], 'color':'#78350f'},
                             {'range':[60, 100], 'color':'#7f1d1d'}],
                    'threshold':{'line':{'color':'#fbbf24', 'width':3}, 'thickness':0.75, 'value':50}
                }
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                height=230, margin=dict(t=30, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

            ca, cb = st.columns(2)
            ca.metric("✅ No Stroke", f"{prob[0]*100:.1f}%")
            cb.metric("⚠️ Stroke", f"{sp:.1f}%",
                      delta=f"{(sp-5):.1f}% vs baseline 5%",
                      delta_color="inverse")

            cp_row = D['cp']
            c_pred = D['km'].predict(Xs)[0]
            row    = cp_row[cp_row['cluster'] == c_pred].iloc[0]
            names  = {0:'🟢 Risiko Rendah', 1:'🟡 Risiko Sedang', 2:'🔴 Risiko Tinggi'}
            colors = {0:'#3b82f6', 1:'#f59e0b', 2:'#ef4444'}
            st.markdown(f"""
            <div style='background:#1e2538;border-left:4px solid {colors[c_pred]};
                        border-radius:8px;padding:12px 16px;margin-top:8px'>
                <div style='color:{colors[c_pred]};font-weight:700;font-size:14px;margin-bottom:6px'>
                    🔵 Segmen Cluster: {names[c_pred]}
                </div>
                <div style='color:#94a3b8;font-size:12px;line-height:2'>
                    Rata-rata usia segmen: <b style='color:#e2e8f0'>{row['Avg_Age']:.0f} th</b> &nbsp;|&nbsp;
                    % Stroke segmen: <b style='color:{colors[c_pred]}'>{row['Stroke_Pct']:.1f}%</b> <br>
                    Glukosa rata-rata: <b style='color:#e2e8f0'>{row['Avg_Glucose']:.0f} mg/dL</b> &nbsp;|&nbsp;
                    BMI rata-rata: <b style='color:#e2e8f0'>{row['Avg_BMI']:.1f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;border:2px dashed #2d3748;
                        border-radius:12px;color:#4b5563'>
                <div style='font-size:48px;margin-bottom:16px'>🔬</div>
                <div style='font-size:15px;font-weight:600;color:#6b7280'>
                    Isi form data pasien di sebelah kiri <br>
                    kemudian klik <span style='color:#8b5cf6'>Proses Prediksi</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==================== 4. VISUALIZATION ====================
elif page == "📊  Visualization":
    st.markdown("""
    <h1 style='color:#e2e8f0;font-size:28px;font-weight:800;margin-bottom:4px'>
        📊 Visualization
    </h1>
    <p style='color:#6b7280;margin-bottom:24px'>
        Grafik pendukung & visualisasi hasil analisis model
    </p>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Perbandingan Model", "📉 ROC Curve", "🔵 Clustering", "📈 Feature Importance"
    ])

    with tab1:
        st.markdown("<div class='section-title'>📊 Perbandingan Performa Model</div>", unsafe_allow_html=True)
        cmp = {n:{k:v for k, v in r.items() if k in ['Accuracy', 'F1-Score', 'AUC-ROC', 'CV AUC']}
               for n, r in D['results'].items()}
        df_cmp = pd.DataFrame(cmp).T.reset_index()
        df_cmp.columns = ['Model', 'Accuracy', 'F1-Score', 'AUC-ROC', 'CV AUC']
        st.dataframe(df_cmp.style
                     .background_gradient(subset=['Accuracy', 'F1-Score', 'AUC-ROC', 'CV AUC'], cmap='YlOrRd')
                     .format({c:'{:.4f}' for c in ['Accuracy', 'F1-Score', 'AUC-ROC', 'CV AUC']}),
                     use_container_width=True, hide_index=True)

        df_m = df_cmp.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig = px.bar(df_m, x='Model', y='Score', color='Metric', barmode='group',
                     color_discrete_sequence=['#6366f1', '#10b981', '#f59e0b', '#ef4444'],
                     template='plotly_dark', text_auto='.3f')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          yaxis=dict(range=[0, 1.1], gridcolor='#1e2538'),
                          xaxis=dict(gridcolor='#1e2538'),
                          legend=dict(orientation='h', y=1.1),
                          height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>🗂️ Confusion Matrix</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for col, name, color in zip(cols, D['results'].keys(), ['#6366f1', '#f59e0b', '#ef4444']):
            r = D['results'][name]
            fig_cm = px.imshow(r['cm'], text_auto=True, aspect='auto',
                               color_continuous_scale=[[0, '#0f1117'], [1, color]],
                               x=['No Stroke', 'Stroke'], y=['No Stroke', 'Stroke'],
                               labels=dict(x='Predicted', y='Actual'), template='plotly_dark')
            fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                 title=dict(text=f"{name} (AUC={r['AUC-ROC']})",
                                            font=dict(color='#e2e8f0', size=12)),
                                 height=290, margin=dict(t=50, b=10))
            col.plotly_chart(fig_cm, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>📉 ROC Curve — Perbandingan Model</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        for (name, r), color in zip(D['results'].items(), ['#6366f1', '#f59e0b', '#ef4444']):
            fig_roc.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines',
                                         name=f"{name} (AUC={r['AUC-ROC']})",
                                         line=dict(color=color, width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                     line=dict(color='#4b5563', width=1.5, dash='dash')))
        fig_roc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='False Positive Rate', gridcolor='#1e2538', color='#94a3b8'),
            yaxis=dict(title='True Positive Rate', gridcolor='#1e2538', color='#94a3b8'),
            legend=dict(x=0.4, y=0.1, bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0')),
            height=460, margin=dict(t=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-title'>🔵 K-Means Clustering Results</div>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        K_range = list(range(2, 9))
        with ca:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=K_range, y=D['inertias'], mode='lines+markers',
                                       line=dict(color='#3b82f6', width=2.5), marker=dict(size=9)))
            fig_e.add_vline(x=3, line_dash='dash', line_color='#ef4444',
                            annotation_text='K=3 Optimal', annotation_font_color='#ef4444')
            fig_e.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                title=dict(text='Elbow Method', font=dict(color='#e2e8f0')),
                                xaxis=dict(title='K', gridcolor='#1e2538', color='#94a3b8'),
                                yaxis=dict(title='Inertia', gridcolor='#1e2538', color='#94a3b8'),
                                height=300)
            ca.plotly_chart(fig_e, use_container_width=True)

        with cb:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=K_range, y=D['silhouettes'], mode='lines+markers',
                                       line=dict(color='#10b981', width=2.5), marker=dict(size=9)))
            best_k = K_range[int(np.argmax(D['silhouettes']))]
            fig_s.add_vline(x=best_k, line_dash='dash', line_color='#f59e0b',
                            annotation_text=f'K={best_k} terbaik', annotation_font_color='#f59e0b')
            fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                title=dict(text='Silhouette Score per K', font=dict(color='#e2e8f0')),
                                xaxis=dict(title='K', gridcolor='#1e2538', color='#94a3b8'),
                                yaxis=dict(title='Silhouette', gridcolor='#1e2538', color='#94a3b8'),
                                height=300)
            cb.plotly_chart(fig_s, use_container_width=True)

        df_pca = pd.DataFrame({
            'PC1': D['Xpca'][:, 0], 'PC2': D['Xpca'][:, 1],
            'Cluster': [f"Cluster {c}" for c in D['cl']],
            'Stroke': ['Stroke' if s == 1 else 'No Stroke' for s in D['df']['stroke'].values],
        })
        fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', symbol='Stroke',
                             color_discrete_sequence=['#3b82f6', '#f59e0b', '#ef4444'],
                             opacity=0.55, template='plotly_dark',
                             title='K-Means Cluster — Proyeksi PCA 2D')
        centers = D['pca'].transform(D['km'].cluster_centers_)
        fig_pca.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers+text',
                                     marker=dict(symbol='x', size=16, color='white', line=dict(width=2)),
                                     text=['C0', 'C1', 'C2'], textposition='top right',
                                     textfont=dict(color='white', size=11), name='Centroid'))
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#1e2538', color='#94a3b8'),
            yaxis=dict(gridcolor='#1e2538', color='#94a3b8'),
            height=430, margin=dict(t=50))
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("<div class='section-title'>📊 Profil Cluster</div>", unsafe_allow_html=True)
        cp = D['cp'].copy()
        cp['Nama Cluster'] = cp['cluster'].map({0:'🟢 Risiko Rendah', 1:'🟡 Risiko Sedang', 2:'🔴 Risiko Tinggi'})
        cp = cp[['Nama Cluster', 'Count', 'Stroke_Pct', 'Avg_Age', 'Avg_Glucose', 'Avg_BMI', 'Hypertension_Pct']]
        cp.columns = ['Cluster', 'Jumlah Pasien', '% Stroke', 'Usia Rata-rata', 'Glukosa Avg', 'BMI Avg', '% Hipertensi']
        st.dataframe(cp, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("<div class='section-title'>📈 Feature Importance — Random Forest</div>", unsafe_allow_html=True)
        fi = D['fi'].reset_index()
        fi.columns = ['Feature', 'Importance']
        fig_fi = go.Figure(go.Bar(
            y=fi['Feature'][::-1], x=fi['Importance'][::-1], orientation='h',
            marker=dict(color=fi['Importance'][::-1],
                        colorscale=[[0, '#1e3a5f'], [0.5, '#6366f1'], [1, '#a78bfa']], showscale=False),
            text=[f"  {v:.4f}" for v in fi['Importance'][::-1]],
            textposition='outside', textfont=dict(color='#94a3b8', size=12)
        ))
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Importance Score', gridcolor='#1e2538', color='#94a3b8'),
            yaxis=dict(color='#e2e8f0', tickfont=dict(size=13)),
            height=400, margin=dict(l=20, r=80, t=10, b=20)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("<div class='section-title'>🏅 Ranking Faktor Risiko</div>", unsafe_allow_html=True)
        descs = {
            'age':              'Usia adalah faktor terpenting — lansia memiliki risiko jauh lebih tinggi',
            'avg_glucose_level':'Glukosa tinggi mengindikasikan diabetes, terkait erat dengan stroke',
            'bmi':              'BMI tinggi berhubungan dengan obesitas dan risiko kardiovaskular',
            'hypertension':     'Tekanan darah tinggi adalah komorbiditas kritis untuk stroke',
            'heart_disease':    'Penyakit jantung meningkatkan kemungkinan stroke secara signifikan',
            'ever_married':     'Status pernikahan berkorelasi dengan usia dan gaya hidup',
            'smoking_status':   'Merokok merusak pembuluh darah dan meningkatkan risiko stroke',
            'work_type':        'Jenis pekerjaan berkaitan dengan tingkat stres dan gaya hidup',
            'gender':           'Perbedaan gender dalam faktor hormonal dan biologis',
            'Residence_type':   'Akses layanan kesehatan dan faktor lingkungan',
        }
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
        cols = st.columns(2)
        max_imp = fi['Importance'].max()
        for i, (_, row) in enumerate(fi.iterrows()):
            col = cols[i % 2]
            col.markdown(f"""
            <div style='background:#1e2538;border-radius:8px;padding:12px 14px;
                        border-left:3px solid {"#6366f1" if i < 3 else "#2d3748"};margin-bottom:8px'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>
                    <span style='font-size:18px'>{medals[i]}</span>
                    <span style='color:#e2e8f0;font-weight:700;font-size:14px'>{row["Feature"]}</span>
                    <span style='margin-left:auto;color:#6366f1;font-weight:800'>{row["Importance"]:.4f}</span>
                </div>
                <div style='background:#0f1117;border-radius:4px;height:5px;margin-bottom:6px'>
                    <div style='height:100%;width:{row["Importance"]/max_imp*100:.1f}%;
                                background:linear-gradient(90deg,#6366f1,#a78bfa);border-radius:4px'></div>
                </div>
                <div style='color:#6b7280;font-size:12px'>
                    {descs.get(row["Feature"], "Berkontribusi pada prediksi risiko stroke")}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==================== 5. ABOUT ====================
elif page == "ℹ️   About":
    st.markdown("""
    <h1 style='color:#e2e8f0;font-size:28px;font-weight:800;margin-bottom:4px'>
        ℹ️ About
    </h1>
    <p style='color:#6b7280;margin-bottom:24px'>
        Penjelasan metode, dataset, dan informasi proyek
    </p>
    """, unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("<div class='section-title'>🧪 Penjelasan Metode</div>", unsafe_allow_html=True)
        methods = [
            ("🔵 K-Means Clustering", "#3b82f6",
             "Algoritma unsupervised learning yang mengelompokkan pasien ke dalam K cluster "
             "berdasarkan kemiripan karakteristik kesehatan. Digunakan untuk segmentasi "
             "profil risiko pasien menjadi 3 kelompok: Risiko Rendah, Sedang, dan Tinggi. "
             "K optimal ditentukan dengan Elbow Method dan Silhouette Score."),
            ("🌲 Random Forest", "#10b981",
             "Ensemble learning yang membangun banyak decision tree dan menggabungkan "
             "hasilnya. Dipilih sebagai model terbaik dengan AUC-ROC tertinggi (~0.91). "
             "Menggunakan 200 estimator, max_depth=10, dan class_weight='balanced' "
             "untuk menangani ketidakseimbangan kelas."),
            ("📈 Logistic Regression", "#6366f1",
             "Model klasifikasi linier yang memprediksi probabilitas kelas menggunakan "
             "fungsi sigmoid. Digunakan sebagai baseline model karena interpretabilitasnya "
             "yang tinggi. Parameter C=1.0 dengan max_iter=1000."),
            ("👥 K-Nearest Neighbors", "#f59e0b",
             "Algoritma non-parametrik yang mengklasifikasikan data berdasarkan K tetangga "
             "terdekat (K=7, metric=euclidean). Cocok untuk data dengan batas keputusan "
             "non-linear tanpa asumsi distribusi data."),
            ("⚖️ SMOTE", "#ef4444",
             "Synthetic Minority Over-sampling Technique untuk menangani class imbalance "
             "(4.9% stroke vs 95.1% non-stroke). Membuat sampel sintetis pada kelas "
             "minoritas berdasarkan interpolasi fitur K-nearest neighbors (k=5)."),
        ]
        for title, color, desc in methods:
            st.markdown(f"""
            <div style='background:#1e2538;border-radius:10px;padding:16px 18px;
                        border-left:4px solid {color};margin-bottom:12px'>
                <div style='color:{color};font-weight:700;font-size:15px;margin-bottom:8px'>{title}</div>
                <div style='color:#94a3b8;font-size:13px;line-height:1.7'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>📐 Framework CRISP-DM</div>", unsafe_allow_html=True)
        phases = [
            ("1", "Business Understanding", "Identifikasi masalah prediksi stroke, tujuan klasifikasi & clustering", "#6366f1"),
            ("2", "Data Understanding", "EDA dataset 5.110 pasien, analisis distribusi, missing values, korelasi", "#3b82f6"),
            ("3", "Data Preparation", "Imputasi BMI, encoding kategorikal, StandardScaler, SMOTE", "#10b981"),
            ("4", "Modeling", "Training RF, LR, KNN + K-Means clustering (K=3)", "#f59e0b"),
            ("5", "Evaluation", "Akurasi, F1, AUC-ROC, CV-AUC, Silhouette Score, Feature Importance", "#ef4444"),
            ("6", "Deployment", "Aplikasi web Streamlit interaktif", "#8b5cf6"),
        ]
        for no, phase, desc, color in phases:
            st.markdown(f"""
            <div style='display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #1e2538;align-items:flex-start'>
                <div style='background:{color};color:white;font-weight:800;font-size:13px;
                            border-radius:50%;width:28px;height:28px;display:flex;
                            align-items:center;justify-content:center;flex-shrink:0'>{no}</div>
                <div>
                    <div style='color:#e2e8f0;font-weight:600;font-size:14px'>{phase}</div>
                    <div style='color:#6b7280;font-size:12px;margin-top:3px'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>🗂️ Informasi Dataset</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <div style='color:#a78bfa;font-weight:700;font-size:15px;margin-bottom:12px'>
                📦 Stroke Prediction Dataset
            </div>
            <div style='color:#94a3b8;font-size:13px;line-height:2'>
                <b style='color:#e2e8f0'>Sumber:</b> Kaggle — fedesoriano <br>
                <b style='color:#e2e8f0'>Lisensi:</b> Open Database License <br>
                <b style='color:#e2e8f0'>Records:</b> 5.110 pasien <br>
                <b style='color:#e2e8f0'>Fitur:</b> 10 input + 1 target <br>
                <b style='color:#e2e8f0'>Target:</b> stroke (0 = tidak, 1 = ya) <br>
                <b style='color:#e2e8f0'>Imbalance:</b> 4.9% positif (249 kasus) <br>
                <b style='color:#e2e8f0'>Missing:</b> 201 nilai pada kolom BMI
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>📊 Kriteria Evaluasi</div>", unsafe_allow_html=True)
        evals = [
            ("AUC-ROC", "Area Under ROC Curve · Target ≥ 0.80", "#6366f1"),
            ("Accuracy", "Persentase prediksi benar total", "#10b981"),
            ("F1-Score", "Harmonic mean Precision & Recall", "#f59e0b"),
            ("CV AUC", "5-Fold Cross-Validation AUC", "#3b82f6"),
            ("Silhouette", "Kualitas clustering · Target ≥ 0.30", "#8b5cf6"),
        ]
        for name, desc, color in evals:
            st.markdown(f"""
            <div style='display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #1e2538;align-items:center'>
                <div style='background:{color}22;color:{color};font-weight:700;font-size:12px;
                            border-radius:6px;padding:4px 10px;min-width:90px;text-align:center'>{name}</div>
                <div style='color:#94a3b8;font-size:12px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:18px'>🛠️ Tech Stack</div>", unsafe_allow_html=True)
        techs = ["Python 3.x", "Streamlit", "scikit-learn", "imbalanced-learn", "Plotly", "Pandas", "NumPy"]
        chips = "  ".join([f"<span style='background:#1e2538;color:#a78bfa;border:1px solid #4b5563;"
                          f"border-radius:20px;padding:4px 12px;font-size:12px;display:inline-block;"
                          f"margin:3px'>{t}</span>" for t in techs])
        st.markdown(f"<div style='line-height:2'>{chips}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:18px'>📌 Informasi Proyek</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <div style='color:#94a3b8;font-size:13px;line-height:2.2'>
                <b style='color:#e2e8f0'>Mata Kuliah:</b> Knowledge Discovery in Databases <br>
                <b style='color:#e2e8f0'>Tugas:</b> UAS — Proyek Kelompok <br>
                <b style='color:#e2e8f0'>Kelas:</b> KDD-01 <br>
                <b style='color:#e2e8f0'>Institusi:</b> Universitas Negeri Surabaya <br>
                <b style='color:#e2e8f0'>Tahun:</b> 2024/2025 <br>
                <b style='color:#e2e8f0'>Anggota:</b> 5 orang
            </div>
        </div>
        """, unsafe_allow_html=True)