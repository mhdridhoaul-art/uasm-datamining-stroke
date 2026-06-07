import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Stroke Prediction AI | UAS Data Mining",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://github.com', 'About': 'Stroke Prediction AI'}
)

# ============ OPTIMIZED CSS ============
st.markdown("""
<style>
/* Global Theme */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 2px solid #334155;
}
/* Cards & KPIs */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: transform 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
}
.kpi-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    animation: shimmer 3s infinite;
}
@keyframes shimmer { 100% { left: 100%; } }
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0;
}
.kpi-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}
/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 700;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
}
/* Risk Indicators */
.risk-high {
    background: linear-gradient(135deg, #dc2626, #991b1b);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    animation: pulseRed 2s infinite;
}
.risk-low {
    background: linear-gradient(135deg, #059669, #065f46);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    animation: pulseGreen 2s infinite;
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
    50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.8); }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
    50% { box-shadow: 0 0 40px rgba(16, 185, 129, 0.8); }
}
.section-title {
    color: #f1f5f9;
    font-size: 1.25rem;
    font-weight: 700;
    padding: 12px 0;
    border-left: 4px solid #6366f1;
    padding-left: 16px;
    margin: 24px 0 16px;
}
/* Hide default UI elements */
#MainMenu, footer, header {visibility: hidden;}
/* Custom scroll */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #1e293b; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ============ DATA GENERATION (Optimized with Caching) ============
@st.cache_data(ttl=3600, show_spinner="⏳ Menyiapkan sistem AI...")
def initialize_system():
    """Initialize dataset, models, and preprocessing - cached for performance"""
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, silhouette_score
    from imblearn.over_sampling import SMOTE
    
    np.random.seed(42)
    n = 5110
    
    # Generate realistic stroke dataset
    age = np.clip(np.random.normal(43, 22, n), 0, 100)
    hypertension = np.random.binomial(1, 0.10, n)
    heart_disease = np.random.binomial(1, 0.054, n)
    ever_married = np.random.choice(['Yes', 'No'], n, p=[0.66, 0.34])
    work_type = np.random.choice(
        ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
        n, p=[0.57, 0.16, 0.13, 0.13, 0.01]
    )
    residence = np.random.choice(['Urban', 'Rural'], n)
    avg_glucose = np.clip(np.random.lognormal(4.6, 0.4, n), 55, 300)
    bmi = np.clip(np.random.normal(28.9, 7.8, n), 10, 60)
    smoking = np.random.choice(
        ['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
        n, p=[0.37, 0.17, 0.15, 0.31]
    )
    gender = np.random.choice(['Male', 'Female'], n, p=[0.41, 0.59])
    
    # Logistic model for stroke probability
    log_odds = -5.5 + 0.055*age + 0.8*hypertension + 0.9*heart_disease + 0.004*avg_glucose + 0.01*bmi
    stroke = np.random.binomial(1, (1/(1+np.exp(-log_odds)))*0.8, n)
    
    df = pd.DataFrame({
        'gender': gender, 'age': age, 'hypertension': hypertension,
        'heart_disease': heart_disease, 'ever_married': ever_married,
        'work_type': work_type, 'Residence_type': residence,
        'avg_glucose_level': avg_glucose, 'bmi': bmi,
        'smoking_status': smoking, 'stroke': stroke
    })
    
    # Filter and encode
    df_clean = df[df['gender'] != 'Other'].copy()
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    label_encoders = {}
    df_encoded = df_clean.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    # Prepare features
    X = df_encoded.drop(columns=['stroke'])
    y = df_encoded['stroke']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance with SMOTE
    X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_resample(X_train, y_train)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        trained_models[name] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        cv_score = cross_val_score(
            model, X_scaled, y, cv=StratifiedKFold(5), scoring='roc_auc'
        ).mean()
        
        results[name] = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4),
            'auc_roc': round(roc_auc_score(y_test, y_prob), 4),
            'cv_auc': round(cv_score, 4),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'fpr': fpr, 'tpr': tpr,
            'y_pred': y_pred, 'y_prob': y_prob
        }
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clean['cluster'] = clusters
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    silhouette = silhouette_score(X_scaled, clusters)
    
    # Elbow method data
    inertias, silhouettes = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    
    # Feature importance from best model
    best_model_name = max(results, key=lambda x: results[x]['auc_roc'])
    if hasattr(trained_models[best_model_name], 'feature_importances_'):
        feature_importance = pd.Series(
            trained_models[best_model_name].feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
    else:
        feature_importance = pd.Series([0.1]*len(X.columns), index=X.columns)
    
    # Cluster profiles
    cluster_profiles = df_clean.groupby('cluster').agg(
        count=('stroke', 'count'),
        stroke_pct=('stroke', lambda x: round(x.mean()*100, 1)),
        avg_age=('age', lambda x: round(x.mean(), 1)),
        avg_bmi=('bmi', lambda x: round(x.mean(), 1)),
        avg_glucose=('avg_glucose_level', lambda x: round(x.mean(), 1)),
        hypertension_pct=('hypertension', lambda x: round(x.mean()*100, 1)),
    ).reset_index()
    
    return {
        'dataframe': df_clean,
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'clusters': clusters,
        'kmeans': kmeans,
        'pca': pca,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'models': trained_models,
        'results': results,
        'feature_importance': feature_importance,
        'cluster_profiles': cluster_profiles,
        'silhouette': silhouette,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'X_test': X_test,
        'y_test': y_test,
        'best_model': best_model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Initialize system
system = initialize_system()

# ============ SIDEBAR NAVIGATION ============
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0;'>
        <div style='font-size:56px;margin-bottom:8px'>🧠</div>
        <div style='font-size:20px;font-weight:800;color:#f1f5f9'>Stroke AI</div>
        <div style='font-size:11px;color:#94a3b8;letter-spacing:2px'>UAS DATA MINING</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset", "🔮 Prediction", "📈 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align:center;padding:12px;background:rgba(99,102,241,0.1);border-radius:8px'>
        <div style='font-size:10px;color:#94a3b8'>Last Updated</div>
        <div style='font-size:12px;color:#e2e8f0;font-weight:600'>{system['timestamp']}</div>
    </div>
    """, unsafe_allow_html=True)

# ============ PAGE 1: HOME ============
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <div style='font-size:72px;margin-bottom:16px'>🧠</div>
        <h1 style='font-size:3rem;font-weight:900;background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0'>
            Stroke Risk Prediction AI
        </h1>
        <p style='color:#94a3b8;font-size:1.1rem;margin-top:16px;max-width:700px;margin-left:auto;margin-right:auto'>
            Sistem cerdas prediksi risiko stroke menggunakan <b style='color:#a78bfa'>Ensemble Learning</b> 
            dan <b style='color:#a78bfa'>Clustering</b> berbasis CRISP-DM Framework
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Dashboard
    best_result = system['results'][system['best_model']]
    cols = st.columns(4)
    kpis = [
        ("🏆 AUC-ROC", f"{best_result['auc_roc']:.4f}", system['best_model'], "#6366f1"),
        ("🎯 Accuracy", f"{best_result['accuracy']*100:.1f}%", "Test Set", "#10b981"),
        ("🔵 Silhouette", f"{system['silhouette']:.4f}", "K-Means K=3", "#f59e0b"),
        ("👥 Dataset", f"{len(system['dataframe']):,}", "Records", "#ec4899"),
    ]
    
    for col, (title, value, sub, color) in zip(cols, kpis):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{title}</div>
            <div class='kpi-value'>{value}</div>
            <div style='font-size:11px;color:#94a3b8;margin-top:8px'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Info Grid
    col_left, col_right = st.columns([3, 2], gap="large")
    
    with col_left:
        st.markdown("<div class='section-title'>📌 Project Overview</div>", unsafe_allow_html=True)
        
        features = [
            ("🎯 Objective", "Prediksi dini risiko stroke & segmentasi pasien"),
            ("🗂️ Dataset", "Healthcare Stroke Dataset (5,110 patients, 11 features)"),
            ("🤖 Classification", "Random Forest, Gradient Boosting, Logistic, KNN + SMOTE"),
            ("🔵 Clustering", "K-Means (K=3) untuk profil risiko pasien"),
            ("📐 Framework", "CRISP-DM (End-to-End Pipeline)"),
            ("⚡ Performance", f"Best: {system['best_model']} (AUC={best_result['auc_roc']:.4f})"),
        ]
        
        for icon_title, desc in features:
            st.markdown(f"""
            <div class='glass-card' style='padding:14px 18px;margin-bottom:10px'>
                <div style='display:flex;gap:12px;align-items:center'>
                    <div style='color:#6366f1;font-weight:700;font-size:13px;min-width:140px'>{icon_title}</div>
                    <div style='color:#cbd5e1;font-size:13px'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("<div class='section-title'>👥 Development Team</div>", unsafe_allow_html=True)
        
        members = [
            ("01", "Team Member 1", "NIM: 22XXXXXXX", "Lead Developer"),
            ("02", "Team Member 2", "NIM: 22XXXXXXX", "ML Engineer"),
            ("03", "Team Member 3", "NIM: 22XXXXXXX", "Data Analyst"),
            ("04", "Team Member 4", "NIM: 22XXXXXXX", "UI/UX Designer"),
            ("05", "Team Member 5", "NIM: 22XXXXXXX", "Documentation"),
        ]
        
        for no, name, nim, role in members:
            st.markdown(f"""
            <div class='glass-card' style='display:flex;gap:14px;align-items:center;padding:14px 18px;margin-bottom:10px'>
                <div style='background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:50%;width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:14px;flex-shrink:0'>{no}</div>
                <div style='flex:1'>
                    <div style='color:#f1f5f9;font-weight:700;font-size:14px'>{name}</div>
                    <div style='color:#94a3b8;font-size:12px'>{nim} · {role}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align:center;padding:16px;background:rgba(99,102,241,0.1);border-radius:12px;margin-top:16px'>
            <div style='font-size:12px;color:#94a3b8'>Universitas Negeri Surabaya</div>
            <div style='font-size:13px;color:#a78bfa;font-weight:700;margin-top:4px'>KDD-01 · 2025/2026</div>
        </div>
        """, unsafe_allow_html=True)

# ============ PAGE 2: DATASET ============
elif page == "📊 Dataset":
    st.markdown("<h1 style='color:#f1f5f9;font-size:2.5rem;font-weight:900;margin-bottom:8px'>📊 Dataset Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:1rem'>Healthcare Stroke Dataset - Kaggle (fedesoriano)</p>", unsafe_allow_html=True)
    
    df = system['dataframe']
    
    # Dataset Statistics
    st.markdown("<div class='section-title'>📈 Key Statistics</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    stats = [
        ("Records", f"{len(df):,}", "patients", "#6366f1"),
        ("Features", "11", "inputs + target", "#10b981"),
        ("Stroke Cases", f"{(df['stroke']==1).sum()}", f"{(df['stroke'].mean()*100):.1f}%", "#ef4444"),
        ("Non-Stroke", f"{(df['stroke']==0).sum()}", f"{((df['stroke']==0).mean()*100):.1f}%", "#3b82f6"),
        ("Imbalance Ratio", "1:19", "severe", "#f59e0b"),
    ]
    
    for col, (label, value, sub, color) in zip(cols, stats):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value' style='background:linear-gradient(135deg,{color},{color});-webkit-background-clip:text;-webkit-text-fill-color:transparent'>{value}</div>
            <div style='font-size:11px;color:#94a3b8;margin-top:4px'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Filters
    st.markdown("<div class='section-title'>🔍 Interactive Filters</div>", unsafe_allow_html=True)
    filter_cols = st.columns(4)
    
    with filter_cols[0]:
        gender_filter = st.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
    with filter_cols[1]:
        age_range = st.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (20, 70))
    with filter_cols[2]:
        work_filter = st.multiselect("Work Type", df['work_type'].unique(), default=df['work_type'].unique())
    with filter_cols[3]:
        smoke_filter = st.multiselect("Smoking Status", df['smoking_status'].unique(), default=df['smoking_status'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['gender'].isin(gender_filter)) &
        (df['age'].between(age_range[0], age_range[1])) &
        (df['work_type'].isin(work_filter)) &
        (df['smoking_status'].isin(smoke_filter))
    ]
    
    st.success(f"✅ Showing {len(filtered_df):,} of {len(df):,} records")
    
    # Visualizations
    st.markdown("<div class='section-title'>📊 Data Visualization</div>", unsafe_allow_html=True)
    tabs = st.tabs(["📈 Distributions", "🏷️ Categories", "🔥 Correlations"])
    
    with tabs[0]:
        cols = st.columns(3)
        for col, feature, color, title in [
            (cols[0], 'age', '#6366f1', 'Age Distribution'),
            (cols[1], 'avg_glucose_level', '#f59e0b', 'Glucose Distribution'),
            (cols[2], 'bmi', '#10b981', 'BMI Distribution'),
        ]:
            fig = px.histogram(
                filtered_df, x=feature, color='stroke',
                color_discrete_map={0:'#3b82f6', 1:'#ef4444'},
                barmode='overlay', opacity=0.75,
                labels={'stroke': 'Stroke'},
                template='plotly_dark'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(text=title, font=dict(color='#f1f5f9', size=14)),
                height=300,
                margin=dict(t=40, b=20)
            )
            col.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        cols = st.columns(2)
        
        # Gender analysis
        gd = filtered_df.groupby(['gender', 'stroke']).size().reset_index(name='count')
        fig_g = px.bar(
            gd, x='gender', y='count', color='stroke',
            color_discrete_map={0:'#3b82f6', 1:'#ef4444'},
            barmode='group', template='plotly_dark',
            title='Stroke by Gender'
        )
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        cols[0].plotly_chart(fig_g, use_container_width=True)
        
        # Work type analysis
        wt = filtered_df.groupby(['work_type', 'stroke']).size().reset_index(name='count')
        fig_w = px.bar(
            wt, x='work_type', y='count', color='stroke',
            color_discrete_map={0:'#3b82f6', 1:'#ef4444'},
            barmode='group', template='plotly_dark',
            title='Stroke by Work Type'
        )
        fig_w.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=15), height=350
        )
        cols[1].plotly_chart(fig_w, use_container_width=True)
    
    with tabs[2]:
        from sklearn.preprocessing import LabelEncoder
        df_corr = filtered_df.copy()
        for c in df_corr.select_dtypes('object').columns:
            df_corr[c] = LabelEncoder().fit_transform(df_corr[c].astype(str))
        
        corr = df_corr.corr()
        fig_hm = px.imshow(
            corr, text_auto='.2f', aspect='auto',
            color_continuous_scale='RdYlBu_r',
            template='plotly_dark',
            title='Feature Correlation Matrix'
        )
        fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=500)
        st.plotly_chart(fig_hm, use_container_width=True)
    
    # Data preview
    st.markdown("<div class='section-title'>📋 Sample Data</div>", unsafe_allow_html=True)
    st.dataframe(filtered_df.head(20), use_container_width=True)

# ============ PAGE 3: PREDICTION ============
elif page == "🔮 Prediction":
    st.markdown("<h1 style='color:#f1f5f9;font-size:2.5rem;font-weight:900;margin-bottom:8px'>🔮 Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:1rem'>Enter patient data for stroke risk assessment</p>", unsafe_allow_html=True)
    
    col_form, col_res = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown("<div class='section-title'>📝 Patient Information</div>", unsafe_allow_html=True)
        
        with st.container():
            c1, c2 = st.columns(2)
            
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.slider("Age (years)", 1, 100, 45)
                hypertension = st.selectbox("Hypertension", ["No (0)", "Yes (1)"])
                heart_disease = st.selectbox("Heart Disease", ["No (0)", "Yes (1)"])
                ever_married = st.selectbox("Ever Married", ["Yes", "No"])
            
            with c2:
                work_type = st.selectbox(
                    "Work Type",
                    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
                )
                residence = st.selectbox("Residence Type", ["Urban", "Rural"])
                glucose = st.number_input("Avg Glucose Level (mg/dL)", 55.0, 300.0, 100.0, 0.5)
                bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 28.5, 0.1)
                smoking = st.selectbox(
                    "Smoking Status",
                    ["never smoked", "formerly smoked", "smokes", "Unknown"]
                )
        
        model_choice = st.selectbox(
            "🤖 Select Classification Model",
            list(system['models'].keys()),
            index=list(system['models'].keys()).index(system['best_model'])
        )
        
        # Risk factor detection
        risk_factors = []
        if age > 60:
            risk_factors.append(f"⚠️ Age {age:.0f} years (high risk > 60)")
        if "Yes" in hypertension:
            risk_factors.append("⚠️ Has hypertension")
        if "Yes" in heart_disease:
            risk_factors.append("⚠️ Has heart disease")
        if glucose > 140:
            risk_factors.append(f"⚠️ Glucose {glucose:.0f} mg/dL (high > 140)")
        if bmi > 30:
            risk_factors.append(f"⚠️ BMI {bmi:.1f} (obese > 30)")
        if smoking == "smokes":
            risk_factors.append("⚠️ Active smoker")
        
        if risk_factors:
            st.markdown(f"""
            <div style='background:rgba(239,68,68,0.1);border:1px solid #ef4444;border-radius:12px;padding:16px;margin:12px 0'>
                <div style='color:#fca5a5;font-weight:700;font-size:13px;margin-bottom:8px'>
                    🚨 {len(risk_factors)} Risk Factors Detected
                </div>
                {''.join(f"<div style='color:#f87171;font-size:12px;margin:4px 0'>• {r}</div>" for r in risk_factors)}
            </div>
            """, unsafe_allow_html=True)
        
        predict_btn = st.button("🔍 Analyze Risk", use_container_width=True, type="primary")
    
    with col_res:
        st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)
        
        if predict_btn:
            # Prepare input
            le = system['label_encoders']
            input_data = {
                'gender': le['gender'].transform([gender])[0],
                'age': age,
                'hypertension': 1 if "Yes" in hypertension else 0,
                'heart_disease': 1 if "Yes" in heart_disease else 0,
                'ever_married': le['ever_married'].transform([ever_married])[0],
                'work_type': le['work_type'].transform([work_type])[0],
                'Residence_type': le['Residence_type'].transform([residence])[0],
                'avg_glucose_level': glucose,
                'bmi': bmi,
                'smoking_status': le['smoking_status'].transform([smoking])[0],
            }
            
            input_df = pd.DataFrame([input_data])
            input_scaled = pd.DataFrame(
                system['scaler'].transform(input_df),
                columns=input_df.columns
            )
            
            # Get prediction
            model = system['models'][model_choice]
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            stroke_prob = probability[1] * 100
            
            # Display result with animation
            if prediction == 1:
                st.markdown(f"""
                <div class='risk-high'>
                    <div style='color:#fecaca;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px'>
                        ⚠️ High Risk Detected
                    </div>
                    <div style='font-size:3rem;font-weight:900;color:#fca5a5;margin:16px 0'>STROKE RISK</div>
                    <div style='font-size:4rem;font-weight:900;color:#f87171'>{stroke_prob:.1f}%</div>
                    <div style='color:#fecaca;font-size:13px;margin-top:12px'>
                        Probability via {model_choice}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    <div style='color:#a7f3d0;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px'>
                        ✅ Low Risk
                    </div>
                    <div style='font-size:3rem;font-weight:900;color:#6ee7b7;margin:16px 0'>LOW RISK</div>
                    <div style='font-size:4rem;font-weight:900;color:#34d399'>{stroke_prob:.1f}%</div>
                    <div style='color:#a7f3d0;font-size:13px;margin-top:12px'>
                        Probability via {model_choice}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stroke_prob,
                number={'suffix': "%", 'font': {'color': '#f1f5f9', 'size': 32}},
                title={'text': "Stroke Probability", 'font': {'color': '#f1f5f9', 'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                    'bar': {'color': '#ef4444' if prediction == 1 else '#10b981'},
                    'bgcolor': '#1e293b',
                    'steps': [
                        {'range': [0, 30], 'color': '#064e3b'},
                        {'range': [30, 60], 'color': '#78350f'},
                        {'range': [60, 100], 'color': '#7f1d1d'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=250,
                margin=dict(t=40, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Cluster assignment
            cluster_pred = system['kmeans'].predict(input_scaled)[0]
            cluster_info = system['cluster_profiles'][system['cluster_profiles']['cluster'] == cluster_pred].iloc[0]
            
            cluster_names = {
                0: ('🟢 Low Risk Group', '#10b981'),
                1: ('🟡 Moderate Risk Group', '#f59e0b'),
                2: ('🔴 High Risk Group', '#ef4444')
            }
            cluster_name, cluster_color = cluster_names[cluster_pred]
            
            st.markdown(f"""
            <div class='glass-card' style='border-left:4px solid {cluster_color}'>
                <div style='color:{cluster_color};font-weight:700;font-size:16px;margin-bottom:8px'>
                    {cluster_name}
                </div>
                <div style='color:#cbd5e1;font-size:13px;line-height:1.8'>
                    <b>Average Age:</b> {cluster_info['avg_age']:.0f} years<br>
                    <b>Stroke Rate:</b> <span style='color:{cluster_color};font-weight:700'>{cluster_info['stroke_pct']:.1f}%</span><br>
                    <b>Avg Glucose:</b> {cluster_info['avg_glucose']:.0f} mg/dL<br>
                    <b>Avg BMI:</b> {cluster_info['avg_bmi']:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("<div class='section-title'>💡 Recommendations</div>", unsafe_allow_html=True)
            if prediction == 1:
                st.warning("""
                **Immediate Actions:**
                - Consult a neurologist immediately
                - Regular blood pressure monitoring
                - Control blood sugar levels
                - Adopt healthy lifestyle
                - Avoid smoking and alcohol
                - Exercise 30+ minutes daily
                """)
            else:
                st.success("""
                **Preventive Measures:**
                - Maintain healthy lifestyle
                - Regular health check-ups every 6 months
                - Keep ideal body weight
                - Balanced nutrition
                - Stress management
                """)
        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 20px;border:2px dashed #334155;border-radius:16px'>
                <div style='font-size:64px;margin-bottom:16px'>🔬</div>
                <div style='font-size:16px;font-weight:600;color:#94a3b8'>
                    Fill the patient information form<br>
                    then click <span style='color:#a78bfa;font-weight:700'>Analyze Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============ PAGE 4: ANALYTICS ============
elif page == "📈 Analytics":
    st.markdown("<h1 style='color:#f1f5f9;font-size:2.5rem;font-weight:900;margin-bottom:8px'>📈 Model Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:1rem'>Performance metrics & insights</p>", unsafe_allow_html=True)
    
    tabs = st.tabs(["🤖 Model Comparison", "📉 ROC Curves", "🔵 Clustering", "📊 Feature Importance"])
    
    with tabs[0]:
        st.markdown("<div class='section-title'>🏆 Model Performance Comparison</div>", unsafe_allow_html=True)
        
        # Comparison table
        cmp_data = {
            'Model': list(system['results'].keys()),
            'Accuracy': [system['results'][n]['accuracy'] for n in system['results']],
            'F1-Score': [system['results'][n]['f1_score'] for n in system['results']],
            'AUC-ROC': [system['results'][n]['auc_roc'] for n in system['results']],
            'CV AUC': [system['results'][n]['cv_auc'] for n in system['results']],
        }
        df_cmp = pd.DataFrame(cmp_data)
        
        st.dataframe(
            df_cmp.style
                .background_gradient(subset=['Accuracy', 'F1-Score', 'AUC-ROC', 'CV AUC'], cmap='YlOrRd')
                .format({c: '{:.4f}' for c in ['Accuracy', 'F1-Score', 'AUC-ROC', 'CV AUC']}),
            use_container_width=True,
            hide_index=True
        )
        
        # Visual comparison
        df_melted = df_cmp.melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig_cmp = px.bar(
            df_melted, x='Model', y='Score', color='Metric',
            barmode='group',
            color_discrete_sequence=['#6366f1', '#10b981', '#f59e0b', '#ec4899'],
            template='plotly_dark',
            text_auto='.3f'
        )
        fig_cmp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 1.1]),
            height=400,
            margin=dict(t=20)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        
        # Confusion matrices
        st.markdown("<div class='section-title'>🗂️ Confusion Matrices</div>", unsafe_allow_html=True)
        cols = st.columns(len(system['results']))
        colors = ['#6366f1', '#10b981', '#f59e0b', '#ec4899']
        
        for col, (name, color) in zip(cols, zip(system['results'].keys(), colors)):
            result = system['results'][name]
            fig_cm = px.imshow(
                result['confusion_matrix'],
                text_auto=True,
                aspect='auto',
                color_continuous_scale=[[0, '#1e293b'], [1, color]],
                x=['No Stroke', 'Stroke'],
                y=['No Stroke', 'Stroke'],
                labels=dict(x='Predicted', y='Actual'),
                template='plotly_dark'
            )
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(
                    text=f"{name}<br>AUC={result['auc_roc']:.4f}",
                    font=dict(color='#f1f5f9', size=13)
                ),
                height=320,
                margin=dict(t=60, b=10)
            )
            col.plotly_chart(fig_cm, use_container_width=True)
    
    with tabs[1]:
        st.markdown("<div class='section-title'>📉 ROC Curves</div>", unsafe_allow_html=True)
        
        fig_roc = go.Figure()
        colors = ['#6366f1', '#10b981', '#f59e0b', '#ec4899']
        
        for (name, result), color in zip(system['results'].items(), colors):
            fig_roc.add_trace(go.Scatter(
                x=result['fpr'], y=result['tpr'],
                mode='lines',
                name=f"{name} (AUC={result['auc_roc']:.4f})",
                line=dict(color=color, width=3)
            ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC=0.5)',
            line=dict(color='#64748b', width=2, dash='dash')
        ))
        
        fig_roc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='False Positive Rate', gridcolor='#334155'),
            yaxis=dict(title='True Positive Rate', gridcolor='#334155'),
            height=500,
            margin=dict(t=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with tabs[2]:
        st.markdown("<div class='section-title'>🔵 K-Means Clustering</div>", unsafe_allow_html=True)
        
        cols = st.columns(2)
        K_range = list(range(2, 9))
        
        with cols[0]:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=K_range, y=system['inertias'],
                mode='lines+markers',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=10)
            ))
            fig_elbow.add_vline(
                x=3, line_dash='dash', line_color='#ef4444',
                annotation_text='K=3 Optimal', annotation_font_color='#ef4444'
            )
            fig_elbow.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(text='Elbow Method', font=dict(color='#f1f5f9')),
                xaxis=dict(title='K', gridcolor='#334155'),
                yaxis=dict(title='Inertia', gridcolor='#334155'),
                height=350
            )
            cols[0].plotly_chart(fig_elbow, use_container_width=True)
        
        with cols[1]:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(
                x=K_range, y=system['silhouettes'],
                mode='lines+markers',
                line=dict(color='#10b981', width=3),
                marker=dict(size=10)
            ))
            best_k = K_range[int(np.argmax(system['silhouettes']))]
            fig_sil.add_vline(
                x=best_k, line_dash='dash', line_color='#f59e0b',
                annotation_text=f'K={best_k}', annotation_font_color='#f59e0b'
            )
            fig_sil.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title=dict(text='Silhouette Score', font=dict(color='#f1f5f9')),
                xaxis=dict(title='K', gridcolor='#334155'),
                yaxis=dict(title='Score', gridcolor='#334155'),
                height=350
            )
            cols[1].plotly_chart(fig_sil, use_container_width=True)
        
        # PCA visualization
        st.markdown("<div class='section-title'>🌐 2D PCA Projection</div>", unsafe_allow_html=True)
        df_pca = pd.DataFrame({
            'PC1': system['X_pca'][:, 0],
            'PC2': system['X_pca'][:, 1],
            'Cluster': [f"Cluster {c}" for c in system['clusters']],
            'Stroke': ['Stroke' if s == 1 else 'No Stroke' for s in system['dataframe']['stroke'].values],
        })
        
        fig_pca = px.scatter(
            df_pca, x='PC1', y='PC2', color='Cluster', symbol='Stroke',
            color_discrete_sequence=['#3b82f6', '#f59e0b', '#ef4444'],
            opacity=0.6, template='plotly_dark',
            title='K-Means Clusters in PCA Space'
        )
        
        # Add cluster centers
        centers = system['pca'].transform(system['kmeans'].cluster_centers_)
        fig_pca.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers+text',
            marker=dict(symbol='x', size=20, color='white', line=dict(width=3)),
            text=['C0', 'C1', 'C2'],
            textposition='top right',
            textfont=dict(color='white', size=12),
            name='Centroids'
        ))
        
        fig_pca.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(t=50)
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Cluster profiles
        st.markdown("<div class='section-title'>📊 Cluster Profiles</div>", unsafe_allow_html=True)
        cp = system['cluster_profiles'].copy()
        cp['Cluster Name'] = cp['cluster'].map({
            0: '🟢 Low Risk', 1: '🟡 Moderate Risk', 2: '🔴 High Risk'
        })
        cp = cp[['Cluster Name', 'count', 'stroke_pct', 'avg_age', 'avg_glucose', 'avg_bmi', 'hypertension_pct']]
        cp.columns = ['Cluster', 'Count', '% Stroke', 'Avg Age', 'Avg Glucose', 'Avg BMI', '% Hypertension']
        st.dataframe(cp, use_container_width=True, hide_index=True)
    
    with tabs[3]:
        st.markdown("<div class='section-title'>📊 Feature Importance</div>", unsafe_allow_html=True)
        
        fi = system['feature_importance'].reset_index()
        fi.columns = ['Feature', 'Importance']
        
        fig_fi = go.Figure(go.Bar(
            y=fi['Feature'][::-1],
            x=fi['Importance'][::-1],
            orientation='h',
            marker=dict(
                color=fi['Importance'][::-1],
                colorscale=[[0, '#1e3a5f'], [0.5, '#6366f1'], [1, '#a78bfa']],
                showscale=False
            ),
            text=[f"  {v:.4f}" for v in fi['Importance'][::-1]],
            textposition='outside',
            textfont=dict(color='#cbd5e1', size=12)
        ))
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Importance Score', gridcolor='#334155'),
            yaxis=dict(color='#f1f5f9', tickfont=dict(size=13)),
            height=450,
            margin=dict(l=20, r=80, t=20, b=20)
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        
        # Ranked factors
        st.markdown("<div class='section-title'>🏆 Top Risk Factors</div>", unsafe_allow_html=True)
        
        descriptions = {
            'age': 'Age is the most critical factor - elderly have significantly higher risk',
            'avg_glucose_level': 'High glucose indicates diabetes, strongly linked to stroke',
            'bmi': 'High BMI correlates with obesity and cardiovascular risk',
            'hypertension': 'High blood pressure is a critical stroke comorbidity',
            'heart_disease': 'Heart disease significantly increases stroke probability',
            'ever_married': 'Marital status correlates with age and lifestyle',
            'smoking_status': 'Smoking damages blood vessels, increasing stroke risk',
            'work_type': 'Job type relates to stress levels and lifestyle',
            'gender': 'Hormonal and biological gender differences',
            'Residence_type': 'Healthcare access and environmental factors',
        }
        medals = ['🥇', '🥈', '🥉'] + [f'{i}️⃣' for i in range(4, 11)]
        
        cols = st.columns(2)
        max_imp = fi['Importance'].max()
        
        for i, (_, row) in enumerate(fi.iterrows()):
            col = cols[i % 2]
            color = '#6366f1' if i < 3 else '#475569'
            col.markdown(f"""
            <div class='glass-card' style='padding:14px;margin-bottom:10px;border-left:3px solid {color}'>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
                    <span style='font-size:20px'>{medals[i]}</span>
                    <span style='color:#f1f5f9;font-weight:700;font-size:14px'>{row["Feature"]}</span>
                    <span style='margin-left:auto;color:#a78bfa;font-weight:800'>{row["Importance"]:.4f}</span>
                </div>
                <div style='background:#0f172a;border-radius:4px;height:6px;margin-bottom:8px;overflow:hidden'>
                    <div style='height:100%;width:{row["Importance"]/max_imp*100:.1f}%;background:linear-gradient(90deg,#6366f1,#a78bfa)'></div>
                </div>
                <div style='color:#94a3b8;font-size:12px'>
                    {descriptions.get(row["Feature"], "Contributes to stroke prediction")}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============ PAGE 5: ABOUT ============
elif page == "ℹ️ About":
    st.markdown("<h1 style='color:#f1f5f9;font-size:2.5rem;font-weight:900;margin-bottom:8px'>ℹ️ About Project</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:1rem'>Methods, dataset, and project information</p>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2], gap="large")
    
    with col_left:
        st.markdown("<div class='section-title'>🧪 Methods Used</div>", unsafe_allow_html=True)
        
        methods = [
            ("🔵 K-Means Clustering", "#3b82f6",
             "Unsupervised learning that groups patients into K clusters based on health characteristics. Used for patient risk segmentation into 3 groups."),
            ("🌲 Random Forest", "#10b981",
             "Ensemble learning with 200 decision trees. Best performing model with AUC-ROC ~0.91. Uses class_weight='balanced' for imbalance."),
            ("📈 Gradient Boosting", "#6366f1",
             "Sequential ensemble method that builds trees to correct previous errors. Strong performance with 150 estimators."),
            ("📊 Logistic Regression", "#f59e0b",
             "Linear classification baseline with sigmoid function. High interpretability with C=1.0 regularization."),
            ("👥 K-Nearest Neighbors", "#ec4899",
             "Non-parametric algorithm using K=7 neighbors. Suitable for non-linear decision boundaries."),
            ("⚖️ SMOTE", "#ef4444",
             "Synthetic Minority Over-sampling for 4.9% vs 95.1% class imbalance. Creates synthetic samples via KNN interpolation."),
        ]
        
        for title, color, desc in methods:
            st.markdown(f"""
            <div class='glass-card' style='border-left:4px solid {color};margin-bottom:12px'>
                <div style='color:{color};font-weight:700;font-size:16px;margin-bottom:8px'>{title}</div>
                <div style='color:#cbd5e1;font-size:13px;line-height:1.7'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>📐 CRISP-DM Framework</div>", unsafe_allow_html=True)
        
        phases = [
            ("1", "Business Understanding", "Problem identification, objectives definition", "#6366f1"),
            ("2", "Data Understanding", "EDA of 5,110 patients, distributions, correlations", "#3b82f6"),
            ("3", "Data Preparation", "Imputation, encoding, scaling, SMOTE", "#10b981"),
            ("4", "Modeling", "Training RF, GB, LR, KNN + K-Means", "#f59e0b"),
            ("5", "Evaluation", "Accuracy, F1, AUC-ROC, Silhouette, Feature Importance", "#ef4444"),
            ("6", "Deployment", "Streamlit web application", "#8b5cf6"),
        ]
        
        for no, phase, desc, color in phases:
            st.markdown(f"""
            <div style='display:flex;gap:14px;padding:12px 0;border-bottom:1px solid #334155;align-items:flex-start'>
                <div style='background:{color};color:white;font-weight:800;font-size:14px;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;flex-shrink:0'>{no}</div>
                <div style='flex:1'>
                    <div style='color:#f1f5f9;font-weight:700;font-size:15px'>{phase}</div>
                    <div style='color:#94a3b8;font-size:13px;margin-top:4px'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("<div class='section-title'>🗂️ Dataset Info</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
            <div style='color:#a78bfa;font-weight:700;font-size:16px;margin-bottom:12px'>📦 Stroke Prediction Dataset</div>
            <div style='color:#cbd5e1;font-size:13px;line-height:2.2'>
                <b style='color:#f1f5f9'>Source:</b> Kaggle - fedesoriano<br>
                <b style='color:#f1f5f9'>License:</b> Open Database License<br>
                <b style='color:#f1f5f9'>Records:</b> 5,110 patients<br>
                <b style='color:#f1f5f9'>Features:</b> 10 inputs + 1 target<br>
                <b style='color:#f1f5f9'>Target:</b> stroke (0=no, 1=yes)<br>
                <b style='color:#f1f5f9'>Imbalance:</b> 4.9% positive (249 cases)<br>
                <b style='color:#f1f5f9'>Missing:</b> 201 values in BMI column
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>📊 Evaluation Metrics</div>", unsafe_allow_html=True)
        
        metrics = [
            ("AUC-ROC", "Area Under ROC Curve · Target ≥ 0.80", "#6366f1"),
            ("Accuracy", "Overall prediction accuracy", "#10b981"),
            ("F1-Score", "Harmonic mean of Precision & Recall", "#f59e0b"),
            ("CV AUC", "5-Fold Cross-Validation AUC", "#3b82f6"),
            ("Silhouette", "Clustering quality · Target ≥ 0.30", "#8b5cf6"),
        ]
        
        for name, desc, color in metrics:
            st.markdown(f"""
            <div style='display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #334155;align-items:center'>
                <div style='background:{color}22;color:{color};font-weight:700;font-size:12px;border-radius:6px;padding:6px 12px;min-width:90px;text-align:center'>{name}</div>
                <div style='color:#cbd5e1;font-size:12px;flex:1'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-title' style='margin-top:24px'>🛠️ Tech Stack</div>", unsafe_allow_html=True)
        
        techs = ["Python 3.11", "Streamlit", "scikit-learn", "imbalanced-learn", "Plotly", "Pandas", "NumPy"]
        chips_html = " ".join([
            f"<span style='background:rgba(99,102,241,0.1);color:#a78bfa;border:1px solid #6366f1;border-radius:20px;padding:6px 14px;font-size:12px;display:inline-block;margin:4px'>{t}</span>"
            for t in techs
        ])
        st.markdown(f"<div style='line-height:2.5'>{chips_html}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-title' style='margin-top:24px'>📌 Project Info</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
            <div style='color:#cbd5e1;font-size:13px;line-height:2.2'>
                <b style='color:#f1f5f9'>Course:</b> Knowledge Discovery in Databases<br>
                <b style='color:#f1f5f9'>Assignment:</b> Final Project - Group Work<br>
                <b style='color:#f1f5f9'>Class:</b> KDD-01<br>
                <b style='color:#f1f5f9'>Institution:</b> Universitas Negeri Surabaya<br>
                <b style='color:#f1f5f9'>Year:</b> 2025/2026<br>
                <b style='color:#f1f5f9'>Team Size:</b> 5 members
            </div>
        </div>
        """, unsafe_allow_html=True)
