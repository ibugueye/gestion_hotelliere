import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
# Import statsmodels pour ARIMA. 
# NOTE: Nécessite l'installation via pip install statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    st.error("La bibliothèque 'statsmodels' est requise pour ARIMA. Veuillez l'installer.")
    ARIMA = None

# Configuration de la page
st.set_page_config(page_title="HotelCorp Analytics Suite", layout="wide", initial_sidebar_state="expanded")

# Styles CSS personnalisés pour un look "Enterprise" et Navigation
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #444;
    }
    /* Style pour la navigation radio horizontale */
    div.row-widget.stRadio > div {
        flex-direction: row;
        justify-content: space-between;
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #444;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #0E1117;
        padding: 10px 20px;
        border-radius: 5px;
        border: 1px solid #333;
        margin: 0 5px;
        transition: all 0.3s;
        flex-grow: 1;
        text-align: center;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        border-color: #FF4B4B;
        color: #FF4B4B;
    }
    .stAlert {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #444;
    }
    /* Masquer le label par défaut des radios */
    div.row-widget.stRadio > label {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- FONCTION DE PRÉVISION (ARIMA) ---
def calculate_arima_forecast(series, periods=90):
    """
    Utilise un modèle ARIMA (AutoRegressive Integrated Moving Average)
    pour des prévisions plus robustes avec intervalles de confiance.
    """
    if ARIMA is None:
        return None, None, None

    try:
        # Configuration ARIMA(p,d,q). (5,1,0) est un bon point de départ pour des séries journalières/hebdo
        # p=5 (lags), d=1 (differencing pour stationnarité), q=0 (moving average)
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        
        # Prévision
        forecast_result = model_fit.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean
        
        # Intervalle de confiance (95%)
        conf_int = forecast_result.conf_int(alpha=0.05)
        
        # Nettoyage des index pour aligner avec les dates futures
        return forecast_mean, conf_int.iloc[:, 0], conf_int.iloc[:, 1] # mean, lower, upper
        
    except Exception as e:
        st.warning(f"Le modèle ARIMA n'a pas pu converger : {e}. Affichage des données brutes uniquement.")
        return None, None, None

# --- FONCTION DE CHARGEMENT DE DONNÉES ---
@st.cache_data
def generate_simulation():
    """Génère des données simulées si aucun fichier n'est fourni"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    
    data = pd.DataFrame(index=dates)
    data['Occupancy_Rate'] = np.random.uniform(0.55, 0.95, size=len(dates)) + (np.sin(np.arange(len(dates))/30) * 0.1)
    data['ADR'] = np.random.uniform(150, 300, size=len(dates)) + (data['Occupancy_Rate'] * 50)
    data['Available_Rooms'] = 200
    return data

@st.cache_data
def process_data(df_input):
    """Calcule les métriques dérivées et les KPIs financiers"""
    df = df_input.copy()
    
    # Calculs opérationnels de base si absents
    if 'RevPAR' not in df.columns:
        df['RevPAR'] = df['Occupancy_Rate'] * df['ADR']
    if 'Sold_Rooms' not in df.columns:
        df['Sold_Rooms'] = (df['Available_Rooms'] * df['Occupancy_Rate']).astype(int)
    if 'Room_Revenue' not in df.columns:
        df['Room_Revenue'] = df['Sold_Rooms'] * df['ADR']
    
    # Simulation des centres de profit si données manquantes (pour la démo)
    if 'FnB_Revenue' not in df.columns:
        df['FnB_Revenue'] = df['Sold_Rooms'] * np.random.uniform(40, 80, size=len(df))
    if 'Spa_Revenue' not in df.columns:
        df['Spa_Revenue'] = df['Sold_Rooms'] * np.random.uniform(10, 30, size=len(df))
    if 'Events_Revenue' not in df.columns:
        df['Events_Revenue'] = np.random.choice([0, 5000, 15000, 30000], size=len(df), p=[0.7, 0.2, 0.08, 0.02])
    
    # Gestion des Autres Revenus (si non présents dans le CSV principal ou fusionnés)
    if 'Other_Revenue' not in df.columns:
        # Simulation : Parking, Commissions, etc. (~5% du Room Rev)
        df['Other_Revenue'] = df['Room_Revenue'] * np.random.uniform(0.03, 0.08, size=len(df))
    
    df['Total_Revenue'] = df['Room_Revenue'] + df['FnB_Revenue'] + df['Spa_Revenue'] + df['Events_Revenue'] + df['Other_Revenue']
    
    # Dépenses (USALI Model)
    if 'COGS' not in df.columns:
        df['COGS'] = df['Total_Revenue'] * 0.25
    if 'Payroll' not in df.columns:
        df['Payroll'] = 15000 + (df['Sold_Rooms'] * 20) 
    if 'Other_Expenses' not in df.columns:
        df['Other_Expenses'] = 5000 + (df['Total_Revenue'] * 0.1)
    
    df['GOP'] = df['Total_Revenue'] - df['COGS'] - df['Payroll'] - df['Other_Expenses']
    df['EBITDA'] = df['GOP'] - (df['Total_Revenue'] * 0.05)
    
    return df

# --- SIDEBAR & CONFIGURATION ---
with st.sidebar:
    st.title("🏨 HotelCorp 360")
    st.caption("Système de Contrôle de Gestion Unifié")
    
    st.header("Sources de Données")
    uploaded_file = st.file_uploader("1. Données Principales (CSV)", type=['csv'], key="main_csv")
    uploaded_extras = st.file_uploader("2. Autres Revenus (CSV)", type=['csv'], key="extra_csv", help="Colonnes requises: Date, Other_Revenue")
    
    use_simulation = True
    df_raw = None
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Validation des colonnes requises
            required_cols = ['Date', 'Occupancy_Rate', 'ADR', 'Available_Rooms']
            missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
            
            if missing_cols:
                st.error(f"⚠️ Colonnes manquantes (Main): {', '.join(missing_cols)}")
                st.warning("Retour aux données simulées.")
            else:
                df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'])
                df_uploaded = df_uploaded.set_index('Date')
                df_raw = df_uploaded.sort_index()
                use_simulation = False
                st.success("✅ Données principales chargées !")
                
        except Exception as e:
            st.error(f"Erreur fichier principal : {e}")

    # Gestion du fichier Autres Revenus
    if df_raw is not None and uploaded_extras is not None:
        try:
            df_extras = pd.read_csv(uploaded_extras)
            if 'Date' in df_extras.columns and 'Other_Revenue' in df_extras.columns:
                df_extras['Date'] = pd.to_datetime(df_extras['Date'])
                df_extras = df_extras.set_index('Date')
                # Fusion (Join)
                df_raw = df_raw.join(df_extras['Other_Revenue'], how='left').fillna(0)
                st.success("✅ Autres revenus fusionnés !")
            else:
                st.error("⚠️ Le CSV 'Autres Revenus' doit contenir 'Date' et 'Other_Revenue'.")
        except Exception as e:
            st.error(f"Erreur fichier extras : {e}")

    if use_simulation:
        df_raw = generate_simulation()
        if not uploaded_file:
            st.info("💡 Mode Démo : Données simulées actives.")

    # --- FILTRES TEMPORELS ---
    st.divider()
    st.header("📅 Période d'Analyse")
    
    # Récupération des bornes min/max des données
    min_date = df_raw.index.min().date()
    max_date = df_raw.index.max().date()
    
    col_filter1, col_filter2 = st.columns(2)
    
    # Sélecteur de date interactif
    date_range = st.date_input(
        "Sélectionnez une plage",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY"
    )
    
    # Logique de filtrage du DataFrame
    start_date, end_date = min_date, max_date
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_raw.index.date >= start_date) & (df_raw.index.date <= end_date)
        df_filtered = df_raw.loc[mask]
    else:
        st.warning("Veuillez sélectionner une date de début et de fin.")
        df_filtered = df_raw # Fallback

    st.caption(f"Données du {start_date} au {end_date}")

# Traitement final des données (sur le dataset filtré)
df = process_data(df_filtered)
revenue_split = {
    'Hébergement': df['Room_Revenue'].sum(),
    'F&B': df['FnB_Revenue'].sum(),
    'Spa': df['Spa_Revenue'].sum(),
    'Events': df['Events_Revenue'].sum(),
    'Autres': df['Other_Revenue'].sum()
}

# --- MAIN CONTENT ---
st.title("📊 Tableau de Bord Exécutif & Performance")

if df.empty:
    st.error("Aucune donnée disponible pour la période sélectionnée.")
else:
    # KPIs Haut Niveau
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    current_rev = df['Total_Revenue'].sum()
    current_gop = df['GOP'].sum()
    current_occ = df['Occupancy_Rate'].mean()
    current_adr = df['ADR'].mean()

    with kpi1:
        st.metric("Chiffre d'Affaires Total", f"{current_rev/1e6:.2f} M€", "Consolidé")
    with kpi2:
        st.metric("GOP (Marge Ops)", f"{current_gop/1e6:.2f} M€", f"{(current_gop/current_rev)*100:.1f}% Marge")
    with kpi3:
        st.metric("Taux d'Occupation (TO)", f"{current_occ*100:.1f}%", "Moyenne Période")
    with kpi4:
        st.metric("ADR (Prix Moyen)", f"{current_adr:.0f} €", "Moyenne Période")

    st.divider()
    
    # --- NAVIGATION WORKFLOW ---
    st.markdown("### 🧭 Cycle d'Analyse Stratégique")
    
    # Définition des étapes
    steps_options = [
        "1. 📈 Revenue Management", 
        "2. 🍽️ Centres de Profit", 
        "3. 💰 États Financiers", 
        "4. ⚖️ Ratios & KPI"
    ]
    
    # Navigation Radio Horizontale
    selected_step = st.radio("Navigation", steps_options, horizontal=True, index=0)
    
    # Barre de progression
    current_step_index = steps_options.index(selected_step)
    progress_value = (current_step_index + 1) / len(steps_options)
    st.progress(progress_value)
    
    st.markdown("---")

    # --- CONTENU CONDITIONNEL ---

    if selected_step == "1. 📈 Revenue Management":
        st.subheader("Analyse du Revenue Management (RevPAR & TO)")
        st.caption("Performance des ventes de chambres et politique tarifaire.")
        
        # --- SECTION PRÉVISION ---
        st.markdown("#### 🔮 Projection & Prévisions (ARIMA)")
        col_forecast_opt, col_forecast_viz = st.columns([1, 3])
        
        with col_forecast_opt:
            st.info("Le modèle utilise l'algorithme ARIMA pour identifier les tendances auto-régressives et projeter les revenus futurs.")
            enable_forecast = st.toggle("Activer Prévision (3 mois)", value=False)
            if enable_forecast:
                st.caption("Le calcul peut prendre quelques secondes...")
            
        with col_forecast_viz:
            # Préparation des données pour le graphique de revenu
            # Utilisation de données hebdo pour la stabilité du modèle ARIMA sur des séries courtes
            df_chart = df.resample('W').sum() if len(df) > 90 else df.resample('D').sum() 
            
            fig_forecast = go.Figure()
            
            # Données Historiques
            fig_forecast.add_trace(go.Scatter(
                x=df_chart.index, 
                y=df_chart['Total_Revenue'], 
                mode='lines',
                name='Revenu Historique',
                line=dict(color='#00CC96', width=3)
            ))
            
            if enable_forecast:
                future_periods = 12 if len(df) > 90 else 90 
                freq = 'W' if len(df) > 90 else 'D'
                last_date = df_chart.index.max()
                future_dates = [last_date + timedelta(weeks=x) if freq == 'W' else last_date + timedelta(days=x) for x in range(1, future_periods + 1)]
                
                # Calcul ARIMA
                forecast_values, lower_conf, upper_conf = calculate_arima_forecast(df_chart['Total_Revenue'], periods=future_periods)
                
                if forecast_values is not None:
                    # 1. Zone de Confiance (Ribbon)
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates + future_dates[::-1], # Aller-retour pour fermer le polygone
                        y=list(upper_conf) + list(lower_conf)[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 75, 75, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalle de Confiance (95%)',
                        hoverinfo="skip"
                    ))
                    
                    # 2. Ligne de Prévision Moyenne
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates, 
                        y=forecast_values, 
                        mode='lines',
                        name='Prévision ARIMA',
                        line=dict(color='#FF4B4B', width=2, dash='dash')
                    ))

            fig_forecast.update_layout(
                title="Projection du Chiffre d'Affaires (Modèle ARIMA)",
                xaxis_title="Date",
                yaxis_title="Revenu Total (€)",
                hovermode="x unified",
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        st.divider()

        # Graphique TO vs ADR existant
        st.subheader("Corrélation Prix vs Volume")
        fig_rev = go.Figure()
        # Resampling pour lisibilité si trop de données
        df_chart_kpi = df.resample('M').mean() if len(df) > 90 else df
        
        fig_rev.add_trace(go.Bar(name='Taux d\'Occupation', x=df_chart_kpi.index, y=df_chart_kpi['Occupancy_Rate'], yaxis='y2', marker_color='#636EFA', opacity=0.6))
        fig_rev.add_trace(go.Scatter(name='ADR (€)', x=df_chart_kpi.index, y=df_chart_kpi['ADR'], line=dict(color='white', width=3)))
        
        fig_rev.update_layout(
            title="Évolution TO vs ADR",
            yaxis=dict(title='ADR (€)'),
            yaxis2=dict(title='Occupation (%)', overlaying='y', side='right', range=[0, 1]),
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_rev, use_container_width=True)
        
        # Ajout Pie Chart ici aussi pour contexte
        col_mix1, col_mix2 = st.columns([1, 3])
        with col_mix1:
             st.info("💡 **Conseil IA** : Une hausse de l'ADR combinée à une baisse du TO peut indiquer une résistance tarifaire. Surveillez le RevPAR.")

    elif selected_step == "2. 🍽️ Centres de Profit":
        st.subheader("Détail par Centre de Profit")
        st.caption("Contribution de chaque département à la rentabilité globale.")
        
        col_profit1, col_profit2 = st.columns([1, 2])
        
        with col_profit1:
            # Donut chart
            fig_donut = px.pie(values=list(revenue_split.values()), names=list(revenue_split.keys()), hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_donut.update_layout(showlegend=True, title="Mix Revenu (Détail)", font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_profit2:
            # Données pour le graphique comparatif
            dept_data = pd.DataFrame({
                'Département': list(revenue_split.keys()),
                'Revenu': list(revenue_split.values()),
                'Marge Directe (%)': [85, 35, 60, 50, 90] # Hypothèses standard hôtellerie, Autres = Parking/Commissions (marge haute)
            })
            dept_data['Marge Directe (€)'] = dept_data['Revenu'] * (dept_data['Marge Directe (%)'] / 100)
            
            # Transformation format long pour Plotly (Barres groupées)
            dept_long = dept_data.melt(id_vars='Département', value_vars=['Revenu', 'Marge Directe (€)'], var_name='Métrique', value_name='Montant')
            
            fig_bar = px.bar(
                dept_long, 
                x='Département', 
                y='Montant', 
                color='Métrique',
                barmode='group', 
                title="Comparaison : Revenu vs Marge Directe",
                color_discrete_map={'Revenu': '#636EFA', 'Marge Directe (€)': '#00CC96'}
            )
            fig_bar.update_layout(
                font=dict(color='white'), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("💡 **Note d'analyse** : Le département 'Hébergement' a généralement la marge la plus élevée (gros GOP), tandis que le F&B génère du volume mais avec des coûts plus élevés.")

    elif selected_step == "3. 💰 États Financiers":
        st.subheader("Analyse Financière Détaillée")
        st.caption("Compte de Résultat (P&L) consolidé et tendances des coûts.")
        
        pl_data = {
            'Poste': ['Total Revenu', 'Coût des Ventes (COGS)', 'Marge Brute', 'Dépenses Personnels', 'Autres Dépenses', 'GOP (RBE)', 'EBITDA'],
            'Montant (€)': [
                df['Total_Revenue'].sum(),
                -df['COGS'].sum(),
                (df['Total_Revenue'].sum() - df['COGS'].sum()),
                -df['Payroll'].sum(),
                -df['Other_Expenses'].sum(),
                df['GOP'].sum(),
                df['EBITDA'].sum()
            ]
        }
        
        # 1. Waterfall Chart
        fig_waterfall = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "total", "relative", "relative", "total", "total"],
            x = pl_data['Poste'],
            textposition = "outside",
            y = pl_data['Montant (€)'],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(title = "Cascade de Profitabilité (P&L)", showlegend = False, font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.divider()
        
        # 2. Trends Chart
        st.subheader("Tendances des Revenus et Dépenses")
        
        # Calcul des dépenses totales pour le graphique
        df['Total_Expenses'] = df['COGS'] + df['Payroll'] + df['Other_Expenses']
        
        # Resampling mensuel pour une meilleure lisibilité si beaucoup de données
        df_trends = df.resample('W').sum() if len(df) > 90 else df

        fig_trends = go.Figure()
        
        # Ligne Revenus
        fig_trends.add_trace(go.Scatter(
            x=df_trends.index, 
            y=df_trends['Total_Revenue'], 
            name='Revenu Total',
            line=dict(color='#00CC96', width=3)
        ))

        # Ligne Dépenses
        fig_trends.add_trace(go.Scatter(
            x=df_trends.index, 
            y=df_trends['Total_Expenses'], 
            name='Dépenses Totales',
            line=dict(color='#EF553B', width=3)
        ))

        # Zone GOP
        fig_trends.add_trace(go.Scatter(
            x=df_trends.index, 
            y=df_trends['GOP'], 
            name='GOP (Marge)',
            fill='tozeroy',
            line=dict(color='#636EFA', width=1),
            opacity=0.2
        ))

        fig_trends.update_layout(
            title="Évolution Temporelle : Chiffre d'Affaires vs Coûts",
            xaxis_title="Date",
            yaxis_title="Montant (€)",
            hovermode="x unified",
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=1.1)
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)

    elif selected_step == "4. ⚖️ Ratios & KPI":
        st.subheader("Ratios de Gestion & KPI Financiers")
        
        # --- FILTRES LOCAUX INTERACTIFS ---
        with st.expander("⚙️ Configuration de l'Analyse (Filtres & Segments)", expanded=True):
            st.markdown("Affinez les indicateurs ci-dessous sans affecter le tableau de bord principal.")
            col_kf1, col_kf2 = st.columns(2)
            
            with col_kf1:
                # Filtre de date local (Hérite des bornes globales par défaut)
                kpi_dates = st.date_input(
                    "📅 Période Spécifique (Zoom)",
                    value=(start_date, end_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="kpi_date_input"
                )
            
            with col_kf2:
                # Filtre par catégorie de jour (Segmentation)
                day_segment = st.selectbox(
                    "📊 Segment d'Activité", 
                    ["Tous les jours", "Semaine (Affaires / Corporate)", "Week-end (Loisirs)"],
                    key="kpi_segment"
                )
                
        # --- LOGIQUE DE FILTRAGE LOCAL ---
        df_kpi = df.copy()
        
        # 1. Filtre Date
        if isinstance(kpi_dates, tuple) and len(kpi_dates) == 2:
            ks, ke = kpi_dates
            df_kpi = df_kpi.loc[(df_kpi.index.date >= ks) & (df_kpi.index.date <= ke)]
            
        # 2. Filtre Segment
        if day_segment == "Semaine (Affaires / Corporate)":
            df_kpi = df_kpi[df_kpi.index.dayofweek < 4] # Lundi=0, Jeudi=3
        elif day_segment == "Week-end (Loisirs)":
            df_kpi = df_kpi[df_kpi.index.dayofweek >= 4] # Vendredi=4, Dimanche=6

        if df_kpi.empty:
            st.warning("⚠️ Aucune donnée ne correspond à vos critères de filtrage.")
        else:
            # Calculs KPIs sur données filtrées
            total_rev_kpi = df_kpi['Total_Revenue'].sum()
            avail_rooms_kpi = df_kpi['Available_Rooms'].sum()
            
            personnel_cost_ratio = (df_kpi['Payroll'].sum() / total_rev_kpi * 100) if total_rev_kpi > 0 else 0
            goppar = (df_kpi['GOP'].sum() / avail_rooms_kpi) if avail_rooms_kpi > 0 else 0
            fnb_capture = (df_kpi['FnB_Revenue'].sum() / df_kpi['Room_Revenue'].sum() * 100) if df_kpi['Room_Revenue'].sum() > 0 else 0
            
            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            
            with r1:
                st.markdown("### 👥 Payroll %")
                st.metric("Coût Personnel / CA", f"{personnel_cost_ratio:.1f}%")
                st.progress(min(personnel_cost_ratio/100, 1.0))
                st.caption(f"Cible: < 35% | Basé sur {len(df_kpi)} jours")
                
            with r2:
                st.markdown("### 🏨 GOPPAR")
                st.metric("GOP par chambre", f"{goppar:.2f} €")
                # Petit graph sparkline
                st.area_chart(df_kpi['GOP'] / df_kpi['Available_Rooms'], height=80, color="#00CC96")
                
            with r3:
                st.markdown("### 🍽️ Capture F&B")
                st.metric("F&B sur Room Rev", f"{fnb_capture:.1f}%")
                st.caption(f"Segment: {day_segment}")
            
            st.divider()
            st.subheader("🏗️ Structure Financière & Solvabilité (Placeholders)")
            
            fin1, fin2, fin3 = st.columns(3)
            
            with fin1:
                st.metric(label="Ratio d'Endettement", value="--", delta="Données Bilan Req.", delta_color="off")
            with fin2:
                 st.metric(label="Return on Equity (ROE)", value="--", delta="Données Bilan Req.", delta_color="off")
            with fin3:
                 st.metric(label="Liquidité Générale", value="--", delta="Données Bilan Req.", delta_color="off")
            
            st.info("ℹ️ Connectez votre module 'Bilan Comptable' pour activer ces ratios en temps réel.")

            st.divider()
            # --- FEATURE PARTAGE ---
            col_share1, col_share2 = st.columns([3, 1])
            with col_share1:
                st.markdown("### 📤 Partager l'analyse")
                st.caption("Générez un lien incluant vos filtres actuels (dates et segmentation) pour partager cette vue avec un collègue.")
            
            with col_share2:
                if st.button("📋 Générer un lien de partage", type="primary"):
                    # Récupération des paramètres
                    p_start = kpi_dates[0].strftime("%Y-%m-%d") if isinstance(kpi_dates, tuple) and len(kpi_dates) > 0 else ""
                    p_end = kpi_dates[1].strftime("%Y-%m-%d") if isinstance(kpi_dates, tuple) and len(kpi_dates) > 1 else ""
                    
                    # Mise à jour des query params (URL bar)
                    st.query_params["start"] = p_start
                    st.query_params["end"] = p_end
                    st.query_params["segment"] = day_segment
                    
                    st.success("Lien généré dans la barre d'adresse !")
                    st.code(f"?start={p_start}&end={p_end}&segment={day_segment}", language="text")
