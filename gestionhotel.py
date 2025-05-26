import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import openpyxl
from io import BytesIO

# Configuration de la page
st.set_page_config(page_title="Dashboard Hôtelier", layout="wide", page_icon="🏨")

# Titre de l'application
st.title("📊 Dashboard de Contrôle de Gestion Hôtelier")

# Fonction pour charger les données (simulée pour l'exemple)
@st.cache_data
def load_data():
    # Génération de données fictives
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
    occupancy = np.random.randint(30, 100, size=len(dates))
    adr = np.random.uniform(100, 300, size=len(dates))
    revenue = occupancy * adr
    
    hotel_data = pd.DataFrame({
        'Date': dates,
        'Occupancy': occupancy,
        'ADR': adr,
        'Revenue': revenue,
        'Restaurant': np.random.uniform(2000, 8000, size=len(dates)),
        'Bar': np.random.uniform(1000, 5000, size=len(dates)),
        'Spa': np.random.uniform(500, 3000, size=len(dates)),
        'Budget_Revenue': revenue * np.random.uniform(0.9, 1.1, size=len(dates)),
        'Budget_Occupancy': occupancy * np.random.uniform(0.9, 1.1, size=len(dates))
    })
    
    return hotel_data

# Chargement des données
df = load_data()
df['Month'] = df['Date'].dt.month_name()
df['Year'] = df['Date'].dt.year

# Sidebar - Paramètres
st.sidebar.header("Paramètres")
selected_year = st.sidebar.selectbox("Année", options=df['Year'].unique())
selected_metric = st.sidebar.selectbox(
    "KPI Principal", 
    options=['Occupancy', 'ADR', 'Revenue', 'RevPAR', 'GOPPAR']
)

# Calcul des KPIs
monthly_data = df.groupby('Month').agg({
    'Occupancy': 'mean',
    'ADR': 'mean',
    'Revenue': 'sum',
    'Restaurant': 'sum',
    'Bar': 'sum',
    'Spa': 'sum',
    'Budget_Revenue': 'sum',
    'Budget_Occupancy': 'mean'
}).reset_index()

monthly_data['RevPAR'] = monthly_data['Revenue'] / 30  # Supposons 30 chambres
monthly_data['GOPPAR'] = (monthly_data['Revenue'] * 0.7) / 30  # Marge brute simplifiée

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["Tableau de bord", "Analyse Budget/Réel", "Prévisions", "Analyse par Service"])

with tab1:
    st.header("Tableau de bord global")
    
    # KPIs clés
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Taux d'occupation", f"{df['Occupancy'].mean():.1f}%", "2.5% vs budget")
    with col2:
        st.metric("ADR", f"${df['ADR'].mean():.2f}", "-1.2% vs budget")
    with col3:
        st.metric("RevPAR", f"${monthly_data['RevPAR'].mean():.2f}", "3.1% vs budget")
    with col4:
        st.metric("GOPPAR", f"${monthly_data['GOPPAR'].mean():.2f}", "1.8% vs budget")
    
    # Graphique principal
    fig = px.line(monthly_data, x='Month', y=selected_metric, 
                 title=f"Évolution du {selected_metric} par mois")
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap d'occupation
    heatmap_data = df.pivot_table(values='Occupancy', index=df['Date'].dt.day, 
                                 columns=df['Date'].dt.month_name(), aggfunc='mean')
    fig = px.imshow(heatmap_data, labels=dict(x="Mois", y="Jour", color="Taux d'occupation"),
                   title="Heatmap d'occupation par jour et mois")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Analyse Budget vs Réel")
    
    # Sélection du service à analyser
    service = st.selectbox("Service à analyser", 
                          options=['Hébergement', 'Restaurant', 'Bar', 'Spa'])
    
    if service == 'Hébergement':
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_data['Month'], y=monthly_data['Revenue'], 
                            name='Revenu réel'))
        fig.add_trace(go.Bar(x=monthly_data['Month'], y=monthly_data['Budget_Revenue'], 
                            name='Budget'))
        fig.update_layout(barmode='group', title="Revenu hébergement: Budget vs Réel")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcul des écarts
        monthly_data['Ecart'] = monthly_data['Revenue'] - monthly_data['Budget_Revenue']
        monthly_data['Ecart_pct'] = (monthly_data['Ecart'] / monthly_data['Budget_Revenue']) * 100
        
        fig = px.bar(monthly_data, x='Month', y='Ecart_pct', 
                    title="Écart en pourcentage entre budget et réel")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Prévisions d'occupation et revenus")
    
    # Choix du modèle
    model_type = st.radio("Modèle de prévision", 
                         options=['ARIMA', 'Prophet'], horizontal=True)
    
    # Préparation des données pour les modèles
    ts_data = df.groupby('Date')['Occupancy'].mean().reset_index()
    ts_data.columns = ['ds', 'y']
    
    if model_type == 'ARIMA':
        st.write("Prévision avec modèle ARIMA (3 mois)")
        model = ARIMA(ts_data['y'], order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=90)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name='Historique'))
        fig.add_trace(go.Scatter(x=pd.date_range(start=ts_data['ds'].iloc[-1], periods=91)[1:], 
                                y=forecast, name='Prévision'))
        fig.update_layout(title="Prévision d'occupation avec ARIMA")
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Prophet
        st.write("Prévision avec Prophet (3 mois)")
        model = Prophet()
        model.fit(ts_data)
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)
        
        fig = model.plot(forecast)
        st.pyplot(fig)
        
        # Composantes de la prévision
        st.subheader("Composantes de la prévision")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

with tab4:
    st.header("Analyse par point de vente")
    
    # Répartition des revenus
    revenue_by_service = monthly_data[['Restaurant', 'Bar', 'Spa']].sum()
    fig = px.pie(revenue_by_service, values=revenue_by_service.values, 
                 names=revenue_by_service.index, title="Répartition des revenus par service")
    st.plotly_chart(fig, use_container_width=True)
    
    # Evolution des différents services
    fig = go.Figure()
    for service in ['Restaurant', 'Bar', 'Spa']:
        fig.add_trace(go.Scatter(x=monthly_data['Month'], y=monthly_data[service], 
                                mode='lines+markers', name=service))
    fig.update_layout(title="Evolution des revenus par service")
    st.plotly_chart(fig, use_container_width=True)

# Téléchargement des rapports
st.sidebar.header("Export des données")
if st.sidebar.button("Générer rapport PDF"):
    # Ici on générerait un PDF avec les principaux graphiques et indicateurs
    st.sidebar.success("Rapport généré (simulation)")

if st.sidebar.button("Exporter les KPIs en Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        monthly_data.to_excel(writer, sheet_name='KPIs Mensuels')
        df.to_excel(writer, sheet_name='Données Journalières')
    st.sidebar.download_button(
        label="Télécharger le fichier Excel",
        data=output.getvalue(),
        file_name="kpis_hotel.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Documentation
st.sidebar.header("Aide")
st.sidebar.info("""
**Indicateurs clés:**
- **ADR**: Average Daily Rate (Revenu moyen par chambre occupée)
- **RevPAR**: Revenue per Available Room (Revenu par chambre disponible)
- **GOPPAR**: Gross Operating Profit per Available Room
""")