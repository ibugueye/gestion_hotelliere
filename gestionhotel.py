import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import openpyxl
from io import BytesIO
from datetime import datetime

# Configuration de la page
st.set_page_config(page_title="Dashboard Hôtelier", layout="wide", page_icon="🏨")

# Titre de l'application
st.title("📊 Dashboard de Contrôle de Gestion Hôtelière")

# Fonction pour charger les données (simulée pour l'exemple)
@st.cache_data
def load_data():
    # Génération de données fictives pour plusieurs années
    years = [2023, 2024, 2025]
    all_data = []
    
    for year in years:
        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
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
            'Other_Services': np.random.uniform(300, 2000, size=len(dates)),
            'Budget_Revenue': revenue * np.random.uniform(0.9, 1.1, size=len(dates)),
            'Budget_Occupancy': occupancy * np.random.uniform(0.9, 1.1, size=len(dates)),
            'Budget_Restaurant': np.random.uniform(2000, 8000, size=len(dates)) * np.random.uniform(0.9, 1.1, size=len(dates)),
            'Budget_Bar': np.random.uniform(1000, 5000, size=len(dates)) * np.random.uniform(0.9, 1.1, size=len(dates)),
            'Budget_Spa': np.random.uniform(500, 3000, size=len(dates)) * np.random.uniform(0.9, 1.1, size=len(dates)),
            'Budget_Other': np.random.uniform(300, 2000, size=len(dates)) * np.random.uniform(0.9, 1.1, size=len(dates)),
            'Year': year
        })
        
        all_data.append(hotel_data)
    
    return pd.concat(all_data)

# Chargement des données
df = load_data()
df['Month'] = df['Date'].dt.month_name()
df['Year'] = df['Date'].dt.year

# Sidebar - Paramètres
st.sidebar.header("Paramètres")
selected_year = st.sidebar.selectbox("Année d'exercice", options=sorted(df['Year'].unique(), reverse=True))
selected_metric = st.sidebar.selectbox(
    "KPI Principal", 
    options=['Occupancy', 'ADR', 'Revenue', 'RevPAR', 'GOPPAR']
)

# Filtrer les données selon l'année sélectionnée
df_year = df[df['Year'] == selected_year]

# Calcul des KPIs pour l'année sélectionnée
monthly_data = df_year.groupby('Month').agg({
    'Occupancy': 'mean',
    'ADR': 'mean',
    'Revenue': 'sum',
    'Restaurant': 'sum',
    'Bar': 'sum',
    'Spa': 'sum',
    'Other_Services': 'sum',
    'Budget_Revenue': 'sum',
    'Budget_Occupancy': 'mean',
    'Budget_Restaurant': 'sum',
    'Budget_Bar': 'sum',
    'Budget_Spa': 'sum',
    'Budget_Other': 'sum'
}).reset_index()

# Calcul des indicateurs de performance
monthly_data['RevPAR'] = monthly_data['Revenue'] / 30  # Supposons 30 chambres
monthly_data['GOPPAR'] = (monthly_data['Revenue'] * 0.7) / 30  # Marge brute simplifiée
monthly_data['Total_Revenue'] = monthly_data[['Revenue', 'Restaurant', 'Bar', 'Spa', 'Other_Services']].sum(axis=1)
monthly_data['Total_Budget'] = monthly_data[['Budget_Revenue', 'Budget_Restaurant', 'Budget_Bar', 'Budget_Spa', 'Budget_Other']].sum(axis=1)

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["Tableau de bord", "Analyse Budget/Réel", "Prévisions", "Analyse détaillée"])

with tab1:
    st.header(f"Tableau de bord global - Année {selected_year}")
    
    # KPIs clés
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Taux d'occupation", f"{df_year['Occupancy'].mean():.1f}%", 
                 f"{(df_year['Occupancy'].mean() - monthly_data['Budget_Occupancy'].mean())/monthly_data['Budget_Occupancy'].mean()*100:.1f}% vs budget")
    with col2:
        st.metric("ADR", f"${df_year['ADR'].mean():.2f}", 
                 f"{(df_year['ADR'].mean() - (monthly_data['Budget_Revenue'].sum()/monthly_data['Revenue'].sum()*df_year['ADR'].mean()))/(monthly_data['Budget_Revenue'].sum()/monthly_data['Revenue'].sum()*df_year['ADR'].mean())*100:.1f}% vs budget")
    with col3:
        st.metric("RevPAR", f"${monthly_data['RevPAR'].mean():.2f}", 
                 f"{(monthly_data['RevPAR'].mean() - (monthly_data['Budget_Revenue'].sum()/30)/12)/((monthly_data['Budget_Revenue'].sum()/30)/12)*100:.1f}% vs budget")
    with col4:
        st.metric("GOPPAR", f"${monthly_data['GOPPAR'].mean():.2f}", 
                 f"{(monthly_data['GOPPAR'].mean() - (monthly_data['Budget_Revenue'].sum()*0.7/30)/12)/((monthly_data['Budget_Revenue'].sum()*0.7/30)/12)*100:.1f}% vs budget")
    
    # Graphique principal
    fig = px.line(monthly_data, x='Month', y=selected_metric, 
                 title=f"Évolution du {selected_metric} par mois ({selected_year})")
    st.plotly_chart(fig, use_container_width=True)
    
    # Répartition des revenus
    revenue_dist = monthly_data[['Revenue', 'Restaurant', 'Bar', 'Spa', 'Other_Services']].sum()
    fig = px.pie(revenue_dist, values=revenue_dist.values, names=revenue_dist.index,
                title=f"Répartition des revenus ({selected_year})")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header(f"Analyse Budget vs Réel - {selected_year}")
    
    # Sélection du service à analyser
    service = st.selectbox("Service à analyser", 
                         options=['Hébergement', 'Restaurant', 'Bar', 'Spa', 'Autres services', 'Tous les services'])
    
    if service != 'Tous les services':
        # Analyse pour un service spécifique
        service_map = {
            'Hébergement': ('Revenue', 'Budget_Revenue'),
            'Restaurant': ('Restaurant', 'Budget_Restaurant'),
            'Bar': ('Bar', 'Budget_Bar'),
            'Spa': ('Spa', 'Budget_Spa'),
            'Autres services': ('Other_Services', 'Budget_Other')
        }
        
        real_col, budget_col = service_map[service]
        
        # Graphique comparatif
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_data['Month'], y=monthly_data[real_col], 
                           name='Revenu réel'))
        fig.add_trace(go.Bar(x=monthly_data['Month'], y=monthly_data[budget_col], 
                           name='Budget'))
        fig.update_layout(barmode='group', title=f"{service}: Budget vs Réel ({selected_year})",
                        yaxis_title="Revenu ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcul des écarts
        monthly_data['Ecart'] = monthly_data[real_col] - monthly_data[budget_col]
        monthly_data['Ecart_pct'] = (monthly_data['Ecart'] / monthly_data[budget_col]) * 100
        
        # Graphique des écarts
        fig = px.bar(monthly_data, x='Month', y='Ecart_pct', 
                    title=f"Écart en pourcentage ({service})",
                    labels={'Ecart_pct': 'Écart (%)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau récapitulatif
        st.subheader("Récapitulatif mensuel")
        summary = monthly_data[['Month', real_col, budget_col, 'Ecart', 'Ecart_pct']].copy()
        summary.columns = ['Mois', 'Réel', 'Budget', 'Écart ($)', 'Écart (%)']
        st.dataframe(summary.style.format({
            'Réel': '${:,.2f}',
            'Budget': '${:,.2f}',
            'Écart ($)': '${:,.2f}',
            'Écart (%)': '{:.1f}%'
        }), use_container_width=True)
    else:
        # Vue consolidée pour tous les services
        st.subheader("Synthèse des écarts Budget/Réel")
        
        # Préparation des données
        services = ['Hébergement', 'Restaurant', 'Bar', 'Spa', 'Autres services']
        real_cols = ['Revenue', 'Restaurant', 'Bar', 'Spa', 'Other_Services']
        budget_cols = ['Budget_Revenue', 'Budget_Restaurant', 'Budget_Bar', 'Budget_Spa', 'Budget_Other']
        
        comparison_data = []
        for service, real_col, budget_col in zip(services, real_cols, budget_cols):
            temp_df = monthly_data[['Month', real_col, budget_col]].copy()
            temp_df['Service'] = service
            temp_df.columns = ['Month', 'Reel', 'Budget', 'Service']
            temp_df['Ecart'] = temp_df['Reel'] - temp_df['Budget']
            temp_df['Ecart_pct'] = (temp_df['Ecart'] / temp_df['Budget']) * 100
            comparison_data.append(temp_df)
        
        comparison_df = pd.concat(comparison_data)
        
        # Graphique des écarts en valeur absolue
        fig = px.bar(comparison_df, x='Month', y='Ecart', color='Service',
                    barmode='group', title=f"Écarts Budget/Réel par service ({selected_year})",
                    labels={'Ecart': 'Écart ($)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique des écarts en pourcentage
        fig = px.bar(comparison_df, x='Month', y='Ecart_pct', color='Service',
                    barmode='group', title=f"Écarts en % par service ({selected_year})",
                    labels={'Ecart_pct': 'Écart (%)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau récapitulatif
        st.subheader("Récapitulatif par service")
        pivot_table = comparison_df.pivot_table(index='Service', values=['Reel', 'Budget', 'Ecart', 'Ecart_pct'], aggfunc='sum')
        pivot_table['Reel'] = pivot_table['Reel'].map('${:,.2f}'.format)
        pivot_table['Budget'] = pivot_table['Budget'].map('${:,.2f}'.format)
        pivot_table['Ecart'] = pivot_table['Ecart'].map('${:,.2f}'.format)
        pivot_table['Ecart_pct'] = pivot_table['Ecart_pct'].map('{:.1f}%'.format)
        st.dataframe(pivot_table, use_container_width=True)

with tab3:
    st.header(f"Prévisions - {selected_year}")
    
    col1, col2 = st.columns(2)
    with col1:
        # Choix du service à prévoir
        forecast_service = st.selectbox("Service à prévoir", 
                                      options=['Occupancy', 'ADR', 'Revenue', 'Restaurant', 'Bar', 'Spa', 'Other_Services'])
    with col2:
        # Choix de l'horizon de prévision
        forecast_horizon = st.selectbox("Horizon de prévision", 
                                      options=['1 mois', '3 mois', '6 mois'])
    
    # Choix du modèle
    model_type = st.radio("Modèle de prévision", 
                         options=['ARIMA', 'Prophet (recommandé)'], horizontal=True)
    
    # Préparation des données pour les modèles
    ts_data = df_year.groupby('Date')[forecast_service].mean().reset_index()
    ts_data.columns = ['ds', 'y']
    
    # Déterminer le nombre de périodes à prévoir
    periods = 30 if forecast_horizon == '1 mois' else 90 if forecast_horizon == '3 mois' else 180
    
    if model_type == 'ARIMA':
        try:
            st.write(f"Prévision {forecast_service} avec modèle ARIMA ({forecast_horizon})")
            model = ARIMA(ts_data['y'], order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name='Historique'))
            fig.add_trace(go.Scatter(x=pd.date_range(start=ts_data['ds'].iloc[-1], periods=periods+1)[1:], 
                                    y=forecast, name='Prévision'))
            fig.update_layout(title=f"Prévision {forecast_service} ({forecast_horizon})")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur avec le modèle ARIMA: {str(e)}")
            st.info("Essayez avec le modèle Prophet ou vérifiez vos données")
        
    else:  # Prophet
        try:
            st.write(f"Prévision {forecast_service} avec Prophet ({forecast_horizon})")
            model = Prophet()
            model.fit(ts_data)
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            fig = model.plot(forecast)
            st.pyplot(fig)
            
            # Composantes de la prévision
            st.subheader("Composantes de la prévision")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
            
            # Affichage des valeurs prévues
            st.subheader("Détails des prévisions")
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            forecast_df.columns = ['Date', 'Prévision', 'Borne inférieure', 'Borne supérieure']
            st.dataframe(forecast_df.style.format({
                'Prévision': '{:.2f}',
                'Borne inférieure': '{:.2f}',
                'Borne supérieure': '{:.2f}'
            }), use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur avec le modèle Prophet: {str(e)}")

with tab4:
    st.header(f"Analyse détaillée par service - {selected_year}")
    
    # Sélection du service
    service = st.selectbox("Sélectionnez un service", 
                         options=['Hébergement', 'Restaurant', 'Bar', 'Spa', 'Autres services'])
    
    service_map = {
        'Hébergement': ('Revenue', 'Budget_Revenue'),
        'Restaurant': ('Restaurant', 'Budget_Restaurant'),
        'Bar': ('Bar', 'Budget_Bar'),
        'Spa': ('Spa', 'Budget_Spa'),
        'Autres services': ('Other_Services', 'Budget_Other')
    }
    
    real_col, budget_col = service_map[service]
    
    # KPIs spécifiques au service
    st.subheader(f"Indicateurs clés - {service}")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_real = monthly_data[real_col].sum()
        st.metric(f"Revenu total réel", f"${total_real:,.2f}")
    with col2:
        total_budget = monthly_data[budget_col].sum()
        st.metric(f"Budget total", f"${total_budget:,.2f}")
    with col3:
        variance = total_real - total_budget
        variance_pct = (variance / total_budget) * 100
        st.metric(f"Écart total", f"${variance:,.2f}", f"{variance_pct:.1f}%")
    
    # Analyse mensuelle
    st.subheader(f"Analyse mensuelle - {service}")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_data['Month'], y=monthly_data[real_col], name='Réel'))
    fig.add_trace(go.Bar(x=monthly_data['Month'], y=monthly_data[budget_col], name='Budget'))
    fig.add_trace(go.Scatter(x=monthly_data['Month'], 
                           y=(monthly_data[real_col] - monthly_data[budget_col]), 
                           name='Écart', mode='lines+markers'))
    fig.update_layout(barmode='group', title=f"Performance mensuelle - {service}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse quotidienne
    st.subheader(f"Analyse quotidienne - {service}")
    daily_data = df_year[['Date', real_col, budget_col]].copy()
    daily_data['Ecart'] = daily_data[real_col] - daily_data[budget_col]
    daily_data['Ecart_pct'] = (daily_data['Ecart'] / daily_data[budget_col]) * 100
    
    fig = px.line(daily_data, x='Date', y=[real_col, budget_col], 
                title=f"Performance quotidienne - {service}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des écarts
    st.subheader(f"Distribution des écarts - {service}")
    fig = px.histogram(daily_data, x='Ecart_pct', nbins=30, 
                      title="Distribution des écarts en %")
    st.plotly_chart(fig, use_container_width=True)

# Téléchargement des rapports
st.sidebar.header("Export des données")
if st.sidebar.button("Générer rapport PDF"):
    st.sidebar.success(f"Rapport {selected_year} généré (simulation)")

if st.sidebar.button("Exporter les données en Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        monthly_data.to_excel(writer, sheet_name='KPIs Mensuels')
        df_year.to_excel(writer, sheet_name='Données Journalières')
        
        # Ajout des données d'analyse budget/réel
        services = ['Hébergement', 'Restaurant', 'Bar', 'Spa', 'Autres services']
        real_cols = ['Revenue', 'Restaurant', 'Bar', 'Spa', 'Other_Services']
        budget_cols = ['Budget_Revenue', 'Budget_Restaurant', 'Budget_Bar', 'Budget_Spa', 'Budget_Other']
        
        for service, real_col, budget_col in zip(services, real_cols, budget_cols):
            temp_df = df_year[['Date', real_col, budget_col]].copy()
            temp_df['Ecart'] = temp_df[real_col] - temp_df[budget_col]
            temp_df['Ecart_pct'] = (temp_df['Ecart'] / temp_df[budget_col]) * 100
            temp_df.to_excel(writer, sheet_name=f'Détails {service}')
    
    st.sidebar.download_button(
        label="Télécharger le fichier Excel",
        data=output.getvalue(),
        file_name=f"dashboard_hotel_{selected_year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Documentation
st.sidebar.header("Aide")
st.sidebar.info("""
**Indicateurs clés:**
- **ADR**: Average Daily Rate (Revenu moyen par chambre occupée)
- **RevPAR**: Revenue per Available Room (Revenu par chambre disponible)
- **GOPPAR**: Gross Operating Profit per Available Room

**Modèles de prévision:**
- **ARIMA**: Modèle statistique simple
- **Prophet**: Modèle avancé développé par Facebook, gère mieux les saisonnalités
""")
