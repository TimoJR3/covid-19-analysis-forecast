import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# конфиг страницы
st.set_page_config(
    page_title="COVID-19 Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# кастомный стиль
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# заголовок
st.title("🌍 COVID-19: Анализ и Прогноз")
st.markdown("Интерактивный анализ пандемии COVID-19 в разных странах")

# загружаем данные
@st.cache_data
def load_data():
    df = pd.read_csv('data/covid_clean.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_clusters():
    return pd.read_csv('results/country_clusters.csv')

@st.cache_data
def load_forecast_comparison():
    return pd.read_csv('results/forecast_comparison.csv')

# загружаем все данные
df = load_data()
clusters_df = load_clusters()
forecast_comp = load_forecast_comparison()

# боковая панель с фильтрами
st.sidebar.header("⚙️ Настройки")

# выбор страны
selected_country = st.sidebar.selectbox(
    "Выберите страну:",
    sorted(df['Country'].unique()),
    index=0
)

# выбор дат
date_range = st.sidebar.date_input(
    "Выберите период:",
    value=(df['Date'].min().date(), df['Date'].max().date()),
    min_value=df['Date'].min().date(),
    max_value=df['Date'].max().date()
)

# фильтруем данные
country_data = df[df['Country'] == selected_country].sort_values('Date')
if len(date_range) == 2:
    start_date, end_date = date_range
    country_data = country_data[(country_data['Date'].dt.date >= start_date) & 
                               (country_data['Date'].dt.date <= end_date)]

# главное содержимое
tab1, tab2, tab3, tab4 = st.tabs(["📊 Обзор", "📈 Тренды", "🎯 Прогноз", "🔍 Кластеры"])

with tab1:
    st.header(f"Статистика {selected_country}")
    
    # ключевые метрики
    col1, col2, col3, col4 = st.columns(4)
    
    total_cases = country_data['Confirmed'].sum()
    total_deaths = country_data['Deaths'].sum()
    total_recovered = country_data['Recovered'].sum()
    mortality_rate = (total_deaths / total_cases * 100) if total_cases > 0 else 0
    
    col1.metric("📍 Всего случаев", f"{int(total_cases):,}", 
                delta=f"{int(country_data['Confirmed'].iloc[-1]):,}" if len(country_data) > 1 else None)
    col2.metric("💔 Всего смертей", f"{int(total_deaths):,}", 
                delta=f"{int(country_data['Deaths'].iloc[-1]):,}" if len(country_data) > 1 else None)
    col3.metric("✅ Выздоровело", f"{int(total_recovered):,}",
                delta=f"{int(country_data['Recovered'].iloc[-1]):,}" if len(country_data) > 1 else None)
    col4.metric("💀 Смертность", f"{mortality_rate:.2f}%", 
                delta=f"{country_data['Deaths'].iloc[-1]/country_data['Confirmed'].iloc[-1]*100:.2f}%" if len(country_data) > 1 else None)
    
    st.divider()
    
    # график временного ряда
    st.subheader("Динамика заболеваемости")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=country_data['Date'],
        y=country_data['Confirmed'],
        mode='lines',
        name='Случаи',
        line=dict(color='steelblue', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Случаи: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=country_data['Date'],
        y=country_data['Deaths'],
        mode='lines',
        name='Смерти',
        line=dict(color='crimson', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Смерти: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"COVID-19 в {selected_country}",
        xaxis_title="Дата",
        yaxis_title="Количество",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📈 Анализ Тренов")
    
    # скользящее среднее
    country_data_copy = country_data.copy()
    country_data_copy['MA7'] = country_data_copy['Confirmed'].rolling(window=7).mean()
    country_data_copy['MA30'] = country_data_copy['Confirmed'].rolling(window=30).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=country_data_copy['Date'],
        y=country_data_copy['Confirmed'],
        mode='lines',
        name='Ежедневно',
        line=dict(color='lightblue', width=1),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=country_data_copy['Date'],
        y=country_data_copy['MA7'],
        mode='lines',
        name='7-дневное среднее',
        line=dict(color='orange', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=country_data_copy['Date'],
        y=country_data_copy['MA30'],
        mode='lines',
        name='30-дневное среднее',
        line=dict(color='red', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Скользящие средние (обнаружение волн)",
        xaxis_title="Дата",
        yaxis_title="Случаи",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # темп роста
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Темп роста")
        country_data_copy['Daily_Change'] = country_data_copy['Confirmed'].diff()
        
        fig2 = go.Figure()
        colors = ['red' if x > 0 else 'green' for x in country_data_copy['Daily_Change']]
        
        fig2.add_trace(go.Bar(
            x=country_data_copy['Date'],
            y=country_data_copy['Daily_Change'],
            marker_color=colors,
            name='Ежедневное изменение',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:,.0f}<extra></extra>'
        ))
        
        fig2.update_layout(height=300, template='plotly_white', showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Процент роста")
        country_data_copy['Growth_Rate'] = (country_data_copy['Daily_Change'] / 
                                           country_data_copy['Confirmed'].shift(1) * 100).fillna(0)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=country_data_copy['Date'],
            y=country_data_copy['Growth_Rate'],
            mode='lines',
            fill='tozeroy',
            name='% рост',
            line=dict(color='steelblue', width=1),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>'
        ))
        
        fig3.update_layout(height=300, template='plotly_white', showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.header("🔮 Прогноз")
    
    # сравнение методов
    st.subheader("Сравнение методов прогнозирования")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(forecast_comp, x='Method', y='MAE', 
                     color='Method', title='Средняя абсолютная ошибка (MAE)',
                     color_discrete_sequence=['steelblue', 'orange', 'green'])
        fig.update_layout(template='plotly_white', showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(forecast_comp, x='Method', y='RMSE',
                     color='Method', title='Корень среднеквадратичной ошибки (RMSE)',
                     color_discrete_sequence=['steelblue', 'orange', 'green'])
        fig.update_layout(template='plotly_white', showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("🎯 Prophet показывает лучшую точность для долгосрочных прогнозов")

with tab4:
    st.header("🔍 Кластеризация стран")
    
    # таблица кластеров
    st.subheader("Группировка стран по стадии эпидемии")
    
    # смотрим информацию о текущей стране
    current_cluster = clusters_df[clusters_df['Country'] == selected_country]
    
    if not current_cluster.empty:
        cluster_num = int(current_cluster['Cluster'].values[0])
        
        st.info(f"📍 {selected_country} находится в **Кластере {cluster_num}**")
        
        # страны в этом же кластере
        countries_in_cluster = clusters_df[clusters_df['Cluster'] == cluster_num]['Country'].tolist()
        st.write(f"Страны в этом кластере: {', '.join(countries_in_cluster)}")
    
    # общая таблица
    st.subheader("Все страны и кластеры")
    
    display_df = clusters_df[['Country', 'Cluster', 'Growth_Rate', 'Mortality_Rate', 'Recovery_Stage']].copy()
    display_df = display_df.round(2)
    display_df = display_df.sort_values('Cluster')
    
    st.dataframe(display_df, use_container_width=True)
    
    # график распределения по кластерам
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(clusters_df, x='Growth_Rate', y='Mortality_Rate',
                        color='Cluster', size='Total_Cases', hover_data=['Country'],
                        title='Рост vs Смертность',
                        color_discrete_sequence=['steelblue', 'coral', 'lightgreen'])
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(clusters_df, x='Recovery_Stage', y='Mortality_Rate',
                        color='Cluster', size='Total_Cases', hover_data=['Country'],
                        title='Стадия восстановления vs Смертность',
                        color_discrete_sequence=['steelblue', 'coral', 'lightgreen'])
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# нижний раздел: информация
st.divider()

st.header("📌 О проекте")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Данные")
    st.write("""
    - Периода: март 2020 - декабрь 2023
    - Стран: 10
    - Метрик: 3 (cases, deaths, recovered)
    """)

with col2:
    st.subheader("🔬 Методы")
    st.write("""
    - EDA (Exploratory Data Analysis)
    - Time Series Decomposition
    - ARIMA & Prophet Forecasting
    - K-Means Clustering
    - Statistical Analysis
    """)

with col3:
    st.subheader("📁 Ноутбуки")
    st.write("""
    - 01_eda_analysis.ipynb
    - 02_time_series_analysis.ipynb
    - 03_forecasting_models.ipynb
    - 04_clustering_countries.ipynb
    """)

st.markdown("---")
st.caption("COVID-19 Analysis Dashboard | Made with Streamlit | Data: Kaggle")
