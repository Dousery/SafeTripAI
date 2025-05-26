# --- Gerekli kütüphaneler ---
import os
import re
from ast import literal_eval
import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Rastgeleliğin tekrarlanabilir olması için
np.random.seed(42)

# --- Sayfa konfigürasyonu ---
st.set_page_config(
    page_title="Turizm Sağlık Danışmanı",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Özel CSS Stilleri ---
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .subheader {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #0D47A1;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .info-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
    }
    
    .footer {
        text-align: center;
        color: #616161;
        margin-top: 50px;
        padding: 10px;
        border-top: 1px solid #EEEEEE;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .metric-label {
        color: #616161;
        font-size: 14px;
    }
    
    .chart-container {
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    div[data-testid="stSelectbox"] div[role="button"] div {
        color: #1E88E5;
        font-weight: 500;
    }
    
    div[data-testid="stDateInput"] div[role="button"] div {
        color: #1E88E5;
        font-weight: 500;
    }
    
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #1565C0;
    }
    
    .recommendation-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid #43A047;
    }
    
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid #E53935;
    }
</style>
""", unsafe_allow_html=True)

# --- ENV Ayarları ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# --- Başlık ve Giriş ---
st.markdown('<h1 class="main-header">🌍 Turizm Destinasyonları için Bulaşıcı Hastalık Danışmanı</h1>', unsafe_allow_html=True)

with st.expander("ℹ️ Uygulama Hakkında", expanded=False):
    st.markdown("""
    Bu uygulama, seyahat planlaması yaparken bulaşıcı hastalık riski açısından en güvenli destinasyonları 
    belirlemenize yardımcı olur. Yapay zeka destekli analiz ile seçtiğiniz eyaletin ve benzer eyaletlerin 
    sağlık verilerini değerlendirir ve en güvenli seyahat seçeneğini önerir.
    
    **Özellikler:**
    - Eyaletlere göre bulaşıcı hastalık tahminleri
    - Benzer iklim ve coğrafi koşullara sahip alternatif destinasyon önerileri
    - Kişi başına düşen hastalık oranlarına göre risk değerlendirmesi
    - Görselleştirilmiş veri analizleri
    """)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🔍 Destinasyon Analizi")
    
    # --- CSV dosyalarını oku ---
    @st.cache_data
    def load_data():
        raw_df1 = pd.read_csv("data_1.csv", delimiter=';', engine='python')
        raw_df2 = pd.read_csv("data_2.csv", delimiter=';', engine='python')
        
        # --- Tarih formatlarını düzelt ---
        raw_df1["Week Ending Date"] = raw_df1["Week Ending Date"].str.replace(".", "/", regex=False)
        raw_df1["Week Ending Date"] = pd.to_datetime(raw_df1["Week Ending Date"], format="%m/%d/%Y", errors="coerce")

        raw_df2["Week Ending Date"] = raw_df2["Week Ending Date"].str.replace(".", "/", regex=False)
        raw_df2["Week Ending Date"] = pd.to_datetime(raw_df2["Week Ending Date"], format="%d/%m/%Y", errors="coerce")
        
        # --- Sütunları filtrele ---
        columns_to_hold_df1 = ['Jurisdiction of Occurrence', 'Week Ending Date', 'Septicemia (A40-A41)', 'Influenza and pneumonia (J10-J18)']
        columns_to_hold_df2 = ['Jurisdiction of Occurrence', 'Week Ending Date', 'Septicemia (A40-A41)', 'Influenza and pneumonia (J09-J18)']

        filtered_raw_df1 = raw_df1[columns_to_hold_df1]
        filtered_raw_df2 = raw_df2[columns_to_hold_df2]

        filtered_raw_df1.columns = filtered_raw_df1.columns.str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
        filtered_raw_df2.columns = filtered_raw_df1.columns.str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
        filtered_raw_df1 = filtered_raw_df1[filtered_raw_df2.columns]
        
        # --- Birleştir ---
        merged_df = pd.concat([filtered_raw_df1, filtered_raw_df2], axis=0, ignore_index=True)
        copy_merged = merged_df.copy()
        copy_merged.set_index('Week Ending Date', inplace=True)
        
        # --- Sayısallaştır ---
        infectious_diseases = ['Septicemia', 'Influenza and pneumonia']
        for col in infectious_diseases:
            copy_merged[col] = pd.to_numeric(copy_merged[col], errors='coerce')
            
        return copy_merged
    
    copy_merged = load_data()
    
    # --- Eyalet listesi ---
    states_list = copy_merged['Jurisdiction of Occurrence'].dropna().unique().tolist()
    states_list = [state for state in states_list if state != "United States"]
    
    state_options = ["-- Eyalet seçin --"] + sorted(states_list)
    state = st.selectbox("🏙️ Ziyaret etmek istediğiniz eyaleti seçin:", state_options)

    if state == "-- Eyalet seçin --":
        st.warning("Lütfen analiz yapmadan önce bir eyalet seçin.")
    
    # --- Takvimle tarih seçimi ---
    min_date = datetime.date(2024, 1, 1)
    selected_date = st.date_input("📅 Seyahat tarihinizi seçin:", value=min_date, min_value=min_date)

    if selected_date == min_date:
        st.warning("Lütfen seyahat tarihi seçin.")

    
    st.markdown(f"""
    <div class="info-box">
        <strong>Seçilen Eyalet:</strong> {state}<br>
        <strong>Seçilen Tarih:</strong> {selected_date.strftime("%d-%m-%Y")}
    </div>
    """, unsafe_allow_html=True)
    
    if state != "-- Eyalet seçin --" and selected_date != min_date:
        if st.button("🔍 Analizi Başlat"):
            with st.spinner("Veriler analiz ediliyor..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    import time
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("Analiz tamamlandı!")
    else:
        st.info("Analize başlamak için lütfen eyalet ve tarih seçin.")

if state != "-- Eyalet seçin --" and selected_date != min_date:
   
    # --- LLM modeli ---
   llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)
   
   def extract_list_from_response(response: str) -> list[str]:
       try:
           code_blocks = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)
           if code_blocks:
               return literal_eval(code_blocks[0])
           list_text = re.search(r"\[.*?\]", response)
           if list_text:
               return literal_eval(list_text.group(0))
       except Exception as e:
           print("Liste çıkarma hatası:", e)
       return []
   
   @st.cache_data(show_spinner=True)
   def get_similar_states_via_llm(user_state: str, states: list[str], date: str) -> list[str]:
       prompt = f"""
   Aşağıda listesi verilen ABD eyaletleri içinde, {user_state} eyaletine {date} tarihinde
   iklim, coğrafya ve kültürel yapı açısından en çok benzeyen 3 tanesini sırala.
   
   Bu tarihteki mevsimsel koşulları da dikkate al.
   
   Eyalet listesi: {states}
   
   Yalnızca Python listesi formatında 3 eyalet döndür: örneğin ["Arizona", "Nevada", "New Mexico"]
   """
       try:
           response = llm.invoke(prompt).content
           return extract_list_from_response(response)
       except Exception as e:
           print("Benzer eyalet tahmini hatası:", e)
           return []
   
   # --- Benzer eyaletleri al ---
   states = get_similar_states_via_llm(state, states_list, selected_date)
   states_copy = states
   states = [state] + states  # ilk eyalet kullanıcı seçimi
   
   # --- Ana ekran ikiye bölünmüş düzen ---
   col1, col2 = st.columns([1, 1])
   
   # --- Sol Kolon: Benzer Eyaletler ve Harita ---
   with col1:
       st.markdown('<h3 class="subheader">🗺️ Benzer Destinasyonlar</h3>', unsafe_allow_html=True)
       
       # ABD haritası (Plotly ile)
       @st.cache_data
       def create_us_map(states_list, highlight_states):
           # Tüm eyaletler için boş renk değerleri
           state_colors = {state: "lightgrey" for state in states_list}
           
           # Vurgulanan eyaletlerin renklerini güncelle
           for i, state in enumerate(highlight_states):
               if i == 0:  # Kullanıcının seçtiği eyalet
                   state_colors[state] = "#1E88E5"
               else:  # Benzer eyaletler
                   state_colors[state] = "#90CAF9"
           
           fig = go.Figure(data=go.Choropleth(
               locations=[state_abbr.get(state, state) for state in states_list],  # Eyalet kısaltmaları
               z=[1 if state in highlight_states else 0 for state in states_list],  # Vurgulama için kukla değişken
               locationmode='USA-states',
               colorscale=[[0, 'lightgrey'], [1, '#1E88E5']],
               showscale=False,
               marker_line_color='white',
               marker_line_width=0.5,
               colorbar_title='',
               customdata=[state for state in states_list],
               hovertemplate='%{customdata}<extra></extra>'
           ))
   
           fig.update_layout(
               geo = dict(
                   scope='usa',
                   projection=go.layout.geo.Projection(type='albers usa'),
                   showlakes=True,
                   lakecolor='rgb(255, 255, 255)'),
               title_text='',
               height=350,
               margin={"r":0,"t":0,"l":0,"b":0},
               paper_bgcolor='rgba(0,0,0,0)',
               plot_bgcolor='rgba(0,0,0,0)'
           )
           
           return fig
       
       # Eyalet kısaltmaları sözlüğü
       state_abbr = {
           "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
           "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
           "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
           "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
           "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
           "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
           "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
           "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
           "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
           "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
           "District of Columbia": "DC"
       }
       
       st.plotly_chart(create_us_map(states_list, states), use_container_width=True)
       
       # Benzer eyaletlerin detayları
       st.markdown(f"""
       <div class="info-box">
           <strong>Seçilen Eyalete Benzer Destinasyonlar:</strong><br>
           {", ".join(states_copy)}
       </div>
       """, unsafe_allow_html=True)
   
   # --- Verileri süz ---
   filtered_df_1 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[0]]
   filtered_df_2 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[1]]
   filtered_df_3 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[2]]
   filtered_df_4 = copy_merged[copy_merged['Jurisdiction of Occurrence'] == states[3]]
   
   dfs = [filtered_df_1, filtered_df_2, filtered_df_3, filtered_df_4]
   
   # --- Eksik verileri doldurma fonksiyonu ---
   def generate_imputed_value(katsayi):
       if katsayi >= 0.7:
           return np.random.randint(1, 3)
       elif 0.4 <= katsayi < 0.7:
           return np.random.randint(3, 6)
       else:
           return np.random.randint(6, 10)
   
   def fill_missing_across_states(dfs, group_col, cols_to_fill):
       combined_df = pd.concat(dfs, ignore_index=True)
       filled_df_list = []
   
       for df in dfs:
           filled_df = df.copy()
           for col in cols_to_fill:
               nan_counts = combined_df.groupby(group_col)[col].apply(lambda x: x.isna().sum())
               total_nan = nan_counts.sum()
               for state in df[group_col].unique():
                   state_mask = (df[group_col] == state)
                   state_nan_count = df.loc[state_mask, col].isna().sum()
                   if total_nan != 0:
                       katsayi = 1 - (nan_counts[state] / total_nan)
                       filled_values = [generate_imputed_value(katsayi) for _ in range(state_nan_count)]
                       filled_df.loc[state_mask & df[col].isna(), col] = filled_values
           filled_df_list.append(filled_df)
   
       return filled_df_list
   
   # --- Aykırı değer smoothing fonksiyonu ---
   def smooth_outliers_in_dfs_for_multiple_diseases(dfs, diseases, sigma=2, z_threshold=2):
       smoothed_dfs = []
   
       # Iterate through each DataFrame
       for df_idx, dataframe in enumerate(dfs):
           df = dataframe.copy()
           jurisdiction = df['Jurisdiction of Occurrence'].iloc[0] if 'Jurisdiction of Occurrence' in df.columns else f"Dataset {df_idx}"
           print(f"Processing: {jurisdiction}")
   
           # Iterate through each disease
           for disease_name in diseases:
               if disease_name not in df.columns:
                   print(f"  - {disease_name} column not found in {jurisdiction}")
                   continue
   
               try:
                   # Get the disease data and fill missing values
                   counts = df[disease_name].copy().fillna(method='ffill').fillna(method='bfill').fillna(0)
   
                   # Ensure there are enough data points for processing
                   if len(counts) > 2:
                       # Calculate Z-scores
                       z_scores = zscore(counts)
   
                       # Detect outliers based on Z-score threshold
                       outliers = np.abs(z_scores) > z_threshold
   
                       # Apply Gaussian smoothing to the data
                       smoothed_counts = gaussian_filter1d(counts.values, sigma=sigma)
   
                       # Replace outliers with smoothed values
                       counts[outliers] = smoothed_counts[outliers]
   
                       # Update the DataFrame with smoothed values
                       df[disease_name] = counts
   
                       print(f"    ✓ Success: {sum(outliers)} outliers smoothed")
                   else:
                       print(f"    ! Not enough data for {disease_name} in {jurisdiction}")
   
               except Exception as e:
                   print(f"    ! Error processing {disease_name} in {jurisdiction}: {e}")
   
           # Append the processed DataFrame to the list
           smoothed_dfs.append(df)
   
       return smoothed_dfs
   
   
   group_col = 'Jurisdiction of Occurrence'
   cols_to_fill = ['Septicemia', 'Influenza and pneumonia']
   
   # --- Eksik verileri doldur ---
   filled_dfs = fill_missing_across_states(dfs, group_col, cols_to_fill)
   
   # --- Ayrı ayrı al ---
   filled_df_1, filled_df_2, filled_df_3, filled_df_4 = filled_dfs
   
   # --- Smoothing uygula ---
   smoothed_dfs = smooth_outliers_in_dfs_for_multiple_diseases(filled_dfs, cols_to_fill)
   
   
   def forecast_cases_for_smooth_dfs(location, date, diseases, smooth_dfs, exog_column=None):
       """
       Make forecasts based on smoothed DataFrames for a given location and diseases.
       """
       result = {location: {}}
   
       try:
           # Find the relevant data for the given location in smoothed DataFrames
           location_data = None
           for df in smooth_dfs:
               if location in df['Jurisdiction of Occurrence'].unique():
                   location_data = df[df['Jurisdiction of Occurrence'] == location]
                   break
   
           if location_data is None:
               raise ValueError(f"No data found for {location}.")
   
           # Ensure the DataFrame has a time index
           if not isinstance(location_data.index, pd.DatetimeIndex):
               raise ValueError("The DataFrame does not have a time index.")
   
           # Convert date to Timestamp
           date = pd.Timestamp(date)
           last_date = location_data.index[-1]
   
           # Skip forecasting if the target date is in the past or is already in the data
           if date <= last_date:
               print("Selected date is in the past or within the data range. Forecasting not needed.")
               return result
   
           # Calculate the number of weeks ahead to forecast
           delta_weeks = ((date - last_date).days) // 7
           if delta_weeks < 1:
               delta_weeks = 1  # Ensure at least one week of forecast
   
           # Forecast each disease
           for disease in diseases:
               if disease not in location_data.columns:
                   result[location][disease] = "No data"
                   continue
   
               # Get the disease data and drop NaN values
               y = location_data[disease].dropna()
   
               if len(y) > 2:
                   if exog_column and exog_column in location_data.columns:
                       # Use external data if available
                       X = location_data[exog_column].dropna()
                       common_index = y.index.intersection(X.index)
                       y = y.loc[common_index]
                       X = X.loc[common_index]
   
                       # Fit SARIMAX model with exogenous variables
                       model = SARIMAX(y, order=(2, 1, 2), seasonal_order=(1, 1, 1, 52), exog=X)
                       model_fit = model.fit(disp=False)
                       forecast = model_fit.forecast(steps=delta_weeks, exog=None)  # Exogenous variables are not forecasted
                   else:
                       # Fit SARIMAX model without exogenous variables
                       model = SARIMAX(y, order=(1, 1, 1))
                       model_fit = model.fit(disp=False)
                       forecast = model_fit.forecast(steps=delta_weeks)
   
                   # Get the forecasted value for the last step
                   forecast_value = round(forecast.iloc[-1])
                   result[location][disease] = int(forecast_value)
               else:
                   result[location][disease] = "Insufficient data"
   
       except Exception as e:
           print(f"Error occurred: {type(e).__name__} - {e}")
           result[location] = "Forecasting failed"
   
       return result
   
   # Forecast for all states
   forecast_results_all_states = []
   diseases = ['Septicemia', 'Influenza and pneumonia']
   
   for state in states:
       forecast_results_smooth = forecast_cases_for_smooth_dfs(
           location=state,
           date=selected_date,
           diseases=diseases,
           smooth_dfs=smoothed_dfs
       )
   
       forecast_results_all_states.append(forecast_results_smooth)
   
       print(f"\nForecast Results (Smoothed Data) - {state}:")
       print(forecast_results_smooth)
   
   # Function to process CSV and calculate death rates
   @st.cache_data
   def calculate_death_rate(csv_path, forecast_results_all_states):
       try:
           # Load the CSV file from the provided path
           df = pd.read_csv(csv_path)
   
           # Convert 'Population' to integer
           df['Population'] = df['Population'].astype(int)
   
           # Convert to dict: State -> Population
           population_data = dict(zip(df['State'], df['Population']))
   
           results_with_death_rate = []
   
           # Iterate through forecast results and calculate death rates
           for state_data in forecast_results_all_states:
               for state, disease_data in state_data.items():
                   if state in population_data:
                       total_deaths = disease_data.get('Septicemia', 0) + disease_data.get('Influenza and pneumonia', 0)
                       population = population_data[state]
                       death_rate = (total_deaths / population) * 1000000  # Death rate per 1M
                       disease_data['Death Rate per 1M'] = round(death_rate, 2)
                       results_with_death_rate.append({state: disease_data})
                   else:
                       print(f"Population data not found for: {state}")
                       
           return results_with_death_rate
       except Exception as e:
           print(f"Error processing CSV or calculating death rates: {str(e)}")
           return []
       
   csv_path = "state_populations.csv"
       
   # Call function to calculate death rates
   results_with_death_rate = calculate_death_rate(csv_path, forecast_results_all_states)
   
   # --- Sağ Kolon: Hastalık Tahminleri ve Öneriler ---  
   with col2:
       st.markdown('<h3 class="subheader">📊 Hastalık Risk Analizi</h3>', unsafe_allow_html=True)
       
       # Hastalık verileri için yeni bir tablo oluştur
       disease_data = {}
       
       for result in results_with_death_rate:
           for state_name, values in result.items():
               disease_data[state_name] = {
                   'Septicemia': values.get('Septicemia', 0),
                   'Influenza and pneumonia': values.get('Influenza and pneumonia', 0),
                   'Death Rate per 1M': values.get('Death Rate per 1M', 0)
               }
       
       # Veri tablosu
       disease_df = pd.DataFrame.from_dict(disease_data, orient='index')
       disease_df.index.name = 'Eyalet'
       disease_df.reset_index(inplace=True)
       disease_df.columns = ['Eyalet', 'Septicemia (Tahmin)', 'Influenza ve Pnömoni (Tahmin)', 'Milyon Başına Ölüm']
       
       # Karşılaştırma grafiği oluştur
       fig = px.bar(
           disease_df, 
           x='Eyalet', 
           y='Milyon Başına Ölüm',
           color='Milyon Başına Ölüm',
           color_continuous_scale='Blues',
           title='Eyaletlere Göre 1 Milyon Başına Tahmin Edilen Ölüm Oranları',
           height=400
       )
       
       fig.update_layout(
           xaxis_title="Eyalet",
           yaxis_title="1 Milyon Başına Ölüm",
           paper_bgcolor='rgba(0,0,0,0)',
           plot_bgcolor='rgba(0,0,0,0)',
           margin=dict(l=20, r=20, t=50, b=20),
       )
       
       st.plotly_chart(fig, use_container_width=True)
       
       # Metrikleri göster
       metric_cols = st.columns(len(states))
       
       for i, (col, s) in enumerate(zip(metric_cols, states)):
           with col:
               rate = disease_data[s]['Death Rate per 1M']
               
               color_class = ""
               if rate == min([disease_data[x]['Death Rate per 1M'] for x in states]):
                   color_class = "style='color: #43A047'"
               elif rate == max([disease_data[x]['Death Rate per 1M'] for x in states]):
                   color_class = "style='color: #E53935'"
                   
               st.markdown(f"""
               <div class="metric-card">
                   <div class="metric-label">{s}</div>
                   <div class="metric-value" {color_class}>{rate}</div>
                   <div class="metric-label">1M başına ölüm</div>
               </div>
               """, unsafe_allow_html=True)
       
       # Interaktif tablo
       st.markdown('<div class="chart-container">', unsafe_allow_html=True)
       st.dataframe(
           disease_df.style.highlight_min(subset=['Milyon Başına Ölüm'], color="#07F61B")
                          .highlight_max(subset=['Milyon Başına Ölüm'], color="#F50529"),
           use_container_width=True
       )
       st.markdown('</div>', unsafe_allow_html=True)
   
   # --- Güvenli eyalet seçimi fonksiyonu ---
   def choose_safest_state_via_llm(date: str, death_data_list: list) -> str:
       # Liste içindeki dictionary'leri tek bir dictionary'ye dönüştür
       death_data = {}
       for item in death_data_list:
           death_data.update(item)
   
       # Normalize edilmiş (küçük harfli) eyalet verisi
       normalized_death_data = {
           state.strip().lower(): data for state, data in death_data.items()
       }
   
       # Gerekli alanları içeren eyaletleri filtrele
       filtered_data = {
           state: normalized_death_data[state]
           for state in normalized_death_data
           if (
               'Septicemia' in normalized_death_data[state] and
               'Influenza and pneumonia' in normalized_death_data[state] and
               'Death Rate per 1M' in normalized_death_data[state]
           )
       }
   
       if not filtered_data:
           return "❗ Uygun ölüm verisi bulunamadı. Lütfen verileri kontrol edin."
   
       # En güvenli eyaleti belirle (en düşük 1M başına ölüm oranı)
       safest_state = min(filtered_data.items(), key=lambda x: x[1]['Death Rate per 1M'])[0].title()
   
       # Metin olarak veri gösterimi
       death_info_str = "\n".join([ 
           f"{state.title()}:\n"
           f"  - Septicemia: {data['Septicemia']} ölüm\n"
           f"  - Influenza and pneumonia: {data['Influenza and pneumonia']} ölüm\n"
           f"  - 1M kişi başına ölüm oranı: {data['Death Rate per 1M']}"
           for state, data in filtered_data.items()
       ])
   
       # LLM'e gönderilecek prompt
       prompt = f"""
   Aşağıda {date} tarihi için bazı ABD eyaletlerinde iki hastalık (Septicemia ve Influenza and pneumonia) nedeniyle tahmin edilen ölüm verileri yer alıyor.
   
   Amacın, seyahat için **en güvenli eyaleti** belirlemek.
   
   Ölüm verileri:
   
   {death_info_str}
   
   Verilere göre, **1 milyon kişi başına ölüm oranı** en düşük olan eyalet: **{safest_state}**.
   
   Kullanıcıya şu bilgileri anlaşılır ve sohbet havasında aktar:
   
   - Her eyalette **Septicemia** ve **Influenza and pneumonia** için ölüm sayılarını ve toplam 1 milyon kişi başına düşen ölüm oranını belirt.
   - **{safest_state}** eyaletini neden en güvenli seçenek olarak önerdiğini açık ve net bir şekilde açıkla.
   - Eğer tercih edilen eyalet **{state.title()}** değilse:
     - **{state.title()}** ile **{safest_state}** arasında iklim, coğrafya veya kültürel benzerlikleri belirt.
   - Son olarak, kullanıcıya içten ve kısa bir seyahat önerisiyle mesajını tamamla.
   """
   
       # LLM çağrısı
       try:
           response = llm.invoke(prompt).content
           return response
       except Exception as e:
           print("Karar LLM hatası:", e)
           return "⚠️ LLM üzerinden karar verilemedi."
   
   # Tam ekran sonuçlar bölümü
   st.markdown('<h3 class="subheader">🏆 Seyahat Önerileri</h3>', unsafe_allow_html=True)
   
   try:
       # Önce arkaplan işlemini göster
       with st.spinner('En güvenli destinasyon belirleniyor...'):
           # LLM sonucu
           safest_state_info = choose_safest_state_via_llm(
               date=str(selected_date), 
               death_data_list=results_with_death_rate
           )
   
       # Güvenli eyalet bilgileri
       safest_state = ""
       for result in results_with_death_rate:
           for state_name, values in result.items():
               if safest_state == "" or values.get('Death Rate per 1M', float('inf')) < min_death_rate:
                   safest_state = state_name
                   min_death_rate = values.get('Death Rate per 1M', float('inf'))
   
       # Sonucu kullanıcıya göster
       st.markdown(f"""
           <div class="recommendation-box">
               <p>{safest_state_info}</p>
           </div>
           """, unsafe_allow_html=True)
       
           
       # Eyalet bilgileri ve öneriler
       st.markdown('<h3 class="subheader">🌍 Destinasyon Detayları</h3>', unsafe_allow_html=True)
       
       # Detaylı eyalet bilgileri için sekmeler oluştur
       tabs = st.tabs([f"🏙️ {s}" for s in states])
       
       for i, tab in enumerate(tabs):
           with tab:
               current_state = states[i]
               current_data = None
               for item in results_with_death_rate:
                   if current_state in item:
                       current_data = item[current_state]
                       break
                       
               if current_data:
                   # Detaylı bilgiler için 3 sütun oluştur
                   detail_cols = st.columns(3)
                   
                   with detail_cols[0]:
                       st.markdown(f"""
                       <div class="metric-card">
                           <div class="metric-label">Septicemia Tahmini</div>
                           <div class="metric-value">{current_data.get('Septicemia', 'N/A')}</div>
                           <div class="metric-label">Vaka</div>
                       </div>
                       """, unsafe_allow_html=True)
                       
                   with detail_cols[1]:
                       st.markdown(f"""
                       <div class="metric-card">
                           <div class="metric-label">Influenza ve Pnömoni</div>
                           <div class="metric-value">{current_data.get('Influenza and pneumonia', 'N/A')}</div>
                           <div class="metric-label">Vaka</div>
                       </div>
                       """, unsafe_allow_html=True)
                       
                   with detail_cols[2]:
                       st.markdown(f"""
                       <div class="metric-card">
                           <div class="metric-label">Risk Oranı</div>
                           <div class="metric-value">{current_data.get('Death Rate per 1M', 'N/A')}</div>
                           <div class="metric-label">1M başına ölüm</div>
                       </div>
                       """, unsafe_allow_html=True)
                   
                   # Eyalet hakkında bilgi al
                   state_info_prompt = f"""
                   {current_state} eyaleti hakkında turistler için kısa ve faydalı bilgiler ver (görsel yerler, kültür, iklim özellikleri).
                   Özellikle {selected_date.strftime("%B")} ayında ziyaret edilmesinin avantaj ve dezavantajları nelerdir?
                   """
                   
                   with st.spinner(f"{current_state} hakkında bilgiler alınıyor..."):
                       try:
                           state_info = llm.invoke(state_info_prompt).content
                           st.markdown(f"""
                           <div class="info-box">
                               <h4>📝 {current_state} Hakkında</h4>
                               {state_info}
                           </div>
                           """, unsafe_allow_html=True)
                       except Exception as e:
                           st.error(f"Bilgi alınırken bir hata oluştu: {str(e)}")
               else:
                   st.error(f"{current_state} için veri bulunamadı.")
   
   except Exception as e:
       st.error(f"Veri işlenirken hata oluştu: {str(e)}")
   
   # --- Footer ---
   st.markdown("""
   <div class="footer">
       <p>© 2025 Turizm Sağlık Danışmanı | Bu uygulama yapay zeka destekli tahminler sunmaktadır.</p>
       <p><small>Not: Seyahat kararlarınızı verirken lütfen resmi sağlık kurumlarının tavsiyelerini de dikkate alınız.</small></p>
   </div>
   """, unsafe_allow_html=True)