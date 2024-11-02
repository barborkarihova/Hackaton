import streamlit as st
import os
import time
from PIL import Image
import numpy as np
import io
import pandas as pd
import pyglc
import altair as alt
import datetime
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PerFEKTPump",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
col = st.columns((2, 1), gap='medium')

# Initialize DataFrames
basal_data = pd.DataFrame()
bolus_data = pd.DataFrame()
cgm_data = pd.DataFrame()

# df_final_exists = False
if 'df_final_exists' not in st.session_state:
    st.session_state.df_final_exists = False
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = datetime.date(2024,7,5) #None           # datetime.date(2024, 6, 19)


def categorize_time_of_day(df, timestamp_column):
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Create a new column for the category
    df['time_category'] = df[timestamp_column].dt.hour // 2 + 1
    
    return df




@st.cache_data  # Cache the function output
def load_dataframes(d_basal, d_bolus, d_glc):
    return pyglc.process_data(basal_data, bolus_data, cgm_data)


with st.sidebar:
    # st.markdown('**Vyberte cestu k .csv souboru**')
    path = st.file_uploader("Vyberte CSV soubor", accept_multiple_files=True)
    for uploaded_file in path:
        df_final_exists = False
        if uploaded_file.name.startswith('basal'):
            basal_data = pd.read_csv(uploaded_file, skiprows=1)
        elif uploaded_file.name.startswith('bolus'):
            bolus_data = pd.read_csv(uploaded_file, skiprows=1)
        elif uploaded_file.name.startswith('cgm'):
            cgm_data = pd.read_csv(uploaded_file, skiprows=1)      
        # print(uploaded_file)

    if st.button('Potvrdit', type='primary'):
        if not basal_data.empty and not bolus_data.empty and not cgm_data.empty:
            with st.spinner('Probíhá zpracování...'):
                st.session_state.df_final = load_dataframes(basal_data, bolus_data, cgm_data)
                # st.session_state.df_final = pyglc.process_data(basal_data, bolus_data, cgm_data)
                st.session_state.df_final_exists = True
                # st.markdown(st.session_state.df_final.head())
        # df_final = pyglc.process_data(basal_data, bolus_data, cgm_data)
        else:
            st.markdown('**Nedostatečná data**')


st.divider()
if st.session_state.df_final_exists:
    # if st.button('Zobrazit statistiky'):
        with col[0]:
            st.title('PerFEKTpump')

            st.subheader("Profil bazálního podání inzulínu")
            # st.markdown(st.session_state.df_final.head())
            fig, df_means_basal = pyglc.plot_mean(st.session_state.df_final, 'Basal_Rate')
            st.pyplot(fig)    

            st.subheader("Ambulantorní Glykemický Profil (AGP)")
            fig, df_means_glc  = pyglc.plot_mean(st.session_state.df_final, 'GLC')
            st.pyplot(fig)


            st.subheader("Daily records")
            start_date = st.session_state.df_final['Timestamp'].dt.date.min()
            end_date = st.session_state.df_final['Timestamp'].dt.date.max()
            # Streamlit date input within the specified range
            date_input = st.date_input(
                "Select date for plotting",
                value=st.session_state.selected_date,
                min_value=start_date,
                max_value=end_date
            )
            
            # Update the selected date in the session state
            if date_input != st.session_state.selected_date:
                st.session_state.selected_date = date_input  # This will update the selected date in session state

            selected_date = pd.Timestamp(st.session_state.selected_date).date()  # Use the updated session state
            fig = pyglc.plot_day(st.session_state.df_final, selected_date)
            st.pyplot(fig) 




            # # Define the line chart for the mean
            # line = alt.Chart(df_means_glc).mark_line(color='blue').encode(
            #     x=alt.X('Timestamp:T', title='Time'),
            #     y=alt.Y('mean:Q', title='Value')
            # )

            # # Define the error band for the 25th and 75th percentiles
            # band = alt.Chart(df_means_glc).mark_area(opacity=0.3).encode(
            #     x='Timestamp:T',
            #     y='p25:Q',
            #     y2='p75:Q'
            # )

            # # Combine the line and band charts
            # chart = band + line

            # # Display the chart in Streamlit
            # st.altair_chart(chart, use_container_width=True)


   
        with col[1]:
            st.subheader("Statistika chyb pacienta")
            histogram = pyglc.plot_wrong_boluses(st.session_state.df_final)
            st.altair_chart(histogram, use_container_width=True)
        
            st.subheader("Procentuální rozložení bazálního a bolusového inzulínu")
            df_combined = pyglc.get_basal_bolus_ratio(st.session_state.df_final)
            chart = pyglc.plot_insulin_statistics(df_combined, 'Basal_Rate_Percent', 'Total_Bolus_Percent', 'Bazál', 'Bolus')
            st.altair_chart(chart, use_container_width=True)
            bolus_means = pyglc.calculate_mean_by_category(st.session_state.df_final, 'Bolus')
            bazal_means = pyglc.calculate_mean_by_category(st.session_state.df_final, 'Basal_Rate')
            total_means = pyglc.calculate_mean_by_categories(st.session_state.df_final, ['Basal_Rate','Bolus'], mean_column_name = 'Celkový průměr (U)')
            # Merge the dataframes on 'time_category'

            combined_means = bolus_means.merge(bazal_means, on='time_category')
            combined_means = combined_means.merge(total_means, on='time_category')
            # Rename columns if needed for clarity
            bins = [i for i in range(1, 13)]  # Create bins: [0, 2, 4, ..., 24]
            labels = [f"{i}-{i+2}" for i in range(0, 23, 2)]  # Create labels: ['0-2', '2-4', ..., '22-24']




            st.subheader("Procentuální rozložení podání inzulinového bolusu pacientem a pumpou")
            df_combined = pyglc.get_auto_user_ratio(st.session_state.df_final)
            chart = pyglc.plot_insulin_statistics(df_combined, 'Auto_Rate_Percent', 'User_Rate', 'Automaticky', 'Manuálně')
            st.altair_chart(chart, use_container_width=True)


            st.subheader("Statistiky")
            combined_means['time_category'] = combined_means['time_category'].apply(lambda x: f"{(x-1) * 2}-{x * 2}")
            combined_means = combined_means.rename(columns={'Bolus': 'Průměrný Bolus (U)', 'Basal_Rate': 'Pruměrná dávka bazálu (U)','time_category': 'Denní doba (h)'})
            combined_means = combined_means.reset_index(drop=True)
            # Round numerical values and format them to 2 decimal places
            combined_means = combined_means.applymap(lambda x: f"{x:.2f}" if isinstance(x, (float)) else x)

            st.table(combined_means)







else:
    st.markdown('**Není co zobrazit**')   
    
