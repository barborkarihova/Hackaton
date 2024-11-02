import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import seaborn as sns

def calculate_mean_by_categories(df, value_columns, mean_column_name='Mean_Value'):
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['day'] = df['Timestamp'].dt.date
    
    # Group by 'time_category' and 'day', summing each column in 'value_columns' separately
    daily_sums = df.groupby(['time_category', 'day'])[value_columns].sum().reset_index()
    
    # Sum the columns together for each day
    daily_sums['day_total'] = daily_sums[value_columns].sum(axis=1)
    
    # Calculate the mean of 'day_total' for each 'time_category'
    mean_values = daily_sums.groupby('time_category')['day_total'].mean().reset_index()
    
    # Rename the mean column as specified
    mean_values = mean_values.rename(columns={'day_total': mean_column_name})
    
    return mean_values
# class Data:
#     def __init__(self, df, info):
#         self.data = df
#         self.info = info

#     def time_transform(self):
#         self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'], format='%Y-%m-%d %H:%M')
#         self.data = self.data.sort_values(by='Timestamp').reset_index(drop=True)

#     # def split_days(self):
#     #     self.days = {}
#     #     for day in self.data['Timestamp'].dt.date.unique():
#     #         self.days[day] = self.data[self.data['Timestamp'].dt.date == day]


def time_transform(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M')
    df.sort_values(by='Timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)


def time_standardize_basal(df):
    rows = []

    for _, row in df.iterrows():
        time = row['Timestamp']
        duration = int(row['Duration (minutes)'])
        for min in range(duration):
            new_row = row.copy()
            new_row['Duration (minutes)'] = 1
            new_row['Timestamp'] = time + pd.Timedelta(minutes=min)
            rows.append(new_row)

    if rows:
        # Concatenate all rows into a new DataFrame
        df_new = pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)
        return df_new

def split_bolus(df):
    df['Manual'] = df['Insulin Delivered (U)'].where(pd.notna(df['Carbs Ratio']), 0)
    df['Automatic'] = df['Insulin Delivered (U)'].where(pd.isna(df['Carbs Ratio']), 0)
    df['Insulin Delivered (U)'] = df['Insulin Delivered (U)'].fillna(0)


def glc_cap(df, col, limit):
    df.loc[df[col] > limit, col] = limit


import pandas as pd

# def time_stand(basal, bolus, glc):
#     start = min(basal['Timestamp'].iloc[0], bolus['Timestamp'].iloc[0], glc['Timestamp'].iloc[0])
#     end = max(basal['Timestamp'].iloc[-1], bolus['Timestamp'].iloc[-1], glc['Timestamp'].iloc[-1])
    
#     timestamps = pd.date_range(start=start, end=end, freq='min')
#     df_final = pd.DataFrame({'Timestamp': timestamps})
#     df_final['Bolus'] = 0.0
#     df_final['Manual_Bolus'] = 0.0
#     df_final['Automatic_Bolus'] = 0.0
#     df_final['Basal_Rate'] = 0.0
#     df_final['GLC'] = 0.0

#     # Aggregate basal rates and durations by timestamp to handle duplicates
#     basal_grouped = basal.groupby('Timestamp').agg({
#         'Rate': 'mean',  # or 'sum' based on your needs
#         'Duration (minutes)': 'mean'  # or 'sum'
#     }).reset_index()
#     basal_rate_map = {row['Timestamp']: row['Rate'] / 60 for _, row in basal_grouped.iterrows()}
#     basal_duration_map = {row['Timestamp']: row['Duration (minutes)'] for _, row in basal_grouped.iterrows()}

#     # Aggregate bolus by timestamp to handle duplicates
#     bolus_grouped = bolus.groupby('Timestamp').agg({
#         'Insulin Delivered (U)': 'sum',
#         'Manual': 'sum',
#         'Automatic': 'sum'
#     }).reset_index()
#     bolus_map = bolus_grouped.set_index('Timestamp')[['Insulin Delivered (U)', 'Manual', 'Automatic']].to_dict('index')

#     # Aggregate glucose values by timestamp to handle duplicates
#     glc_grouped = glc.groupby('Timestamp').agg({'CGM Glucose Value (mmol/l)': 'mean'}).reset_index()
#     glc_map = glc_grouped.set_index('Timestamp')['CGM Glucose Value (mmol/l)'].to_dict()

#     # Initialize tracking variables
#     basal_rate_prev = 0
#     basal_duration = 0
#     glc_prev = 0
#     glc_duration = 5

#     # Process each timestamp
#     for idx, timestamp in enumerate(df_final['Timestamp']):
#         if timestamp in basal_rate_map:
#             basal_rate_prev = basal_rate_map[timestamp]
#             basal_duration = basal_duration_map[timestamp]
#         if basal_duration > 0:
#             df_final.at[idx, 'Basal_Rate'] = basal_rate_prev
#             basal_duration -= 1

#         if timestamp in bolus_map:
#             df_final.at[idx, 'Bolus'] = bolus_map[timestamp]['Insulin Delivered (U)']
#             df_final.at[idx, 'Manual_Bolus'] = bolus_map[timestamp]['Manual']
#             df_final.at[idx, 'Automatic_Bolus'] = bolus_map[timestamp]['Automatic']

#         if timestamp in glc_map:
#             glc_prev = glc_map[timestamp]
#             glc_duration = 5
#         if glc_duration > 0:
#             df_final.at[idx, 'GLC'] = glc_prev
#             glc_duration -= 1

#     return df_final

def time_stand(basal, bolus, glc):
    start = min(basal['Timestamp'].iloc[0], bolus['Timestamp'].iloc[0], glc['Timestamp'].iloc[0])
    end = max(basal['Timestamp'].iloc[-1], bolus['Timestamp'].iloc[-1], glc['Timestamp'].iloc[-1])
    
    timestamps = pd.date_range(start=start, end=end, freq='min')
    df_final = pd.DataFrame({'Timestamp': timestamps})
    df_final['Bolus'] = 0.0
    df_final['Manual_Bolus'] = 0.0
    df_final['Automatic_Bolus'] = 0.0
    df_final['Basal_Rate'] = 0.0
    df_final['GLC'] = 0.0

    # Aggregate basal rates and durations by timestamp to handle duplicates
    basal_grouped = basal.groupby('Timestamp').agg({
        'Rate': 'mean',  # or 'sum' based on your needs
        'Duration (minutes)': 'mean'  # or 'sum'
    }).reset_index()
    basal_rate_map = {row['Timestamp']: row['Rate'] / 60 for _, row in basal_grouped.iterrows()}
    basal_duration_map = {row['Timestamp']: row['Duration (minutes)'] for _, row in basal_grouped.iterrows()}

    # Aggregate bolus by timestamp to handle duplicates
    bolus_grouped = bolus.groupby('Timestamp').agg({
        'Insulin Delivered (U)': 'sum',
        'Manual': 'sum',
        'Automatic': 'sum'
    }).reset_index()
    bolus_map = bolus_grouped.set_index('Timestamp')[['Insulin Delivered (U)', 'Manual', 'Automatic']].to_dict('index')

    # Aggregate glucose values by timestamp to handle duplicates
    glc_grouped = glc.groupby('Timestamp').agg({'CGM Glucose Value (mmol/l)': 'mean'}).reset_index()
    glc_map = glc_grouped.set_index('Timestamp')['CGM Glucose Value (mmol/l)'].to_dict()

    # Initialize tracking variables
    basal_rate_prev = 0
    basal_duration = 0
    glc_prev = 0
    glc_duration = 5

    # Process each timestamp
    for idx, timestamp in enumerate(df_final['Timestamp']):
        if timestamp in basal_rate_map:
            basal_rate_prev = basal_rate_map[timestamp]
            basal_duration = basal_duration_map[timestamp]
        if basal_duration > 0:
            df_final.at[idx, 'Basal_Rate'] = basal_rate_prev
            basal_duration -= 1

        if timestamp in bolus_map:
            df_final.at[idx, 'Bolus'] = bolus_map[timestamp]['Insulin Delivered (U)']
            df_final.at[idx, 'Manual_Bolus'] = bolus_map[timestamp]['Manual']
            df_final.at[idx, 'Automatic_Bolus'] = bolus_map[timestamp]['Automatic']

        if timestamp in glc_map:
            glc_prev = glc_map[timestamp]
            glc_duration = 5
        if glc_duration > 0:
            df_final.at[idx, 'GLC'] = glc_prev
            glc_duration -= 1

    return df_final

# def process_data(basal_dir, bolus_dir, glc_dir):
#     # Read files into dframes dictionary
#     dframes = {}

#     with open(basal_dir, 'r') as f:
#         df_info = f.readline().strip()
#     dframes['basal'] = Data(pd.read_csv(basal_dir, skiprows=1), df_info)
#     dframes['basal'].data.drop(columns=['Percentage (%)', 'Insulin Delivered (U)', 'Serial Number'], inplace=True)

#     with open(bolus_dir, 'r') as f:
#         df_info = f.readline().strip()
#     dframes['bolus'] = Data(pd.read_csv(bolus_dir, skiprows=1), df_info)
#     dframes['bolus'].data.drop(columns=['Insulin Type', 'Initial Delivery (U)', 'Extended Delivery (U)', 'Serial Number'], inplace=True)

#     with open(glc_dir, 'r') as f:
#         df_info = f.readline().strip()
#     dframes['glc'] = Data(pd.read_csv(glc_dir, skiprows=1), df_info)
#     dframes['glc'].data.drop(columns=['Serial Number'], inplace=True)
#     glc_cap(dframes['glc'].data, 'CGM Glucose Value (mmol/l)', 22)

#     # Transform time columns to datetime objects and sort by timestamp
#     for key in dframes.keys():
#         dframes[key].time_transform()

#     split_bolus(dframes['bolus'].data)

#     dframes['basal'].data = time_standardize_basal(dframes['basal'].data)

#     df_final = time_stand(dframes['basal'].data, dframes['bolus'].data, dframes['glc'].data)

#     return df_final


def process_data(basal_df, bolus_df, glc_df):
    basal_df.drop(columns=['Percentage (%)', 'Insulin Delivered (U)', 'Serial Number'], inplace=True)
    bolus_df.drop(columns=['Insulin Type', 'Initial Delivery (U)', 'Extended Delivery (U)', 'Serial Number'], inplace=True)
    glc_df.drop(columns=['Serial Number'], inplace=True)
    glc_cap(glc_df, 'CGM Glucose Value (mmol/l)', 22)

    # Transform time columns to datetime objects and sort by timestamp
    time_transform(basal_df)
    time_transform(bolus_df)
    time_transform(glc_df)

    split_bolus(bolus_df)

    basal_df = time_standardize_basal(basal_df)

    df_final = time_stand(basal_df, bolus_df, glc_df)

    return df_final


def plot_mean(df, column):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['MinutesOfDay'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute
    
    grouped = df.groupby(df['MinutesOfDay'])[column].agg(
        mean='mean',
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        p10=lambda x: x.quantile(0.10),
        p90=lambda x: x.quantile(0.90)
    ).reset_index()

    # Set the MinutesOfDay as the index for rolling calculations
    grouped.set_index('MinutesOfDay', inplace=True)

    # Calculate rolling mean and percentiles with a window of 10 minutes (5 min before and after)
    win_len = 121
    rolling_mean = grouped['mean'].rolling(window=win_len, center=True).mean()
    rolling_p25 = grouped['p25'].rolling(window=win_len, center=True).quantile(0.25)
    rolling_p75 = grouped['p75'].rolling(window=win_len, center=True).quantile(0.75)
    rolling_p10 = grouped['p10'].rolling(window=win_len, center=True).quantile(0.10)
    rolling_p90 = grouped['p90'].rolling(window=win_len, center=True).quantile(0.90)

    # Create a new DataFrame to hold the rolling statistics
    rolling_stats = pd.DataFrame({
        'Mean': rolling_mean,
        'P25': rolling_p25,
        'P75': rolling_p75,
        'P10': rolling_p10,
        'P90': rolling_p90
    }).reset_index()

    # Plot the results using seaborn
    plt.figure(figsize=(12, 6))

    rolling_stats['MinutesOfDay'] = rolling_stats['MinutesOfDay']/60
    # Plot mean glucose level with 25-75% and 10-90% intervals
    sns.lineplot(data=rolling_stats, x='MinutesOfDay', y='Mean', label='Mean Basal_Rate', color='blue')
    plt.fill_between(rolling_stats['MinutesOfDay'], rolling_stats['P25'], rolling_stats['P75'], color='blue', alpha=0.3, label='25-75%')
    plt.fill_between(rolling_stats['MinutesOfDay'], rolling_stats['P10'], rolling_stats['P90'], color='blue', alpha=0.1, label='10-90%')
    

    # Customize plot
    # plt.ylim(0, 24)
    plt.xlabel('Time of Day (Hour)')
    plt.ylabel(f'Mean {column}')
    plt.title(f'Mean {column} Over Time of Day with 25-75% and 10-90% Intervals')
    plt.legend()
    # plt.show()
    return plt.gcf(), rolling_stats


def categorize_time_of_day(df, timestamp_column):
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Create a new column for the category
    df['time_category'] = df[timestamp_column].dt.hour // 2 + 1
    
    return df
def calculate_mean_by_category(df, value_column):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['day'] = df['Timestamp'].dt.date
    daily_sums = df.groupby(['time_category', 'day'])[value_column].sum().reset_index()
    mean_values = daily_sums.groupby('time_category')[value_column].mean().reset_index()
    return mean_values

def plot_basal_bolus_ratio(data):

    categorize_time_of_day(data, 'Timestamp')
    prumery_basal = calculate_mean_by_category(data, 'Basal_Rate')
    prumery_user_bolus = calculate_mean_by_category(data, 'Bolus')
    prumery_auto_bolus = calculate_mean_by_category(data, 'Manual_Bolus')


    # Merging DataFrames on time_category to get a combined DataFrame
    df_combined = prumery_basal.merge(prumery_auto_bolus, on='time_category').merge(prumery_user_bolus, on='time_category')

    # Adding Bolus and Manual_Bolus to create a combined Bolus category
    # df_combined['Total_Bolus'] = df_combined['Automatic_Bolus'] + df_combined['Manual_Bolus']
    # Calculate percentages for each category within each time_category
    df_combined['Total'] = df_combined['Basal_Rate'] + df_combined['Bolus']
    df_combined['Basal_Rate_Percent'] = (df_combined['Basal_Rate'] / df_combined['Total']) * 100
    df_combined['Total_Bolus_Percent'] = (df_combined['Bolus'] / df_combined['Total']) * 100

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bar chart
    ax.bar(df_combined['time_category'], df_combined['Basal_Rate_Percent'], label='Bazál')
    ax.bar(df_combined['time_category'], df_combined['Total_Bolus_Percent'], 
        bottom=df_combined['Basal_Rate_Percent'], label='Bolus')

    # Adjust x-axis ticks to be centered between bars
    ax.set_xticks(df_combined['time_category'] - 0.5)
    ax.set_xticklabels(df_combined['time_category'] * 2-2)

    # Labeling and formatting
    ax.set_xlabel('Čas (h)')
    ax.set_ylabel('Procenta (%)')
    # ax.set_title('P')
    ax.legend()

    # Display plot
    plt.tight_layout()
    plt.show()

def plot_user_auto_bolus_ratio(data):
    df_combined = get_auto_user_ratio(data)
    

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the Auto bars
    ax.bar(df_combined['time_category'], df_combined['Auto_Rate_Percent'], 
       label='Auto', color='blue')

    # Create the Manual bars on top of the Auto bars
    ax.bar(df_combined['time_category'], df_combined['User_Rate'], 
        bottom=df_combined['Auto_Rate_Percent'], 
        label='Manual', color='orange')

    # Adjust x-axis ticks to be centered between bars
    ax.set_xticks(df_combined['time_category'] - 0.5)
    ax.set_xticklabels(df_combined['time_category'] * 2-2)

    # Labeling and formatting
    ax.set_xlabel('Čas (h)')
    ax.set_ylabel('Procenta (%)')
    # ax.set_title('P')
    ax.legend()

    # Display plot
    plt.tight_layout()
    plt.show()


def get_basal_bolus_ratio(data):

    categorize_time_of_day(data, 'Timestamp')
    prumery_basal = calculate_mean_by_category(data, 'Basal_Rate')
    prumery_user_bolus = calculate_mean_by_category(data, 'Bolus')
    prumery_auto_bolus = calculate_mean_by_category(data, 'Manual_Bolus')


    # Merging DataFrames on time_category to get a combined DataFrame
    df_combined = prumery_basal.merge(prumery_auto_bolus, on='time_category').merge(prumery_user_bolus, on='time_category')

    # Adding Bolus and Manual_Bolus to create a combined Bolus category
    # df_combined['Total_Bolus'] = df_combined['Automatic_Bolus'] + df_combined['Manual_Bolus']
    # Calculate percentages for each category within each time_category
    df_combined['Total'] = df_combined['Basal_Rate'] + df_combined['Bolus']
    df_combined['Basal_Rate_Percent'] = (df_combined['Basal_Rate'] / df_combined['Total']) * 100
    df_combined['Total_Bolus_Percent'] = (df_combined['Bolus'] / df_combined['Total']) * 100

    return df_combined

def get_auto_user_ratio(data):
    categorize_time_of_day(data, 'Timestamp')
    prumery_auto_bolus = calculate_mean_by_category(data, 'Automatic_Bolus')
    prumery_user_bolus = calculate_mean_by_category(data, 'Manual_Bolus')
    prumery_bolus = calculate_mean_by_category(data, 'Bolus')

    # Merging DataFrames on time_category to get a combined DataFrame
    df_combined = prumery_auto_bolus.merge(prumery_user_bolus, on='time_category').merge(prumery_bolus, on='time_category')


    df_combined['Auto_Rate_Percent'] = (df_combined['Automatic_Bolus'] / df_combined['Bolus']) * 100
    df_combined['User_Rate'] = (df_combined['Manual_Bolus'] / df_combined['Bolus']) * 100
    return(df_combined)

import streamlit as st
import altair as alt

def plot_insulin_statistics(df_combined, first_category_name, second_category_name, first_legend_name, second_legend_name):
            # st.session_state.df_final
            # df_combined = get_basal_bolus_ratio(complete_df)
            # Create a new column for hour ranges
            time_range_mapping = {
            1: '0-2',
            2: '2-4',
            3: '4-6',
            4: '6-8',
            5: '8-10',
            6: '10-12',
            7: '12-14',
            8: '14-16',
            9: '16-18',
            10: '18-20',
            11: '20-22',
            12: '22-24'
            }

            # Map the values using a new column 'time_range'
            df_combined['time_range'] = df_combined['time_category'].map(time_range_mapping)
            # Define the ordered categories
            ordered_categories = ['0-2', '2-4', '4-6', '6-8', '8-10', 
                                '10-12', '12-14', '14-16', '16-18', 
                                '18-20', '20-22', '22-24']

            # Ensure that time_range is a categorical type with the desired order
            df_combined['time_range'] = pd.Categorical(df_combined['time_range'], 
                                                        categories=ordered_categories, 
                                                        ordered=True)

            # Melt the DataFrame
            df_melted = df_combined.melt(id_vars='time_range', 
                                        value_vars=[first_category_name, second_category_name],
                                        var_name='Rate_Type', 
                                        value_name='Percentage')
            df_melted['Rate_Type'] = df_melted['Rate_Type'].replace({
                first_category_name: first_legend_name,
                second_category_name: second_legend_name,
            })  

            # Create the Altair chart with ordered x-axis
            chart = alt.Chart(df_melted).mark_bar().encode(
                x=alt.X('time_range:O', title='Čas (h)', sort=ordered_categories),  # Explicit sorting
                y=alt.Y('Percentage:Q', title='Procenta (%)', stack='zero'),
                color=alt.Color('Rate_Type:N', title='Typ Inzulinu', 
                legend=alt.Legend(orient='top')),
            ).properties(
                width=600,
                height=250
            )
            return(chart)


def plot_day(df, date):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    day_df = df.loc[df['Timestamp'].dt.date == pd.Timestamp(date).date()]

    # print(day_df.head())
    # print(str(pd.Timestamp(f'{month}-{day}')))

    wrong_boluses = detect_concurrent_boluses_daily(day_df)
    wrong_boluses = [ts.hour + ts.minute / 60 for ts in pd.to_datetime(wrong_boluses)]

    time = day_df['Timestamp'].dt.hour + (day_df['Timestamp'].dt.minute)/60
    basal = day_df['Basal_Rate']
    bolus_aut = day_df['Automatic_Bolus']
    bolus_man = day_df['Manual_Bolus']
    glc = day_df['GLC']

    # Create a figure and axes for the plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)
    length = len(time)
    boundaries = np.tile([4.7, 7.2, 10], (length, 1)).T
    
    if wrong_boluses:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=wrong_boluses, color='grey', linestyle='--', linewidth=3)
    
    # Plot GLC
    ax1.plot(time, glc, color='blue', label='GLC')
    ax1.plot(time, boundaries[0], color='lime')
    ax1.plot(time, boundaries[1], color='lime', linestyle='--')
    ax1.plot(time, boundaries[2], color='lime')
    ax1.set_ylabel('GLC')
    ax1.legend(loc='upper left')
    ax1.set_title('Glycemic Levels (GLC)')
    ax1.set_ylim(0, 23)

    # Plot Bolus
    ax2.plot(time, bolus_aut, color='red', label='Bolus')
    ax2.plot(time, bolus_man, color='orange', label='Manual Bolus')
    ax2.set_ylabel('Bolus')
    ax2.legend(loc='upper left')
    ax2.set_title('Bolus Insulin')

    # Plot Basal Rate
    ax3.plot(time, basal, color='green', label='Basal Rate')
    ax3.set_ylabel('Basal Rate')
    ax3.legend(loc='upper left')
    ax3.set_title('Basal Insulin Rate')

    # Customize the x-axis
    plt.xlabel('Time of Day')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the plots
    # plt.show()
    return plt.gcf()

def detect_concurrent_boluses_daily(data, tolerance=15/60,return_categories=False):
    data = data.reset_index(drop=True)
    concurrent_timestamps = []
    timestamps = pd.to_datetime(data['Timestamp'])
    
    auto_bolus_timestamps = pd.to_datetime(data.loc[data['Automatic_Bolus'] > 0, 'Timestamp'])
    user_bolus_timestamps = pd.to_datetime(data.loc[data['Manual_Bolus'] > 0, 'Timestamp'])
    
    # Calculate the minimum timestamp for time normalization
    min_timestamp = timestamps.min()
    
    # Calculate time in hours from the earliest timestamp
    time_in_hours = (timestamps - min_timestamp).dt.total_seconds() / 3600
    auto_bolus_time_in_hours = (auto_bolus_timestamps - min_timestamp).dt.total_seconds() / 3600
    user_bolus_time_in_hours = (user_bolus_timestamps - min_timestamp).dt.total_seconds() / 3600
    
    # Iterate over auto bolus times
    for auto_time in auto_bolus_time_in_hours:
        # Find any user bolus times within the specified tolerance
        concurrent = user_bolus_time_in_hours[
            (user_bolus_time_in_hours >= auto_time - tolerance) & 
            (user_bolus_time_in_hours <= auto_time + tolerance)
        ]
        
        # If there are concurrent user boluses, check glucose level near those times
        if not concurrent.empty:
            for user_time in concurrent:
                # Find the glucose level near the auto_time (within tolerance)
                closest_index = (abs(time_in_hours - auto_time)).idxmin()
                
                # Check if the closest_index is within valid range
                if 0 <= closest_index < len(data['GLC']):
                    glucose_level = data['GLC'].iloc[closest_index]
                    
                    # Only save the timestamp if the glucose level is above 10
                    if glucose_level > 10:
                        if return_categories == False:
                            concurrent_timestamps.append(data['Timestamp'].iloc[closest_index])
                        else:
                            concurrent_timestamps.append((data['time_category'].iloc[closest_index]))
    
    return concurrent_timestamps


def plot_wrong_boluses(data):
    result = detect_concurrent_boluses_daily(data, return_categories=True)
                
    # Create a DataFrame from the list
    result = pd.DataFrame(result, columns=['time_category'])

    # Define the mapping for time ranges
    time_range_mapping = {
        1: '0-2',
        2: '2-4',
        3: '4-6',
        4: '6-8',
        5: '8-10',
        6: '10-12',
        7: '12-14',
        8: '14-16',
        9: '16-18',
        10: '18-20',
        11: '20-22',
        12: '22-24'
    }

    # Map time_category to time_range
    result['time_range'] = result['time_category'].map(time_range_mapping)

    # Count occurrences of each time range
    counts = result['time_range'].value_counts().reset_index()
    counts.columns = ['time_range', 'count']

    # Define the ordered categories for time_range
    ordered_categories = ['0-2', '2-4', '4-6', '6-8', '8-10', 
                        '10-12', '12-14', '14-16', '16-18', 
                        '18-20', '20-22', '22-24']

    # Ensure that time_range is a categorical type with the desired order
 

    # # Create the histogram with Altair
    # histogram = alt.Chart(counts).mark_bar().encode(
    #     x=alt.X('time_range:O', title='Time Range (hours)', sort=ordered_categories),
    #     y=alt.Y('count:Q', title='Count'),
    #     tooltip=['time_range:O', 'count:Q']
    # ).properties(
    #     title='Histogram of Time Ranges',
    #     width=600,
    #     height=400
    # )
    # Reindex counts DataFrame to include all time ranges with a count of 0 where necessary
    all_counts = counts.set_index('time_range').reindex(ordered_categories, fill_value=0).reset_index()

    # Ensure the 'time_range' column is a categorical type with the correct order
    all_counts['time_range'] = pd.Categorical(all_counts['time_range'], 
                                            categories=ordered_categories, 
                                            ordered=True)

    # Create the histogram with Altair
    histogram = alt.Chart(all_counts).mark_bar().encode(
        x=alt.X('time_range:O', title='Denní doba (h)', sort=ordered_categories),
        y=alt.Y('count:Q', title='Počet výskytů'),
        tooltip=['time_range:O', 'count:Q']
    ).properties(
        # title='Histogram of Time Ranges (Including Zero Counts)',
        width=600,
        height=250
    )

    return(histogram)
