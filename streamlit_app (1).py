
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Function to load cleaned dataframes directly from files
@st.cache_data
def load_cleaned_dataframes():
    cleaned_dataframes = {}
    try:
        cleaned_dataframes['Aggregated_insurance'] = pd.read_parquet('/content/Aggregated_insurance_cleaned.parquet')
        cleaned_dataframes['Aggregated_transaction'] = pd.read_parquet('/content/Aggregated_transaction_cleaned.parquet')
        cleaned_dataframes['Aggregated_user'] = pd.read_parquet('/content/Aggregated_user_cleaned.parquet')
        cleaned_dataframes['Map_insurance'] = pd.read_parquet('/content/Map_insurance_cleaned.parquet')
        cleaned_dataframes['Map_transaction'] = pd.read_parquet('/content/Map_transaction_cleaned.parquet')
        cleaned_dataframes['Map_user'] = pd.read_parquet('/content/Map_user_cleaned.parquet')
        cleaned_dataframes['Top_transaction'] = pd.read_parquet('/content/Top_transaction_cleaned.parquet')
        cleaned_dataframes['Top_user'] = pd.read_parquet('/content/Top_user_cleaned.parquet')
    except FileNotFoundError as e:
        st.error(f"Error loading dataframes: {e}. Please ensure the parquet files are in the /content/ directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading dataframes: {e}")
        return None

    if cleaned_dataframes:
        return cleaned_dataframes
    else:
        st.error("Cleaned dataframes could not be loaded.")
        return None


cleaned_dataframes = load_cleaned_dataframes()

if cleaned_dataframes:
    st.title('PhonePe Transaction Insights Dashboard')

    # Remove sidebar selection and display all sections
    # st.sidebar.title('Dashboard Navigation')
    # dataframe_name = st.sidebar.selectbox('Choose a Dataframe', list(cleaned_dataframes.keys()))
    # df = cleaned_dataframes[dataframe_name]

    # Display sections for all dataframes
    for dataframe_name, df in cleaned_dataframes.items():
        st.header(f'Analyzing: {dataframe_name}')

        # Data Preview, Info, and Missing Values in an expander
        with st.expander(f"Explore Data: {dataframe_name}"):
            st.subheader('Data Preview')
            st.write(df.head())

            st.subheader('Missing Values')
            st.write(df.isnull().sum())

        st.header('Visualizations')

        if dataframe_name == 'Aggregated_transaction':
            st.subheader('Transaction Trends')
            # Ensure the columns exist before plotting
            if 'transaction_type' in df.columns and 'transaction_amount' in df.columns and 'transaction_count' in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('Transaction Type Share of Total Amount')
                    trans_type = df.groupby('transaction_type')['transaction_amount'].sum().reset_index()
                    fig, ax = plt.subplots()
                    ax.pie(trans_type['transaction_amount'], labels=trans_type['transaction_type'], autopct='%1.1f%%', startangle=90)
                    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)

                with col2:
                    st.subheader('Transaction Type Share of Total Count')
                    trans_type_count = df.groupby('transaction_type')['transaction_count'].sum().reset_index()
                    fig, ax = plt.subplots()
                    ax.pie(trans_type_count['transaction_count'], labels=trans_type_count['transaction_type'], autopct='%1.1f%%', startangle=90)
                    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)


                st.subheader('Aggregated Transaction Count and Amount by Year/Quarter')
                if all(col in df.columns for col in ['year', 'quarter', 'transaction_count', 'transaction_amount']):
                    agg_trans_df = df.copy()
                    agg_trans_df['year'] = pd.to_numeric(agg_trans_df['year'], errors='coerce')
                    agg_trans_df['quarter'] = pd.to_numeric(agg_trans_df['quarter'], errors='coerce')
                    agg_trans_df.dropna(subset=['year', 'quarter'], inplace=True)
                    agg_data_time_series = agg_trans_df.groupby(['year', 'quarter'])[['transaction_count', 'transaction_amount']].sum().reset_index()
                    agg_data_time_series['period'] = agg_data_time_series['year'].astype(int).astype(str) + ' Q' + agg_data_time_series['quarter'].astype(int).astype(str)
                    agg_data_time_series.sort_values(['year', 'quarter'], inplace=True)

                    fig, ax = plt.subplots(figsize=(14, 6))
                    sns.barplot(data=agg_data_time_series, x='period', y='transaction_count', palette='viridis', ax=ax)
                    plt.xticks(rotation=45)
                    ax.set_title('Aggregated Transaction Count Over Time')
                    ax.set_xlabel('Period')
                    ax.set_ylabel('Transaction Count')
                    st.pyplot(fig)

                    fig, ax = plt.subplots(figsize=(14, 6))
                    sns.barplot(data=agg_data_time_series, x='period', y='transaction_amount', palette='viridis', ax=ax)
                    plt.xticks(rotation=45)
                    ax.set_title('Aggregated Transaction Amount Over Time')
                    ax.set_xlabel('Period')
                    ax.set_ylabel('Transaction Amount')
                    st.pyplot(fig)

                else:
                    st.warning("Required columns for time series plot not found in Aggregated_transaction dataframe.")

            else:
                st.warning("Required columns for Aggregated_transaction visualizations not found.")

        elif dataframe_name == 'Map_transaction':
            st.subheader('Geographical Transaction Insights')
            if 'state_name' in df.columns and 'amount' in df.columns and 'count' in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader('Total Transaction Amount by State')
                    state_amount_agg = df.groupby('state_name')['amount'].sum().reset_index()
                    if not state_amount_agg.empty:
                        fig, ax = plt.subplots(figsize=(15, 7))
                        sns.barplot(data=state_amount_agg.sort_values('amount', ascending=False).head(10),
                                    x='state_name',
                                    y='amount',
                                    palette='viridis',
                                    ax=ax)
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                    else:
                        st.write("Map Transaction Amount data is empty after aggregation. Cannot generate plot.")

                with col2:
                    st.subheader('Total Transaction Count by State')
                    state_count_agg = df.groupby('state_name')['count'].sum().reset_index()
                    if not state_count_agg.empty:
                        fig, ax = plt.subplots(figsize=(15, 7))
                        sns.barplot(data=state_count_agg.sort_values('count', ascending=False).head(10),
                                    x='state_name',
                                    y='count',
                                    palette='viridis',
                                    ax=ax)
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                    else:
                        st.write("Map Transaction Count data is empty after aggregation. Cannot generate plot.")
            else:
                 st.warning("Required columns for Map_transaction visualizations not found.")


        elif dataframe_name == 'Aggregated_user':
            st.subheader('User Demographics and App Usage')
            if 'brand' in df.columns and 'usercount' in df.columns:
                 st.subheader('Top 10 Mobile Brands by User Count')
                 brand_user_agg = df.groupby('brand')['usercount'].sum().reset_index()
                 if not brand_user_agg.empty:
                     fig, ax = plt.subplots(figsize=(15, 7))
                     sns.barplot(data=brand_user_agg.sort_values('usercount', ascending=False).head(10),
                                 x='brand',
                                 y='usercount',
                                 palette='viridis',
                                 ax=ax)
                     plt.xticks(rotation=45, ha='right')
                     st.pyplot(fig)
                 else:
                     st.write("Aggregated User data is empty after aggregation. Cannot generate plot.")
            else:
                st.warning("Required columns ('brand', 'usercount') not found in Aggregated_user dataframe.")

        elif dataframe_name == 'Top_user':
            st.subheader('Top User Insights')
            if 'state_name' in df.columns and 'registered_users' in df.columns:
                st.subheader('Top 10 States by Registered Users')
                state_user_agg = df.groupby('state_name')['registered_users'].sum().reset_index()
                if not state_user_agg.empty:
                    fig, ax = plt.subplots(figsize=(15, 7))
                    sns.barplot(data=state_user_agg.sort_values('registered_users', ascending=False).head(10),
                                x='state_name',
                                y='registered_users',
                                palette='viridis',
                                ax=ax)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                else:
                    st.write("Top User data is empty after aggregation. Cannot generate plot.")
            else:
                 st.warning("Required columns ('state_name', 'registered_users') not found in Top_user dataframe.")

        elif dataframe_name == 'Aggregated_insurance':
            st.subheader('Insurance Trends')
            if all(col in df.columns for col in ['year', 'quarter', 'amount', 'count']):
                st.subheader('Aggregated Insurance Amount and Count by Year/Quarter')
                agg_insurance_df = df.copy()
                agg_insurance_df['year'] = pd.to_numeric(agg_insurance_df['year'], errors='coerce')
                agg_insurance_df['quarter'] = pd.to_numeric(agg_insurance_df['quarter'], errors='coerce')
                agg_insurance_df.dropna(subset=['year', 'quarter'], inplace=True)
                agg_insurance_time_series = agg_insurance_df.groupby(['year', 'quarter'])[['amount', 'count']].sum().reset_index()
                agg_insurance_time_series['period'] = agg_insurance_time_series['year'].astype(int).astype(str) + ' Q' + agg_insurance_time_series['quarter'].astype(int).astype(str)
                agg_insurance_time_series.sort_values(['year', 'quarter'], inplace=True)

                fig, ax = plt.subplots(figsize=(14, 6))
                sns.barplot(data=agg_insurance_time_series, x='period', y='amount', palette='viridis', ax=ax)
                plt.xticks(rotation=45)
                ax.set_title('Aggregated Insurance Amount Over Time')
                ax.set_xlabel('Period')
                ax.set_ylabel('Insurance Amount')
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(14, 6))
                sns.barplot(data=agg_insurance_time_series, x='period', y='count', palette='viridis', ax=ax)
                plt.xticks(rotation=45)
                ax.set_title('Aggregated Insurance Count Over Time')
                ax.set_xlabel('Period')
                ax.set_ylabel('Insurance Count')
                st.pyplot(fig)

            else:
                st.warning("Required columns for Aggregated_insurance time series plot not found.")

        elif dataframe_name == 'Map_insurance':
            st.subheader('Geographical Insurance Insights')
            if all(col in df.columns for col in ['state', 'metric']):
                st.subheader('Insurance Metric Distribution by State (Choropleth Map)')
                state_metric_agg = df.groupby('state')['metric'].sum().reset_index()
                if not state_metric_agg.empty:
                    # Use Plotly for choropleth map as it's better suited for geographic data
                    fig = px.choropleth(state_metric_agg,
                                        locations='state',
                                        locationmode='country names', # Adjust based on actual data if needed (e.g., 'USA-states')
                                        color='metric',
                                        hover_name='state',
                                        title='Total Insurance Metric by State')
                    st.plotly_chart(fig)
                else:
                    st.write("Map Insurance data is empty after aggregation. Cannot generate plot.")
            else:
                st.warning("Required columns ('state', 'metric') not found in Map_insurance dataframe for choropleth map.")

        elif dataframe_name == 'Map_user':
            st.subheader('Geographical User Insights')
            if all(col in df.columns for col in ['state', 'registered_users']):
                st.subheader('Registered Users Distribution by State (Choropleth Map)')
                state_user_agg = df.groupby('state')['registered_users'].sum().reset_index()
                if not state_user_agg.empty:
                    # Use Plotly for choropleth map
                    fig = px.choropleth(state_user_agg,
                                        locations='state',
                                        locationmode='country names', # Adjust based on actual data if needed
                                        color='registered_users',
                                        hover_name='state',
                                        title='Total Registered Users by State')
                    st.plotly_chart(fig)
                else:
                    st.write("Map User data is empty after aggregation. Cannot generate plot.")
            else:
                st.warning("Required columns ('state', 'registered_users') not found in Map_user dataframe for choropleth map.")


        elif dataframe_name == 'Top_transaction':
            st.subheader('Top Transaction Entity Insights')
            if all(col in df.columns for col in ['entity_name', 'amount', 'level', 'count']):
                st.subheader('Top Transaction Entities by Amount and Count (State Level)')
                # Filter for state level for a clearer visualization
                top_trans_state = df[df['level'] == 'state'].groupby('entity_name')[['amount', 'count']].sum().reset_index()
                if not top_trans_state.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(15, 7))
                        sns.barplot(data=top_trans_state.sort_values('amount', ascending=False).head(10),
                                    x='entity_name',
                                    y='amount',
                                    palette='viridis',
                                    ax=ax)
                        plt.xticks(rotation=45, ha='right')
                        ax.set_title('Top 10 States by Transaction Amount')
                        ax.set_xlabel('State')
                        ax.set_ylabel('Total Transaction Amount')
                        st.pyplot(fig)
                    with col2:
                        fig, ax = plt.subplots(figsize=(15, 7))
                        sns.barplot(data=top_trans_state.sort_values('count', ascending=False).head(10),
                                    x='entity_name',
                                    y='count',
                                    palette='viridis',
                                    ax=ax)
                        plt.xticks(rotation=45, ha='right')
                        ax.set_title('Top 10 States by Transaction Count')
                        ax.set_xlabel('State')
                        ax.set_ylabel('Total Transaction Count')
                        st.pyplot(fig)

                else:
                    st.write("Top Transaction data (state level) is empty after aggregation. Cannot generate plot.")
            else:
                 st.warning("Required columns ('entity_name', 'amount', 'level', 'count') not found in Top_transaction dataframe.")

else:
    st.warning("Cleaned dataframes are not loaded.")

