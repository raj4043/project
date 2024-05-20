import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import numpy as np

# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

data = load_data('Superstore.csv')

# Semantic mappings
semantic_mappings = {
    'Order Date': 'Date of Order',
    'Ship Date': 'Date of Shipment',
    'Ship Mode': 'Mode of Shipment',
    'Customer Name': 'Name of Customer',
    'Segment': 'Customer Segment',
    'Country': 'Country',
    'City': 'City',
    'State': 'State',
    'Postal Code': 'Postal Code',
    'Region': 'Region',
    'Product ID': 'Product Identifier',
    'Category': 'Product Category',
    'Sub-Category': 'Product Sub-Category',
    'Product Name': 'Product Name',
    'Sales': 'Sales Amount',
    'Quantity': 'Quantity Sold',
    'Discount': 'Discount Applied',
    'Profit': 'Profit Earned'
}

# Apply semantic mappings to data
semantic_data = data.rename(columns=semantic_mappings)

# Analysis and Insights Generation Module
def generate_insights(df):
    # Select relevant features and target
    features = ['Quantity Sold', 'Discount Applied', 'Profit Earned']
    target = 'Sales Amount'

    # Prepare the data
    X = df[features]
    y = df[target]

    # Handle missing values by filling with 0 (simplified for this example)
    X = X.fillna(0)
    y = y.fillna(0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for insights
    insights_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    return insights_df

# Insights Summarization and Ranking Module
def summarize_and_rank_insights(insights_df):
    # Rank the insights by the importance of the features
    ranked_insights = insights_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    ranked_insights['Rank'] = ranked_insights.index + 1
    return ranked_insights

# Custom CSS for styling
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        .sidebar .sidebar-content .sidebar-section {
            margin-top: 10px;
        }
        .stButton>button {
            background-color: #2e7bcf;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1e5bbf;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #2e7bcf;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stDataFrame {
            border: 1px solid #2e7bcf;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stHeader {
            color: #2e7bcf;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("Analysis and Insights Generation Module")

# Generate insights
insights_df = generate_insights(semantic_data)

# Display insights
st.header("Generated Insights")
st.write(insights_df)

# Summarize and rank insights
ranked_insights = summarize_and_rank_insights(insights_df)

# Display summarized and ranked insights
st.header("Summarized and Ranked Insights")
st.write(ranked_insights)

# Dashboard with visualizations
st.subheader("Dashboard")

# Sales Distribution by Category
fig1 = px.histogram(data, x='Category', title='Sales Distribution by Category')
st.plotly_chart(fig1)

# Sales Distribution Across Regions
fig2 = px.box(data, x='Region', y='Sales', title='Sales Distribution Across Regions')
st.plotly_chart(fig2)

# Sales vs Profit by Region
fig3 = px.scatter(data, x='Sales', y='Profit', color='Region', title='Sales vs Profit by Region')
st.plotly_chart(fig3)

# Sales Over Time
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['month_year'] = data['Order Date'].dt.to_period('M').astype(str)
sales_over_time = data.groupby('month_year')['Sales'].sum().reset_index()
fig4 = px.line(sales_over_time, x='month_year', y='Sales', title='Sales Over Time')
st.plotly_chart(fig4)
