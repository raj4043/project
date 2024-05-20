import streamlit as st
import pandas as pd

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

# Query function
def query_data(query):
    return semantic_data.query(query)

# Streamlit UI
st.title("Semantic Layer Generation Module")

# Display semantic mappings
st.header("Semantic Mappings")
st.write(semantic_mappings)

# Display data with semantic layer
st.header("Data with Semantic Layer")
st.write(semantic_data)

# Query Interface
st.header("Query Data using Semantic Layer")
query = st.text_input("Enter your query (e.g., `City == \"Los Angeles\" and `Sales Amount` > 1000`)")
query_button = st.button("Run Query")

if query_button:
    try:
        query_result = query_data(query)
        st.write(query_result)
    except Exception as e:
        st.error(f"Query failed: {e}")
