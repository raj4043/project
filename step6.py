import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re


# Load data from CSV file
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


# NLP Query Module
@st.cache_resource
def load_nlp_model():
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def nlp_query_to_pandas(query):
    model, tokenizer = load_nlp_model()
    input_text = f"translate English to pandas DataFrame query: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    generated_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    st.write(f"Generated Query: {generated_query}")  # Debugging line

    # Ensure the query is compatible with pandas
    generated_query = re.sub(r'SELECT \* WHERE ', '', generated_query, flags=re.IGNORECASE).strip()

    # Extract conditions from the query
    conditions = re.findall(r'([a-zA-Z\s]+)\s*(==|>|<|>=|<=|!=)\s*["\']?([^"\']+)["\']?', generated_query)

    # Build the pandas query string
    query_parts = []
    for column, operator, value in conditions:
        column = column.strip()  # Remove any leading or trailing whitespace
        column = semantic_mappings.get(column, column)  # Translate to semantic mapping
        if value.replace('.', '', 1).isdigit():  # Check if value is a number, considering floats
            query_parts.append(f'`{column}` {operator} {value}')
        else:
            query_parts.append(f'`{column}` {operator} "{value}"')

    pandas_query = ' & '.join(query_parts)

    st.write(f"Pandas Query: {pandas_query}")  # Debugging line

    # Check for top N query pattern
    top_n_match = re.search(r'top\s+(\d+)\s+sales', query, re.IGNORECASE)
    top_n = None
    if top_n_match:
        top_n = int(top_n_match.group(1))

    return pandas_query.strip(), top_n


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
st.title("NLP Query Module")

# NLP Query Interface
st.header("NLP Query Module")
user_query = st.text_input("Enter your query (e.g., 'Show sales in Los Angeles' or 'Show top 5 sales')")
query_button = st.button("Run Query")

if query_button:
    try:
        structured_query, top_n = nlp_query_to_pandas(user_query)
        st.write(f"Structured Query: {structured_query}")

        # Execute the structured query on the dataset
        query_result = semantic_data.query(structured_query) if structured_query else semantic_data

        # Apply ordering and limit for top N sales
        if top_n:
            query_result = query_result.sort_values(by='Sales Amount', ascending=False).head(top_n)

        st.write(query_result)
    except ValueError as ve:
        st.error(f"Query failed: {ve}")
    except Exception as e:
        st.error(f"Query failed: {e}")
