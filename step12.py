import streamlit as st
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import plotly.express as px

# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

# Define roles and permissions
roles_permissions = {
    'admin': ['view_all', 'edit', 'view_sensitive'],
    'analyst': ['view_all', 'view_sensitive'],
    'viewer': ['view_all'],
}

# Data masking function
def mask_data(df, columns_to_mask):
    for column in columns_to_mask:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: '*****' if not pd.isnull(x) else x)
    return df

# Hash password function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User data (In a real application, use a secure database)
users = {
    'admin': {'password': hash_password('adminpass'), 'role': 'admin'},
    'analyst': {'password': hash_password('analystpass'), 'role': 'analyst'},
    'viewer': {'password': hash_password('viewerpass'), 'role': 'viewer'},
}

# Login function
def login(username, password):
    if username in users and users[username]['password'] == hash_password(password):
        return users[username]['role']
    return None

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None

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
st.title("Insights360")

if not st.session_state['logged_in']:
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        role = login(username, password)
        if role:
            st.session_state['logged_in'] = True
            st.session_state['role'] = role
            st.sidebar.success(f"Logged in as {username} with role {role}")
        else:
            st.sidebar.error("Invalid username or password")
else:
    st.sidebar.header("Logout")
    logout_button = st.sidebar.button("Logout")

    if logout_button:
        st.session_state['logged_in'] = False
        st.session_state['role'] = None
        st.sidebar.success("Logged out successfully")

if st.session_state['logged_in']:
    role = st.session_state['role']

    # Load data
    data = load_data('Superstore.csv')

    # Permissions and Security Module
    st.header("Permissions and Security Module")

    # User controls for data masking
    st.sidebar.header("Data Masking Options")
    columns_to_mask = st.sidebar.text_input("Columns to mask (comma-separated)", value="Customer Name, Sales, Profit")

    # Process the user input to get the list of columns to mask
    columns_to_mask_list = [col.strip() for col in columns_to_mask.split(',')]

    # Mask data based on user selections
    masked_data = mask_data(data.copy(), columns_to_mask_list)

    st.header("Data")
    st.write(masked_data)

    if 'edit' in roles_permissions[role]:
        st.header("Edit Data")
        row_index = st.number_input("Row index to edit", min_value=0, max_value=len(data) - 1)
        column_name = st.selectbox("Column to edit", data.columns)
        new_value = st.text_input(f"New value for {column_name}")
        update_button = st.button("Update")

        if update_button:
            data.at[row_index, column_name] = new_value
            data.to_csv('Superstore.csv', index=False)
            st.success("Data updated successfully")

    # Semantic Layer Generation Module
    st.title("Semantic Layer Generation Module")

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

    # Analysis and Insights Generation Module
    st.title("Analysis and Insights Generation Module")

    def generate_insights(df):
        # Select relevant features and target
        features = ['Quantity Sold', 'Discount Applied', 'Profit Earned']
        target = 'Sales Amount'

        # Prepare the data
        X = df[features].copy()
        y = df[target].copy()

        # Handle missing values by filling with 0 (simplified for this example)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.to_numeric(y, errors='coerce').fillna(0)

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

    def summarize_and_rank_insights(insights_df):
        # Rank the insights by the importance of the features
        ranked_insights = insights_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        ranked_insights['Rank'] = ranked_insights.index + 1
        return ranked_insights

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

    # NLP Query Module
    st.title("NLP Query Module")

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

    # NLP Query Interface
    st.header("NLP Query Module")
    user_query = st.text_input("Enter your query (e.g., 'Show sales in Los Angeles' or 'Show top 5 sales')")
    query_button = st.button("Run NLP Query")

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

    # Feedback and Continuous Learning Module
    st.title("Feedback and Continuous Learning Module")
    st.header("Superstore Data")
    st.write(data.head())

    # User Feedback Form
    st.header("User Feedback")
    st.write("Please provide your feedback to help us improve the system. Enter new data points as comma-separated values for the columns: Order Date, Ship Date, Ship Mode, Customer Name, Segment, Country, City, State, Postal Code, Region, Product ID, Category, Sub-Category, Product Name, Sales, Quantity, Discount, Profit")

    # Text input for feedback
    feedback = st.text_area("Enter your feedback here")

    # Submit button
    if st.button("Submit Feedback"):
        if feedback:
            with open("feedback_log.csv", "a") as file:
                file.write(feedback + "\n")
            st.success("Thank you for your feedback! The model will be updated with this new data.")

            # Update the model with the new data
            new_data = pd.read_csv("feedback_log.csv", encoding='ISO-8859-1')
            combined_data = pd.concat([data, new_data])
            semantic_data = combined_data.rename(columns=semantic_mappings)

            insights_df = generate_insights(semantic_data)
            ranked_insights = summarize_and_rank_insights(insights_df)

            st.header("Updated Insights")
            st.write(ranked_insights)

            # Update dashboard with new data
            st.subheader("Updated Dashboard")

            # Sales Distribution by Category
            fig1 = px.histogram(combined_data, x='Category', title='Sales Distribution by Category')
            st.plotly_chart(fig1)

            # Sales Distribution Across Regions
            fig2 = px.box(combined_data, x='Region', y='Sales', title='Sales Distribution Across Regions')
            st.plotly_chart(fig2)

            # Sales vs Profit by Region
            fig3 = px.scatter(combined_data, x='Sales', y='Profit', color='Region', title='Sales vs Profit by Region')
            st.plotly_chart(fig3)

            # Sales Over Time
            combined_data['Order Date'] = pd.to_datetime(combined_data['Order Date'])
            combined_data['month_year'] = combined_data['Order Date'].dt.to_period('M').astype(str)
            sales_over_time = combined_data.groupby('month_year')['Sales'].sum().reset_index()
            fig4 = px.line(sales_over_time, x='month_year', y='Sales', title='Sales Over Time')
            st.plotly_chart(fig4)

        else:
            st.error("Please enter your feedback before submitting.")

    # Display current feedback
    st.header("Current Feedback")
    st.write("Feedback collected from users:")

    # Read and display feedback from the file
    try:
        feedback_data = pd.read_csv("feedback_log.csv", encoding='ISO-8859-1')
        st.write(feedback_data)
    except FileNotFoundError:
        st.write("No feedback available.")
