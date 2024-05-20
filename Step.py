import streamlit as st
import pandas as pd
import hashlib


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
        }
        .stButton>button:hover {
            background-color: #1e5bbf;
            color: white;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #2e7bcf;
        }
        .stDataFrame {
            border: 1px solid #2e7bcf;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("Permissions and Security Module")

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

    # User controls for data masking
    st.sidebar.header("Data Masking Options")
    columns_to_mask = st.sidebar.text_input("Columns to mask (comma-separated)", value="Customer Name, Sales, Profit")

    # Process the user input to get the list of columns to mask
    columns_to_mask_list = [col.strip() for col in columns_to_mask.split(',')]

    # Mask data based on user selections
    masked_data = mask_data(data, columns_to_mask_list)

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
