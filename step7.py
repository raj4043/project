import streamlit as st
import pandas as pd

# Load data from CSV file
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

# Placeholder function for updating the model (for illustration purposes)
def update_model(feedback):
    # Append feedback to a file (simulating database storage)
    with open("feedback_log.txt", "a") as file:
        file.write(feedback + "\n")
    # Print feedback to simulate model update
    print("Updating model with feedback:", feedback)

# Load sample data
data = load_data('Superstore.csv')

# Display the data
st.title("Feedback and Continuous Learning Module")
st.header("Superstore Data")
st.write(data.head())

# User Feedback Form
st.header("User Feedback")
st.write("Please provide your feedback to help us improve the system.")

# Text input for feedback
feedback = st.text_area("Enter your feedback here")

# Submit button
if st.button("Submit Feedback"):
    if feedback:
        update_model(feedback)
        st.success("Thank you for your feedback!")
    else:
        st.error("Please enter your feedback before submitting.")

# Display current feedback
st.header("Current Feedback")
st.write("Feedback collected from users:")

# Read and display feedback from the file
try:
    with open("feedback_log.txt", "r") as file:
        feedback_data = file.readlines()
    if feedback_data:
        for line in feedback_data:
            st.write(f"- {line.strip()}")
    else:
        st.write("No feedback available.")
except FileNotFoundError:
    st.write("No feedback available.")
