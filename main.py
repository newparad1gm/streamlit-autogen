import pandas as pd
import streamlit as st
from autogen import AssistantAgent, config_list_from_json, ConversableAgent
from io import StringIO
from uuid import uuid4

# Read the CSV file
def load_data(file):
    print("Loading data...", file)
    try:
        return pd.read_csv(file, low_memory=False)
    except pd.errors.EmptyDataError:
        # If the first attempt fails, try multiple times with increasing delays
        import time
        for attempt in range(3):  # Try up to 3 times
            print("Loading data... attempt", attempt)
            try:
                return pd.read_csv(file)
            except pd.errors.EmptyDataError:
                if attempt < 2:  # Don't sleep after the last attempt
                    time.sleep(0.5 * (attempt + 1))  # Increase delay with each attempt
                else:
                    raise  # If all attempts fail, re-raise the exception

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    uploaded_file.seek(0)
    content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    temp_filepath = f"/tmp/{uuid4()}"
    with open(temp_filepath, "w") as f:
        f.write(content)
    df = load_data(temp_filepath)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Display a warning if no file is uploaded
if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Load the data
df = load_data(uploaded_file)

# Display basic information about the dataset
st.subheader("Dataset Information")
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")
st.write("Columns:", ", ".join(df.columns))

# Display the first few rows of the dataframe
st.subheader("Data Preview")
st.dataframe(df.head())

# Set up Autogen agents
config_list = config_list_from_json("OAI_CONFIG_LIST")
assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

# Define tools for data analysis
def describe_data():
    return df.describe().to_string()

def get_column_info(column_name: str):
    if column_name in df.columns:
        return f"Column '{column_name}':\n" \
               f"Type: {df[column_name].dtype}\n" \
               f"Unique values: {df[column_name].nunique()}\n" \
               f"Top 5 values: {df[column_name].value_counts().nlargest(5).to_dict()}"
    else:
        return f"Column '{column_name}' not found in the dataframe."

def calculate_correlation(column1: str, column2: str):
    if column1 in df.columns and column2 in df.columns:
        return df[column1].corr(df[column2])
    else:
        return "One or both columns not found in the dataframe."

def get_missing_values():
    return df.isnull().sum().to_dict()

def generate_summary_stats(column_name: str):
    if column_name in df.columns:
        return df[column_name].describe().to_dict()
    else:
        return f"Column '{column_name}' not found in the dataframe."
    
assistant.register_for_llm(name="describe_data", description="Describe the data")(describe_data)
assistant.register_for_llm(name="get_column_info", description="Get column information")(get_column_info)
assistant.register_for_llm(name="calculate_correlation", description="Calculate correlation between two columns")(calculate_correlation)
assistant.register_for_llm(name="get_missing_values", description="Get missing values")(get_missing_values)
assistant.register_for_llm(name="generate_summary_stats", description="Generate summary statistics for a column")(generate_summary_stats)

# Create a dictionary of tools
tools = {
    "describe_data": describe_data,
    "get_column_info": get_column_info,
    "calculate_correlation": calculate_correlation,
    "get_missing_values": get_missing_values,
    "generate_summary_stats": generate_summary_stats
}

# Create the user proxy agent
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

user_proxy.register_for_execution(name="describe_data")(describe_data)
user_proxy.register_for_execution(name="get_column_info")(get_column_info)
user_proxy.register_for_execution(name="calculate_correlation")(calculate_correlation)
user_proxy.register_for_execution(name="get_missing_values")(get_missing_values)
user_proxy.register_for_execution(name="generate_summary_stats")(generate_summary_stats)

# Streamlit UI
st.title("CSV Data Analysis with Autogen")

# Display the dataframe
st.subheader("Data Preview")
st.dataframe(df.head())

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to add message to conversation history
def add_to_history(role, content):
    st.session_state.conversation_history.append({"role": role, "content": content})

# Function to display conversation history
def display_conversation():
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Display conversation history
display_conversation()

# User input for analysis
user_query = st.chat_input("Enter your data analysis query:")

if user_query:
    # Add user query to conversation history
    add_to_history("user", user_query)

    # Render user query with user icon
    with st.chat_message("user"):
        st.write(user_query)

    def analyze_data(query):
        # Include conversation history in the prompt
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation_history])
        prompt = f"Conversation history:\n{history}\n\nAnalyze the following data based on this query: {query}"
        result = user_proxy.initiate_chat(assistant, message=prompt)
        return result.summary

    # Perform analysis and display results
    with st.chat_message("assistant"):
        result = analyze_data(user_query)
        st.write(result)
    
    # Add assistant's response to conversation history
    add_to_history("assistant", result)
