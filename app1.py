# from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
# from langchain.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
genai.configure(api_key="AIzaSyDwmsQ-9hJTZV3ig0jioi2ZIAZxK3x9ras")
os.environ["GOOGLE_API_KEY"] = "AIzaSyDwmsQ-9hJTZV3ig0jioi2ZIAZxK3x9ras"
import streamlit as st
import pandas as pd
import os

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="Exp Bot")
st.title("Exp Bot")

uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are supported",
    on_change=clear_submit,
)

if not uploaded_file:
    st.warning("Upload the file to analyze")

if uploaded_file:
    df = load_data(uploaded_file)
    df2 = load_data("Exp Bot Success Metrics.xlsx")
    df3 = pd.merge(df, df2, on=['Experiment Title', 'Iteration'])

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Enter your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, streaming=True)

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df3,
        verbose=True,
        agent_type="zero-shot-react-description",
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)