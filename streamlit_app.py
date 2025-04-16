import os
import time
import pickle
import faiss
import numpy as np
import pandas as pd
import streamlit as st

import google.generativeai as genai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from sklearn.feature_extraction.text import TfidfVectorizer

# -- Secure configuration --
genai.configure(api_key="AIzaSyBstPTDaXEUgh2nZuVf-Wa2nGV5sHkxmaw")
os.environ["GOOGLE_API_KEY"] = "AIzaSyBstPTDaXEUgh2nZuVf-Wa2nGV5sHkxmaw"

# -- Supported file readers --
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """Clear the Submit Button State"""
    st.session_state["submit"] = False

@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    """Load uploaded file with proper reader based on extension"""
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# -- App setup --
st.set_page_config(page_title="Exp Bot")
st.title("Exp Bot")

# uploaded_file = st.file_uploader(
#     "Upload a Data file",
#     type=list(file_formats.keys()),
#     help="Various File formats are supported",
#     on_change=clear_submit,
# )

# if not uploaded_file:
#     st.warning("Upload the file to analyze")

# if uploaded_file:
#     df = load_data(uploaded_file)
#     df2 = load_data("Exp_Bot_Success_Metrics.csv")
#     df3 = pd.merge(df, df2, on=['Experiment Title', 'Iteration'])

# -- Load and merge datasets --
df = load_data("TPre TfW Experiments.csv")
df2 = load_data("Exp_Bot_Success_Metrics.csv")
df3 = pd.merge(df, df2, on=['Experiment Title', 'Iteration'])

# -- Chat session setup --
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# -- Bot module selector --
case = st.sidebar.selectbox("Choose bot module:", ["Post-Hoc Analytics", "Exp FAQs", "Planning & Design Support", "Execution Support"])

# -- Display chat history --
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -- User prompt input --
if prompt := st.chat_input(placeholder="Enter your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if case == "Exp FAQs":
        # with st.chat_message("assistant"):
            # st.write("🚧 This functionality is under construction 🚧")
        # Load the TF-IDF model and FAISS index
        with open('tfidf_model.pkl', 'rb') as model_file:
            vectorizer = pickle.load(model_file)
        index = faiss.read_index('faiss_index.idx')
        with open('qa_data.pkl', 'rb') as qa_file:
            qa_data = pickle.load(qa_file)
            questions = qa_data['questions']
            answers = qa_data['answers']

        def find_best_answer(user_question):
            user_vector = vectorizer.transform([user_question]).toarray().astype(np.float32)
            _, I = index.search(user_vector, 1)
            return answers[I[0][0]]

        best_answer = find_best_answer(prompt)
        with st.chat_message("assistant"):
            time.sleep(2)
            st.session_state.messages.append({"role": "assistant", "content": best_answer})
            st.write(best_answer)

    elif case == "Post-Hoc Analytics":
        df_to_use = df2 if "success metric" in prompt.lower() else df
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, streaming=True)

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df_to_use,
            verbose=True,
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run([msg["content"] for msg in st.session_state.messages], callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    else:
        with st.chat_message("assistant"):
            st.write("🚧 This functionality is under construction 🚧")
