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
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

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
#     df2 = load_data("Exp Bot Success Metrics.xlsx")
#     df3 = pd.merge(df, df2, on=['Experiment Title', 'Iteration'])

df = load_data("TPre TfW Experiments.csv")
df2 = load_data("Exp Bot Success Metrics.xlsx")
df3 = pd.merge(df, df2, on=['Experiment Title', 'Iteration'])

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Enter your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if "?" not in prompt:
        
        paragraph = """4 experiments led to an increase in TPre trial initiations.
                        7 copy experiments were rolled out in July.
                        The average duration for an experiment to see statistically significant results is 14 days.
                        The commonly used success metrics used by TPre experimenters are "Users clicking on the button" and "Users who started the trial".
                        The guidelines for identifying relevant success metrics for your experiment are that they should align with business goals, be quantifiable and measurable, be sensitive to changes and consistent across flights.
                        The common guardrail conditions for growth experiments are that the Core metrics should not exhibit any regressions in Teams features.
                        To design an experiment to test the performance of x copy variants the success metrics should be Click Through Rate for the feature copy.
                        Even if the metrics show no negative movements, the experiment can be considered as successful only if the movement in the metrics is statistically significant.
                    """
        
        sentences = nltk.sent_tokenize(paragraph)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        def find_best_answer(question):
            question_tfidf = vectorizer.transform([question])
            similarities = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
            best_match_index = similarities.argmax()
            return sentences[best_match_index]

        best_answer = find_best_answer(prompt)
        with st.chat_message("assistant"):
            time.sleep(2)
            st.session_state.messages.append({"role": "assistant", "content": best_answer})
            st.write(best_answer)
        
    else:
        if "success metric" in prompt:
            df_to_use = df2
        else:
            df_to_use = df
    
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, streaming=True)
    
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
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
