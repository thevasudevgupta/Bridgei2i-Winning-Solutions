# streamlit run app.py

import streamlit as st
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers.pipelines import pipeline

SUMMARIZER_ID = "weights/mbart-finetuned-exp2-e2"

@st.cache(allow_output_mutation=True)
def get_summarization_agents():
    agents = {
        "model": MBartForConditionalGeneration.from_pretrained(SUMMARIZER_ID),
        "tokenizer": MBartTokenizer.from_pretrained(SUMMARIZER_ID)
    }
    return agents

def write_header():
    st.title('InterIIT 2021 solution')
    st.markdown('''
        - We have built an end2end pipeline as a solution.
    ''')

def summarize():
    summarization_agent = get_summarization_agents()

    article = st.text_area("Leave your article here")
    summarize_button = st.button("update")

    if summarize_button and len(article) > 1:
        summarizer = pipeline("summarization", model=summarization_agent["model"], tokenizer=summarization_agent["tokenizer"])
        summary = summarizer(article, max_length=32)
        st.markdown(summary)

    return summarize


if __name__ == '__main__':
    st.set_page_config(page_title='InterIIT 2021 End2End Solution', layout='wide')
    write_header()
    summarize()
