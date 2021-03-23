# streamlit run app.py

import sys
sys.path.append("./summarization")
from data_utils.dataloader import infer_bart_on_sample

import streamlit as st
from transformers import MBartForConditionalGeneration, MBartTokenizer

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
        We have built an end2end pipeline as a solution.

        *Note: This app may take upto 5 minutes for initial setup*
    ''')

def summarize():
    agent = get_summarization_agents()

    article = st.text_area("Leave your article here")
    summarize_button = st.button("update")

    if summarize_button and len(article) > 1:
        with st.spinner('summarizing ...'):    
            summary = infer_bart_on_sample(article, agent["model"], agent["tokenizer"], max_pred_length=44)
            st.markdown(summary)

    return summarize


if __name__ == '__main__':
    st.set_page_config(page_title='InterIIT 2021 End2End Solution', layout='wide')
    write_header()
    summarize()
