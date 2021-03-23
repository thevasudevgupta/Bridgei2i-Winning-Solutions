
import torch
import streamlit as st
from transformers import MBartForConditionalGeneration, MBartTokenizer
# import aspect_based_sentiment_analysis as absa

# from ner import predict_absa
from preprocess import (
    compose,
    expand_words,
    basic_clean,
    remove_emojis,
    remove_stopwords,
)

MAX_TWEET_LENGTH = 280
SUMMARIZER_ID = "vasudevgupta/mbart-summarizer-interiit"


# @st.cache(allow_output_mutation=True)
# def get_absa():
#     absa_nlp = absa.load()
#     return absa_nlp

@st.cache(allow_output_mutation=True)
def get_summarization_agents():
    agents = {
        "model": MBartForConditionalGeneration.from_pretrained(SUMMARIZER_ID),
        "tokenizer": MBartTokenizer.from_pretrained(SUMMARIZER_ID)
    }
    return agents

# def perform_ner(text):
#     absa_nlp = get_absa()
#     out = predict_absa(text, absa_nlp)
#     return out

def write_header():
    st.title('InterIIT 2021 solution')
    st.markdown('''
        We have built an end2end pipeline as a solution.

        *Note: This app may take upto 5 minutes for initial setup*
    ''')

def summarize(text, summarize_button):

    def infer_bart_on_sample(sample, model, tokenizer, max_pred_length):

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        model.to(device)
        model.eval()

        sample = tokenizer(sample, return_tensors="pt", max_length=544, truncation=True)

        for k in sample:
            sample[k] = sample[k].to(device)

        out = model.generate(**sample, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], max_length=max_pred_length)
        sample = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print(sample)

        return sample

    agent = get_summarization_agents()
    if summarize_button and len(text) > 1:
        with st.spinner('summarizing ...'):    
            summary = infer_bart_on_sample(text, agent["model"], agent["tokenizer"], max_pred_length=44)
    return summary

def preprocess(text):
    args = [expand_words, basic_clean, remove_emojis, remove_stopwords]
    clean_func = compose(*args)
    return clean_func(text)


if __name__ == '__main__':
    st.set_page_config(page_title='InterIIT 2021 End2End Solution', layout='wide')
    write_header()

    text = st.text_area("Leave your article/tweet here")
    summarize_button = st.button("update")
    text = preprocess(text)

    # filter out article
    if len(text) > 280:
        text = summarize(text, summarize_button)
        st.markdown(text)

    # ner_output = perform_ner(text)
    # st.markdown(f"NER: {ner_output}")
