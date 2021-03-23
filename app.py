
import torch
import streamlit as st
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import MBartForConditionalGeneration, MBartTokenizer
except:
    logger.warning("Couldn't import mBART; it's expected if you are running `app_ner.py`")

import sys
sys.path.append("./text-cls")
import train_cls
import phoneme

from preprocess import (
    compose,
    expand_words,
    basic_clean,
    remove_emojis,
    remove_stopwords,
)

MAX_TWEET_LENGTH = 280 # no of characters

def eval_text(self, text, w2ind, device=torch.device("cuda")):
    sent = phoneme.conv_phoneme(text)
    sent = [w2ind[w] for w in sent][0: 500]
    sent = torch.LongTensor(sent).to(device)[None, :]
    pred_logits = self.forward(sent)
    pred_cls = pred_logits.max(1)[1].item()
    if pred_cls == 0:
        return "NON-MOBILE TECH"
    else:
        return "MOBILE TECH"

@st.cache(allow_output_mutation=True)
def get_classifier():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    w2ind = train_cls.load_pickle("text-cls/w2ind.pickle")
    model = train_cls.CNNModel(len(w2ind), 300, 100, [3, 4, 5], 0.5, 2).to(device)
    model.load_ckpt("text-cls/best_cls.ckpt", map_location=device)
    model.eval()
    return model, w2ind, device

@st.cache(allow_output_mutation=True)
def get_summarization_agents():
    agents = {
        "model": MBartForConditionalGeneration.from_pretrained("vasudevgupta/mbart-summarizer-interiit"),
        "tokenizer": MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    }
    return agents

def classify(text):
    model, w2ind, device = get_classifier()
    prediction = model.eval_text(text, w2ind, device)
    return prediction

def write_header():
    st.title('InterIIT 2021 solution')
    st.markdown('''
        We have built an end2end pipeline as our solution.

        *Note: This app may take upto 5 minutes for initial setup*
    ''')

def summarize(text, agent):

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

    if len(text) > 1:
        with st.spinner('summarizing ...'):    
            summary = infer_bart_on_sample(text, agent["model"], agent["tokenizer"], max_pred_length=44)
            if summary.startswith(r"Home\n"):
                summary = summary[6:]
    return summary

def preprocess(text):
    args = [expand_words, basic_clean, remove_emojis, remove_stopwords]
    clean_func = compose(*args)
    return clean_func(text)


if __name__ == '__main__':
    st.set_page_config(page_title='InterIIT 2021 End2End Solution', layout='wide')
    write_header()

    text = st.text_area("Leave your article/tweet here")
    button = st.button("update")

    agent = get_summarization_agents()

    if button:
        text = preprocess(text)
        category = classify(text)
        st.markdown(f"**Category:** {category}")

        # filter out article
        if len(text) > 280:
            text = summarize(text, agent)
            st.markdown(f"**Headline:** {text}")
