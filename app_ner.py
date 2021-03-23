import aspect_based_sentiment_analysis as absa
import streamlit as st

from app import write_header, preprocess, classify
from ner_sent.run import predict_absa

@st.cache(allow_output_mutation=True)
def get_absa():
    absa_nlp = absa.load()
    return absa_nlp


if __name__ == '__main__':
    st.set_page_config(page_title='InterIIT 2021 End2End Solution', layout='wide')
    write_header()

    text = st.text_area("Leave your tweet here")
    button = st.button("update")
    absa_nlp = get_absa()

    if button:
        text = preprocess(text)
        category = classify(text)
        st.markdown(f"**Category:** {category}")

        if category == "MOBILE TECH":
            text = predict_absa(text, absa_nlp)
            st.markdown(f"**NER results:** {text}")
