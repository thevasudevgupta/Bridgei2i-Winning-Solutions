
import sys
sys.path.append("../summarization")
from data_utils.dataloader import preprocess_article, infer_bart_on_sample

import datasets
from transformers import MBartForConditionalGeneration, MBartTokenizer

FILE_PATH = "data/evaluation_data.csv" 
SUMMARIZER_ID = "../weights/mbart-finetuned-exp2-e2"

if __name__ == "__main__":

    tokenizer = MBartTokenizer.from_pretrained(SUMMARIZER_ID)
    model = MBartForConditionalGeneration.from_pretrained(SUMMARIZER_ID)

    dataset = datasets.load_dataset("csv", data_files=FILE_PATH)["train"]

    # filter out articles & get headlines
    dataset = dataset.filter(lambda x: x["Text_ID"].startswith("article"))
    dataset = dataset.map(lambda x: {"Text": preprocess_article(x["Text"])})
    dataset = dataset.map(lambda x: {"Headline_Generated_Eng_Lang": infer_bart_on_sample(x["Text"], model, tokenizer, max_pred_length=44)}, remove_columns=["Text"])
    print(dataset)

    dataset.to_csv("prediction_articles.csv")
