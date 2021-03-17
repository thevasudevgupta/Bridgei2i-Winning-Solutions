from gensim.summarization.summarizer import summarize
from datasets import load_dataset

def summarize_using_gensim(example):
    try:
        example["gensim_summary"] = summarize(example["CleanedText"], word_count=32)
    except:
        example["gensim_summary"] = None
    return example

if __name__ == "__main__":

    data = load_dataset("csv", data_files="results/predictions.csv")["train"]
    data = data.map(summarize_using_gensim)

    print(data.filter(lambda x: x["gensim_summary"] is not None))
    data.to_csv("gensim_summary.csv")
