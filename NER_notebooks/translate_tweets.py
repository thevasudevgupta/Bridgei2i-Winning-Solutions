import torch
import os
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBartTokenizer

FILE_PATH = "Clean_fie_without_stopwords_removal.csv"

def translate(sample, model, tokenizer, max_pred_length):

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


if __name__ == '__main__':

    data = load_dataset("csv", data_files=FILE_PATH)["train"]
    print(data)

    data = data.filter(lambda x: type(x["Tweet"]) == str)
    print(data)

    data = data.map(lambda x: {"length": len(x["Tweet"].split())})
    print(max(data["length"]), sum(data["length"])/ len(data), min(data["length"]))

    fn_kwargs = {
        "model": MBartForConditionalGeneration.from_pretrained("vasudevgupta/mbart-iitb-hin-eng"),
        "tokenizer": MBartTokenizer.from_pretrained("vasudevgupta/mbart-iitb-hin-eng"),
        "max_pred_length": 96,
    }

    data = data.map(lambda x: {"CleanedTweet": translate(x["Tweet"], **fn_kwargs)})
    data.to_csv(os.path.join("cleaned", FILE_PATH))
