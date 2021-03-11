import torch

import re
from datasets import load_dataset

def preprocess_article(article, sep_token="</s>"):

    def remove_sentences_with_links(text):
        text = text.split("\n")
        for sent in text:
            status = False
            for element in sent.split():
                if ("http:/" in element) or ("www." in element) or (".com" in element):
                    status = True
            if status:
                text.remove(sent)
        return ". ".join(text).strip() # `\n` will add so many `<sep>` token

    def remove_hash_tags(text):
        text = re.sub("#\S+", "", text)
        return text

    article = re.sub("\n\n", sep_token, article)
    article = remove_sentences_with_links(article)
    article = remove_hash_tags(article)

    # +1 /d
    return article


class DataLoader(object):

    def __init__(self, tokenizer, args):

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_length = args.max_length
        self.max_target_length = args.max_target_length

        self.file_path = args.file_path

        self.tokenizer = tokenizer

        self.sep_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.sep_token_id)

    def setup(self):

        data = load_dataset("csv", data_files=self.file_path)["train"]
        data = data.map(lambda x: {"article_length": len(x["Text"].split())})
        data = data.map(lambda x: {"summary_length": len(x["Headline"].split())})

        data = data.map(lambda x: {"CleanedText": preprocess_article(x["Text"], self.sep_token)})
        data = data.map(lambda x: {"CleanedHeadline": x["Headline"]}) # translate it

        print("Samples with article length > 560 are", data.filter(lambda x: x["article_length"] > 560))

        lengths = [len(data)-600, 600]
        tr_dataset, val_dataset = torch.utils.data.random_split(data, lengths=lengths)

        return tr_dataset, val_dataset

    def train_dataloader(self, tr_dataset):
        return torch.utils.data.DataLoader(
            tr_dataset,
            pin_memory=True,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def val_dataloader(self, val_dataset):
        return torch.utils.data.DataLoader(
            val_dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

    def collate_fn(self, features):
        # update your collate function here
        article = [f["CleanedText"] for f in features]
        summary = [f["CleanedHeadline"] for f in features]

        # src_lang will be dummy
        features = self.tokenizer.prepare_seq2seq_batch(
            src_texts=article, src_lang="hi_IN", tgt_lang="en_XX", tgt_texts=summary, truncation=True, max_length=self.max_length, max_target_length=self.max_target_length
        )

        return features


if __name__ == '__main__':

    from transformers import MBartTokenizer

    class args:
        batch_size: int = 2
        num_workers: int = 2
        max_length: int = 512
        max_target_length: int = 20
        file_path: str = "data/dev_data_article.csv"

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    dl = DataLoader(tokenizer, args)
    tr_dataset, val_dataset = dl.setup()
    
    tr_dataset = dl.train_dataloader(tr_dataset)
    val_dataset = dl.val_dataloader(val_dataset)

    for ds in val_dataset:
        pass
    print(tokenizer.convert_ids_to_tokens(ds.input_ids[0]))
    print(tokenizer.convert_ids_to_tokens(ds.labels[0]))
    print(dl.sep_token)
