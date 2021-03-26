import torch

import re
from datasets import load_dataset, concatenate_datasets
from transformers import MBartForConditionalGeneration, MBartTokenizer

import random
from nltk.corpus import wordnet, stopwords

STOP_WORDS = stop_words = set(stopwords.words('english'))


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
        return "\n".join(text)

    def remove_hash_tags(text):
        text = re.sub("#\S+", "", text)
        return text

    article = remove_sentences_with_links(article)
    article = remove_hash_tags(article)
    # `\n` will add so many `<sep>` token
    article = re.sub("\n\n", sep_token, article)
    article = re.sub("\n", ".", article)
    return article.strip()


def get_noisy_sent(word_list, v=1):

    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in " qwertyuiopasdfghjklzxcvbnm"])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    n = int(v*len(word_list))
    not_stopwords = list(set([word for word in word_list if word not in STOP_WORDS]))
    rand_word_list = random.choices(not_stopwords, k=min(n, len(not_stopwords)))

    for word in rand_word_list:
        synonyms = get_synonyms(word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            word_list = [synonym if w == word else w for w in word_list]

    return ' '.join(word_list)


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

def translate(example, model, tokenizer, max_pred_length=32):
    example["CleanedHeadline"] = infer_bart_on_sample(example["Headline"], model, tokenizer, max_pred_length)
    return example


class DataLoader(object):

    def __init__(self, tokenizer, args):

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_length = args.max_length
        self.max_target_length = args.max_target_length

        self.file_path = args.file_path

        self.tokenizer = tokenizer
        self.sep_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.sep_token_id)

        self.seed = args.seed

    def setup(self, process_on_fly=True, n_augment=0):

        if process_on_fly:
            data = load_dataset("csv", data_files=self.file_path)["train"]
            data = data.map(lambda x: {"article_length": len(x["Text"].split())})
            data = data.map(lambda x: {"summary_length": len(x["Headline"].split())})

            data = data.map(lambda x: {"CleanedText": preprocess_article(x["cleaned"], self.sep_token)})

            data = data.map(lambda x: {"CleanedHeadline": x["Headline"]})
            fn_kwargs = {
                "model": MBartForConditionalGeneration.from_pretrained("vasudevgupta/mbart-iitb-hin-eng"),
                "tokenizer": MBartTokenizer.from_pretrained("vasudevgupta/mbart-iitb-hin-eng"),
                "max_pred_length": 32,
            }

            data = data.map(translate, fn_kwargs=fn_kwargs)
            data.to_csv(f"cleaned-{self.file_path}")

        else:
            data = load_dataset("csv", data_files=f"cleaned-{self.file_path}")["train"]

        data = data.filter(lambda x: x["article_length"] > 32 and x["summary_length"] > 1)

        removed_samples = data.filter(lambda x: type(x["CleanedHeadline"]) != str or type(x["CleanedText"]) != str)
        print(removed_samples["CleanedHeadline"])
        print(removed_samples["CleanedText"])

        data = data.filter(lambda x: type(x["CleanedHeadline"]) == str and type(x["CleanedText"]) == str)
        print("Dataset", data)

        # print("Samples with article length > 560 are", data.filter(lambda x: x["article_length"] > 560))

        data = data.train_test_split(test_size=600, shuffle=True, seed=self.seed)
        tr_dataset = data["train"].map(lambda x: {"split": "TRAIN"})
        val_dataset = data["test"].map(lambda x: {"split": "VALIDATION"})

        if n_augment > 0:
            print("AUGMENTING")
            tr_dataset = tr_dataset.map(lambda x: {"augmentation_status": "Not Augmented"})
            val_dataset = val_dataset.map(lambda x: {"augmentation_status": "Not Augmented"})
            noisy_dataset = tr_dataset.filter(lambda x: x["Mobile_Tech_Flag"] == 1)
            
            noisy_datasets = []
            for _ in range(n_augment):
                noisy_datasets.append(noisy_dataset.map(lambda x: {"CleanedText": get_noisy_sent(x["CleanedText"].split())}))
            noisy_dataset = concatenate_datasets(noisy_datasets)
            noisy_dataset = noisy_dataset.map(lambda x: {"augmentation_status": "Augmented"})

            tr_dataset = concatenate_datasets([noisy_dataset, tr_dataset])

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

        article = [f["CleanedText"] for f in features]
        summary = [f["CleanedHeadline"] for f in features]

        # src_lang will be dummy
        features = self.tokenizer.prepare_seq2seq_batch(
            src_texts=article, src_lang="hi_IN", tgt_lang="en_XX", tgt_texts=summary, truncation=True, 
            max_length=self.max_length, max_target_length=self.max_target_length, return_tensors="pt"
        )

        return features


if __name__ == '__main__':

    class args:
        batch_size: int = 2
        process_on_fly: bool = False
        num_workers: int = 2
        max_length: int = 512
        max_target_length: int = 20
        file_path: str = "data/dev_data_article.csv"
        seed: int = 42
        n_augment: int = 1

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    dl = DataLoader(tokenizer, args)

    tr_dataset, val_dataset = dl.setup(process_on_fly=args.process_on_fly, n_augment=args.n_augment)

    print(val_dataset)
    print(tr_dataset)

    dataset = concatenate_datasets([tr_dataset, val_dataset]).sort('Text_ID')
    print(dataset)

    print(tr_dataset[:2])

    tr_dataset = dl.train_dataloader(tr_dataset)
    val_dataset = dl.val_dataloader(val_dataset)

    for s in val_dataset:
        pass

    print(s)