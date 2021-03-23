from datasets import load_metric, load_dataset
import re
from tqdm import tqdm
import scipy
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import corpus_bleu

REFERENCE_COLUMN_NAME = "CleanedHeadline" # "CleanedHeadline" "Headline"
PREDICTION_COLUMN_NAME = "predicted_summary" # "predicted_summary" "gensim_summary"
GENSIM = False # True False

if __name__ == '__main__':

    headline_similarity_model = None # SentenceTransformer('bert-base-nli-mean-tokens')
    calculate_bleu = True

    data = load_dataset("csv", data_files="results/predictions-augment-exp2.csv")["train"]
    data = data.map(lambda x: {"predicted_summary": re.sub("</sep>", "\n", x["predicted_summary"])})

    if GENSIM:
        data = load_dataset("csv", data_files="gensim_summary.csv")["train"]
        data = data.filter(lambda x: x["gensim_summary"] is not None)

    metrics = {
        # "sacrebleu": load_metric("sacrebleu"),
        # "rouge": load_metric("rouge"),
        # "wer": load_metric("wer"),
    }

    data = data.filter(lambda x: x["split"] == "VALIDATION")
    print(data)

    scores = {}
    for name, metric in metrics.items():
        if name in ["sacrebleu"]:
            for example in tqdm(data):
                metric.add(prediction=example[PREDICTION_COLUMN_NAME], reference=[example[REFERENCE_COLUMN_NAME]])
        elif name in ["rouge", "wer"]:
            for example in tqdm(data):
                metric.add(prediction=example[PREDICTION_COLUMN_NAME], reference=example[REFERENCE_COLUMN_NAME])
        scores.update({
            name: metric.compute()
        })
        print(name, ':', scores[name], end="\n\n")

    if headline_similarity_model is not None:
        headline_similarity = []
        for example in tqdm(data):
            actual_headline_embeddings = headline_similarity_model.encode(example[REFERENCE_COLUMN_NAME]) # Get a vector for each headlines
            predicted_headline_embeddings = headline_similarity_model.encode(example[PREDICTION_COLUMN_NAME]) # Get a vector for each headlines
            distance = scipy.spatial.distance.cdist([actual_headline_embeddings], [predicted_headline_embeddings], "cosine")[0]
            headline_similarity.append("%.4f" % (1-distance))

        headline_similarity = list(map(float, headline_similarity))
        avg_headline_sim_score = round(sum(headline_similarity)/len(headline_similarity),2)

        print("Headline similarity", avg_headline_sim_score)

    if calculate_bleu:
        bleu_scores = []
        for example in tqdm(data):
            list_of_references = [[example[REFERENCE_COLUMN_NAME].split()]]
            list_of_hypotheses = [example[PREDICTION_COLUMN_NAME].split()]
            bleu_score = corpus_bleu(list_of_references, list_of_hypotheses)
            bleu_scores.append(bleu_score)

        avg_bleu_scores = round(sum(bleu_scores)/len(bleu_scores),2)
        print("BLEU", ':', avg_bleu_scores)
