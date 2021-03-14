from datasets import load_metric, load_dataset
import re

REFERENCE_COLUMN_NAME = "Headline" # "CleanedHeadline"
PREDICTION_COLUMN_NAME = "gensim_summary" # prediction_summary
GENSIM = True # False

if __name__ == '__main__':

    data = load_dataset("csv", data_files="results/predictions.csv")["train"]
    data = data.map(lambda x: {"predicted_summary": re.sub("</sep>", "\n", x["predicted_summary"])})

    if GENSIM:
        data = load_dataset("csv", data_files="gensim_summary.csv")["train"]
        data = data.filter(lambda x: x["gensim_summary"] is not None)

    metrics = {
        "sacrebleu": load_metric("sacrebleu"),
        "rouge": load_metric("rouge"),
        "wer": load_metric("wer"),
    }

    data = data.filter(lambda x: x["split"] == "VALIDATION")
    print(data)

    scores = {}
    for name, metric in metrics.items():
        if name in ["sacrebleu"]:
            for example in data:
                metric.add(prediction=example[PREDICTION_COLUMN_NAME], reference=[example[REFERENCE_COLUMN_NAME]])
        elif name in ["rouge", "wer"]:
            for example in data:
                metric.add(prediction=example[PREDICTION_COLUMN_NAME], reference=example[REFERENCE_COLUMN_NAME])
        scores.update({
            name: metric.compute()
        })
        print(name, ':', scores[name], end="\n\n")
