from datasets import load_metric, load_dataset
import re

if __name__ == '__main__':

    data = load_dataset("csv", data_files="predictions.csv")["train"]
    data = data.map(lambda x: {"predicted_summary": re.sub("</sep>", "\n", x["predicted_summary"])})
    metrics = {
        "sacrebleu": load_metric("sacrebleu"),
    }

    scores = {}
    for name, metric in metrics.items():
        for example in data:
            metric.add(prediction=example["CleanedHeadline"], reference=[example["predicted_summary"]])
        scores.update({
            name: metric.compute()
        })

    print(scores)