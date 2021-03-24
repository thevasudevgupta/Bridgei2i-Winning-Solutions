from datasets import load_dataset
import pandas as pd
import numpy as np
import ast

# combine headlines

# sentiment = pd.read_csv("results/predicted_with preprocess.csv")
# # wop = load_dataset("csv", data_files="results/predicted_without preprocess.csv")
# summaries = pd.read_csv("results/prediction_articles.csv")
# print(sentiment.shape, summaries.shape)
# data = sentiment.merge(summaries, on="Text_ID", how="outer").drop(["Unnamed: 0", "Text"], axis=1)
# for i in range(len(data)):
#     if data.iloc[i, 1] == 0:
#         data.iloc[i, 2] = np.nan
#     if type(data.iloc[i, 2]) == str:
#         if data.iloc[i, 2].startswith(r"Home\n"):
#             data.iloc[i, 2] = data.iloc[i, 2][6:].strip()
#             print(data.iloc[i, 2])
# print(data.shape, data.columns)
# print(data)
# data.to_csv("headlines.csv", index=False)


# combine ner

sentiment = pd.read_csv("results/predicted_with preprocess.csv")
ner = pd.read_csv("results/Eval_data_Tweets&Articles_prediction_using_headlines (1).csv")
print(sentiment.shape, ner.shape)

data = sentiment.merge(ner, on="Text_ID", how="outer").drop(["Unnamed: 0", "Clean_text", "CleanedTweet", "Unnamed: 0.1", "Length", 'Text_x', 'Text_y'], axis=1)
print(data.shape)
print(data.columns)

data['Brands_Entity_Identified'] = np.nan
data["Sentiment_Identified"] = np.nan

for i in range(len(data)):
    dictn = ast.literal_eval(data.iloc[i, 2])
    if len(dictn) > 0:
        data.iloc[i, 3] = str(list(dictn.keys()))
        data.iloc[i, 4] = str(list(dictn.values()))

    if data.iloc[i, 1] == 0:
        data.iloc[i, 3] = np.nan
        data.iloc[i, 4] = np.nan

data.drop("Entity_senti_dict", axis=1, inplace=True)
print(data.shape, data.columns)
print(data)
data.to_csv("ner.csv", index=False)
