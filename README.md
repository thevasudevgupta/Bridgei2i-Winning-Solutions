# Bridgei2i solution

We provide well-commented code for your reference. The overall directory structure is as follows:

**Outputs for the evaluation data is located at ./predictions/**

**High level directory overview**

```
├── text-cls
│   ├── phoneme.py # code conversion to phonemes
│   └── train_cls.py # training classification model
├── summarization
│   ├── train.py # fine mbart model on dataset (refer to training_utils/args.py)
│   ├── evaluate.py
│   └── summarize_using_baseline.py
├── ner
|   └── run.py # run NER + sentiment
├── ppt.pdf # brief solution description
├── preprocess.py # preprocessing script
└── README.md
```

**Running the app for theme classification and summarization**

```bash
streamlit run app.py
```

**Running the app for NER (only for tweets because of dependency issues)**

```bash
streamlit run app_ner.py
```

**IMPORTANT**
- To run the NER code (./ner/run.py) and corresponding app (./app_ner.py), please install transformer==2.5
- To run summarization (./summarization/train.py) and corresponding app (./app.py) please use transformers==4.4