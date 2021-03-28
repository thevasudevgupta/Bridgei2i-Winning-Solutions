# Bridgei2i solution

**Winning solution of competition held by Bridgei2i under InterIIT Tech Meet 2021.**

| Team Members | [Vasudev](https://github.com/vasudevgupta7), [Mukund](https://github.com/MukundVarmaT), [Jayesh](https://github.com/jayeshkumar1734), [Yadukrishnan](https://github.com/YadukrishnanBlk), [Tanay](https://github.com/tanay2001), [Anirudh](https://github.com/anirudhs123), [Siddhant](https://github.com/tokentaker2339) |
|--------------|---------------------------------------------------------------------------------------------|

## Contents

We provide well-commented code for your reference. The overall directory structure is as follows:

**Outputs for the evaluation data is located at ./predictions/**

**High level directory overview**

```
├── text-cls
│   ├── phoneme.py # code conversion to phonemes
│   └── train_cls.py # training classification model
├── summarization
│   ├── train.py # fine mbart model on dataset (refer to training_utils/args.py)
│   └── evaluate.py
├── ner
|   └── run.py # run NER + sentiment
├── assets
|   └── ppt.pdf # brief solution description
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