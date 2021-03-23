from g2p_en import G2p
import re
import pandas as pd
import numpy as np
import pickle

g2p = G2p()

converter = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AX": "ə",
    "AXR": "ɚ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IX": "ɨ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "UX": "ʉ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "DX": "ɾ",
    "EL": "l̩",
    "EM": "m̩",
    "EN": "n̩",
    "F": "f",
    "G": "ɡ",
    "H": "h",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NX": "ɾ̃",
    "NG": "ŋ",
    "P": "p",
    "Q": "ʔ",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "WH": "ʍ",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
    ' ': "SIL",
}
mapping = {
    "अ": "ə",
    "आ": "a:",
    "इ": "ɪ",
    "ई": "i:",
    "उ": "ʊ",
    "ऊ": "u:",
    "ऋ": "ɻ̩",
    "ए": "e:",
    "ऐ": "ɛ:",
    "ओ": "o:",
    "औ": "ɔ:",
    "अँ": "/ə̃/",
    "अः": "/əɦə/",
    "अं": "əm",
    "ऑ": "/ɒ/",
    "ा": "a:",
    "ि": "ɪ",
    "ी": "i:",
    "ु": "ʊ",
    "ू": "u:",
    "ृ": "ɻ̩",
    "े": "e:",
    "ै": "ɛ:",
    "ो": "o:",
    "ौ": "ɔ:",
    "ँ": "/ə̃/",
    "ː": "/əɦə/",
    'ं': "əm",
    "ॆ": "e:",
    "क"+"्": "k",
    "ख"+"्": "kʰ",
    "ग"+"्": "g",
    "घ"+"्": "gʰ",
    "ङ"+"्": "ŋ",
    "च"+"्": "tʃ",
    "छ"+"्": "tʃʰ",
    "ज"+"्": "dʒ",
    "झ"+"्": "dʒʰ",
    "ञ"+"्": "ɲ",
    "ट"+"्": "ʈ",
    "ठ"+"्": "ʈʰ",
    "ड"+"्": "ɖ",
    "ढ"+"्": "ɖʰ",
    "ण"+"्": "ɳ",
    "त"+"्": "t̪",
    "थ"+"्": "t̪ʰ",
    "द"+"्": "d̪",
    "ध"+"्": "d̪ʰ",
    "न"+"्": "n",
    "प"+"्": "p",
    "फ"+"्": "pʰ",
    "ब"+"्": "b",
    "भ"+"्": "bʰ",
    "म"+"्": "m",
    "य"+"्": "j",
    "र"+"्": "ɾ",
    "ल"+"्": "l",
    "व"+"्": "ʋ",
    "स"+"्": "s",
    "श"+"्": "ʃ",
    "ष"+"्": "ʂ",
    "ह"+"्": "ɦ",
    "ळ"+"्": "ɭ̆ɭ̆",
    " ": "SIL",
}
converter.update(mapping)

vowels = ["ा","ि","ी","ु","ू","ृ","े","ै","ो","ौ","ँ",'ं','ॆ',"ː","आ","इ","ई","उ","ऊ","ऋ","ए","ऐ","ओ","औ","अः","अं"]
full_consonants = ["क","ख","ग","घ","ङ","च","छ","ज","झ","ञ","ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","ळ","व","श","ष","स","ह"]
special_vowels = ["ः","ं","ँ"]

def text_phonemes(text="बचाकसट इनाम पढ़ती हैं"):
    # split and add seperator between words - will be silenced later
    tokens = [c for c in text] + [' ']

    # "अ" + <special_vowels>
    for i in range(len(tokens)):
        if tokens[i] == "अ" and tokens[i+1] in special_vowels:
            tokens[i] = "अ" + tokens[i]

    # half mathra 
    temp = []
    cntr = 0
    for i in range(len(tokens)):
        if tokens[i] == "्":
            temp[cntr-1] = temp[cntr-1] + tokens[i]
        else:
            temp.append(tokens[i])
            cntr += 1
    tokens = temp

    # half mathra
    temp = []
    cntr = 0
    for i in range(len(tokens)):
        if tokens[i] in full_consonants and tokens[i+1] not in vowels:
            temp.append(tokens[i] + "्")
            temp.insert(cntr+1, "अ")
            cntr += 2
            if tokens[i+1] == " ":
                temp.append(tokens[i+1])
                cntr += 1
        else:
            temp.append(tokens[i])
            cntr += 1
    tokens = temp

    # half mathra
    for i in range(1, len(tokens)):
        if tokens[i] in vowels and tokens[i-1] not in vowels:
            tokens[i-1] = temp[i-1] + "्"

    # use mapping
    temp = []
    for i in range(len(tokens)):
        if tokens[i] in mapping.keys():
            temp.append(mapping[tokens[i]])
        elif tokens[i] == " ":
            temp.append("SIL")
    tokens = temp
    
    temp = []
    for i in range(len(tokens)):
        if tokens[i] == "SIL" and tokens[i-1] == "SIL":
            continue
        else:
            temp.append(tokens[i])
    tokens = temp
    return tokens

def build_dset(csv_file):
    english_pattern = re.compile("[A-Za-z]+")
    dset = pd.read_csv(csv_file)
    processed_data = []
    for i in range(len(dset)):
        print(i, end="\r")
        sent = dset["cleaned"][i].lower()
        sent = sent.replace(".", "").replace(",", "")
        sent = re.sub(r'[0-9]', '', sent)
        words = sent.split()
        temp = []
        for word in words:
            check = english_pattern.fullmatch(word)
            if check is not None:
                out = g2p(word)
                out = [re.sub(r"[0-9]", "", o) for o in out]
                out = [converter[o] for o in out if o not in [",", ".", ""]]
            else:
                out = text_phonemes(word)
            temp.extend(out)
        processed_data.append(temp)
    all_ele = []
    for d in processed_data:
        all_ele.extend(d)
    vocab = list(set(all_ele))
    print(f"\nprocessed {csv_file}. Vocab size: {len(vocab)}")
    dset["phonemes"] = processed_data
    dset.to_csv(csv_file, index=None)
    word_map = {word: i for i, word in enumerate(vocab)}
    with open('w2ind.pickle', 'wb') as handle:
        pickle.dump(word_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

def conv_phoneme(text):
    english_pattern = re.compile("[A-Za-z]+")
    sent = text.lower()
    sent = sent.replace(".", "").replace(",", "")
    sent = re.sub(r'[0-9]', '', sent)
    words = sent.split()
    temp = []
    for word in words:
        check = english_pattern.fullmatch(word)
        if check is not None:
            out = g2p(word)
            out = [re.sub(r"[0-9]", "", o) for o in out]
            out = [converter[o] for o in out if o not in [",", ".", ""]]
        else:
            out = text_phonemes(word)
        temp.extend(out)
    return temp

if __name__ == "__main__":
    build_dset("../clean_article.csv")