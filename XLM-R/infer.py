import torch
import argparse
from transformers import XLMRobertaTokenizer, AutoConfig, XLMRobertaForSequenceClassification


#either take input from uer or predefined
def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #text = ['New top 100 sports companies ', 'New Top 100 gadgets of the year 2021', 'यहां Apple के 5G स्मार्टफ़ोन की कीमत कितनी हो सकती है ','Infinity wars is the most high tech Avengers make']
    inputs = tokenizer(text, truncation= True, return_tensors= 'pt', padding=True)
    inputs = inputs.to(device)
    model = model.to(device)
    labels = torch.tensor([1]*len(text)).unsqueeze(0) 

    outputs = model(**inputs, labels=labels)
    layer = torch.nn.Softmax(dim=1)
    return layer(outputs.logits.tolist())

if __name__=="__main__":

    tokenizer = XLMRobertaTokenizer.from_pretrained('tanay/xlm-fine-tuned')
    config = AutoConfig.from_pretrained("tanay/xlm-fine-tuned")
    model = XLMRobertaForSequenceClassification.from_pretrained("tanay/xlm-fine-tuned", config = config)

    parser = argparse.ArgumentParser()# use --help for support
    parser.add_argument('--text',default='',required = True, type=str,help='article/tweet')
    args = parser.parse_args()

    #call fucntion
    print(predict(args.text))
    
