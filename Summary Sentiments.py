# Mohammed Ahsan 
# Carter Mondy 

# Code for An Overview of Sentiments Expressed Across Abstractive and Extractive Summaries

import random,json,torch
from datasets import load_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# random.seed(42) ensures we get the same random documents. 
dataset = load_dataset("cnn_dailymail", "3.0.0")
total_articles=dataset['train']
articles=random.sample(list(total_articles), 100)

model=torch.hub()





summaries = []

for article in articles:
    text = article['article']  # Extract the article text
    summary = model(text, min_length=60, max_length=300)
    summaries.append({'article': text, 'summary': summary})
    
with open("summaries.json", "w") as file:
    json.dump(summaries, file, indent=4)

print('Summaries are pushed to json')

