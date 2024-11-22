import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


total_summaries = pd.read_csv('summarization_model_results.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
model.to(device)

def analyze_sentiment(summary):
    # Tokenize and move input to the appropriate device
    inputs = tokenizer(summary, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Model output
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Sentiment classification
    sentiment = torch.argmax(probs, dim=-1).item()
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    
    # Sentiment scores (probabilities)
    scores = probs.tolist()[0]
    
    return {
        "label": sentiment_labels[sentiment],
        "scores": {
            "Negative": scores[0],
            "Neutral": scores[1],
            "Positive": scores[2]
        }
    }

results = []
for article in total_summaries.itertuples():
    extractive_summary = total_summaries.at[article[0],'extractive_summary']
    abstractive_summary = total_summaries.at[article[0],'abstractive_summary']
    human_summary = total_summaries.at[article[0],'human_summary']
    if extractive_summary:
        result = analyze_sentiment(extractive_summary)
        results.append({
            "extractive_summary": extractive_summary,
            "extractive_sentiment": result
        })

    if abstractive_summary:
        result = analyze_sentiment(abstractive_summary)
        results.append({
            "abstractive_summary": abstractive_summary,
            "abstractive_sentiment": result
        })

    if human_summary:
        result = analyze_sentiment(human_summary)
        results.append({
            "human_summary": human_summary,
            "human_summary": result
        })

# Save results to a new JSON file
with open('sentiment_results.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)
    
print('Outputs sucessfully written to file')