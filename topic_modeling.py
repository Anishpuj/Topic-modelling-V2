import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import json
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.decomposition import PCA
import glob
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

def preprocess_text(text, stop_words, lemmatizer):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def get_sentiment_scores(text, sia):
    scores = sia.polarity_scores(text)
    return {'compound': scores['compound'], 'positive': scores['pos'], 'negative': scores['neg'], 'neutral': scores['neu']}

def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def get_topic_words(topic_idx, cluster_labels, data, n_words=10):
    cluster_indices = np.where(cluster_labels == topic_idx)[0]
    top_words = []
    for idx in cluster_indices:
        words = data['Processed_Reviews'].iloc[idx].split()
        top_words.extend(words)
    word_counts = Counter(top_words)
    return [word for word, _ in word_counts.most_common(n_words)]

def run_analysis_for_file(csv_file, output_dir):
    product_name = os.path.basename(csv_file).replace('Reviews_', '').replace('.csv', '')
    print(f"\n--- Processing {product_name} reviews ---")

    # Create a specific directory for this product's output
    product_output_dir = os.path.join(output_dir, product_name)
    os.makedirs(product_output_dir, exist_ok=True)

    # Load dataset
    try:
        data = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        data = pd.read_csv(csv_file, encoding='latin1')

    # Find the reviews column and other metadata columns
    review_col = next((col for col in data.columns if 'review' in col.lower()), data.columns[0])
    asin_col = next((col for col in data.columns if 'asin' in col.lower()), None)
    product_name_col = next((col for col in data.columns if 'product name' in col.lower()), None)
    
    # Extract ASIN and Product Name (most frequent value)
    product_asin = data[asin_col].mode()[0] if asin_col and not data[asin_col].empty else 'N/A'
    product_display_name = data[product_name_col].mode()[0] if product_name_col and not data[product_name_col].empty else product_name

    print(f"Using column '{review_col}' for reviews.")
    data.rename(columns={review_col: 'Reviews'}, inplace=True)

    # Initialize tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    sia = SentimentIntensityAnalyzer()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Preprocess and sentiment analysis
    data['Processed_Reviews'] = data['Reviews'].apply(lambda x: preprocess_text(x, stop_words, lemmatizer))
    data['Sentiment_Scores'] = data['Reviews'].apply(lambda x: get_sentiment_scores(x, sia))
    data['Compound_Score'] = data['Sentiment_Scores'].apply(lambda x: x['compound'])

    # BERT embeddings
    embeddings = [get_bert_embeddings(review, tokenizer, model) for review in data['Processed_Reviews']]
    X_bert = np.vstack(embeddings)

    # K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_bert)

    # Topic and sentiment summaries
    topic_summaries = []
    topic_sentiments = []
    for topic_idx in range(5):
        top_words = get_topic_words(topic_idx, cluster_labels, data)
        num_reviews = sum(cluster_labels == topic_idx)
        cluster_reviews = data[cluster_labels == topic_idx]
        avg_sentiment = {
            'compound': cluster_reviews['Compound_Score'].mean(),
            'positive': cluster_reviews['Sentiment_Scores'].apply(lambda x: x['positive']).mean(),
            'negative': cluster_reviews['Sentiment_Scores'].apply(lambda x: x['negative']).mean(),
            'neutral': cluster_reviews['Sentiment_Scores'].apply(lambda x: x['neutral']).mean()
        }
        topic_sentiments.append(avg_sentiment)
        topic_summaries.append({
            'topic': f'Topic {topic_idx + 1}',
            'top_words': top_words,
            'description': f'Reviews related to {product_name} - Topic {topic_idx+1}',
            'num_reviews': int(num_reviews),
            'sentiment': avg_sentiment
        })

    # Prepare final output
    output = [{
        'product_name': product_display_name,
        'product_asin': product_asin,
        'topic_summaries': topic_summaries
    }]
    for idx, row in data.iterrows():
        topic_idx = cluster_labels[idx]
        output.append({
            'review': row['Reviews'],
            'sentences': sent_tokenize(str(row['Reviews'])),
            'topic': f'Topic {topic_idx + 1}',
            'top_words': get_topic_words(topic_idx, cluster_labels, data),
            'sentiment': row['Sentiment_Scores']
        })

    # Save JSON output to the product's directory
    json_path = os.path.join(product_output_dir, f'topic_modeling_output_{product_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"Saved output to {json_path}")

    # Generate visualizations
    # Word clouds
    plt.figure(figsize=(15, 10))
    for i in range(5):
        words = get_topic_words(i, cluster_labels, data)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
        plt.subplot(2, 3, i + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {i+1}')
        plt.axis('off')
    plt.tight_layout()
    wc_path = os.path.join(product_output_dir, f'topic_wordclouds_{product_name}.png')
    plt.savefig(wc_path)
    plt.close()
    print(f"Saved word clouds to {wc_path}")

    return product_name

if __name__ == '__main__':
    # Define the main output directory for the dashboard's data
    dashboard_data_dir = 'public/data'
    os.makedirs(dashboard_data_dir, exist_ok=True)

    csv_files = glob.glob('Reviews_*.csv')
    processed_products = []

    for file in csv_files:
        product = run_analysis_for_file(file, dashboard_data_dir)
        processed_products.append(product)

    # Create a manifest file for the dashboard
    manifest = {'products': sorted(processed_products)}
    with open('public/products.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)

    print("\n--- All files processed successfully! ---")
    print("Created products.json manifest for the dashboard.") 