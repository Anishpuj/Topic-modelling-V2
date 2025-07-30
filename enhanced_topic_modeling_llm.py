import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import json
import requests
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import glob
import os
from datetime import datetime
import time

# For local LLM integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not installed. Install with: pip install ollama")

# For Hugging Face transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class LocalLLMTopicModeling:
    def __init__(self, model_type="ollama", llm_model="llama3.2:3b"):
        """
        Initialize the enhanced topic modeling system with local LLM support
        
        Args:
            model_type: "ollama", "huggingface", or "sentence_transformer"
            llm_model: Model name to use
        """
        self.model_type = model_type
        self.llm_model = llm_model
        
        # Initialize traditional NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize embeddings model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize local LLM
        self.setup_local_llm()
        
    def setup_local_llm(self):
        """Setup the local LLM based on the chosen type"""
        if self.model_type == "ollama" and OLLAMA_AVAILABLE:
            try:
                # Check if model is available, if not pull it
                models = ollama.list()
                model_names = [model['name'] for model in models['models']]
                
                if self.llm_model not in model_names:
                    print(f"Pulling {self.llm_model} model...")
                    ollama.pull(self.llm_model)
                
                self.llm_client = ollama
                print(f"Ollama model {self.llm_model} ready!")
                
            except Exception as e:
                print(f"Error setting up Ollama: {e}")
                self.model_type = "sentence_transformer"
                
        elif self.model_type == "huggingface":
            try:
                print("Loading Hugging Face model...")
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.hf_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                print("Hugging Face model ready!")
            except Exception as e:
                print(f"Error loading Hugging Face model: {e}")
                self.model_type = "sentence_transformer"
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def get_embeddings(self, texts):
        """Get embeddings using Sentence Transformers"""
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def perform_clustering(self, embeddings, n_clusters=5):
        """Perform clustering on embeddings"""
        print(f"Performing clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels, kmeans
    
    def get_topic_words(self, topic_idx, cluster_labels, processed_reviews, n_words=10):
        """Extract top words for a topic"""
        cluster_indices = np.where(cluster_labels == topic_idx)[0]
        top_words = []
        for idx in cluster_indices:
            words = processed_reviews.iloc[idx].split()
            top_words.extend(words)
        word_counts = Counter(top_words)
        return [word for word, _ in word_counts.most_common(n_words)]
    
    def get_sample_reviews(self, topic_idx, cluster_labels, reviews, n_samples=3):
        """Get sample reviews for a topic"""
        cluster_indices = np.where(cluster_labels == topic_idx)[0]
        sample_indices = np.random.choice(cluster_indices, 
                                        min(n_samples, len(cluster_indices)), 
                                        replace=False)
        return [reviews.iloc[idx] for idx in sample_indices]
    
    def generate_topic_description_with_llm(self, top_words, sample_reviews, topic_idx):
        """Generate topic description using local LLM"""
        if self.model_type == "ollama" and OLLAMA_AVAILABLE:
            return self.generate_with_ollama(top_words, sample_reviews, topic_idx)
        else:
            # Fallback to rule-based description
            return self.generate_rule_based_description(top_words, topic_idx)
    
    def generate_with_ollama(self, top_words, sample_reviews, topic_idx):
        """Generate description using Ollama"""
        try:
            prompt = f"""
            Analyze this topic from customer reviews and provide insights:
            
            Topic {topic_idx + 1} Key Words: {', '.join(top_words[:8])}
            
            Sample Reviews:
            {chr(10).join([f"- {review[:200]}..." for review in sample_reviews[:2]])}
            
            Please provide:
            1. A concise topic name (2-4 words)
            2. A brief description of what customers are discussing
            3. The main sentiment theme (positive/negative/mixed)
            
            Format your response as:
            Name: [topic name]
            Description: [description]
            Sentiment: [sentiment theme]
            """
            
            response = self.llm_client.chat(
                model=self.llm_model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'max_tokens': 200
                }
            )
            
            return self.parse_llm_response(response['message']['content'])
            
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return self.generate_rule_based_description(top_words, topic_idx)
    
    def parse_llm_response(self, response_text):
        """Parse LLM response to extract structured information"""
        lines = response_text.strip().split('\n')
        result = {
            'name': f'Topic Analysis',
            'description': 'Customer feedback analysis',
            'sentiment_theme': 'Mixed'
        }
        
        for line in lines:
            if line.startswith('Name:'):
                result['name'] = line.replace('Name:', '').strip()
            elif line.startswith('Description:'):
                result['description'] = line.replace('Description:', '').strip()
            elif line.startswith('Sentiment:'):
                result['sentiment_theme'] = line.replace('Sentiment:', '').strip()
        
        return result
    
    def generate_rule_based_description(self, top_words, topic_idx):
        """Fallback rule-based description generation"""
        # Simple rule-based topic naming
        word_themes = {
            'quality': ['quality', 'good', 'excellent', 'perfect', 'great'],
            'delivery': ['delivery', 'shipping', 'fast', 'quick', 'time'],
            'price': ['price', 'cheap', 'expensive', 'value', 'money', 'cost'],
            'design': ['design', 'look', 'beautiful', 'color', 'style'],
            'service': ['service', 'support', 'help', 'staff', 'customer'],
            'product': ['product', 'item', 'package', 'box', 'received']
        }
        
        theme_scores = {}
        for theme, theme_words in word_themes.items():
            score = sum(1 for word in top_words if word in theme_words)
            theme_scores[theme] = score
        
        main_theme = max(theme_scores, key=theme_scores.get)
        
        return {
            'name': f'{main_theme.title()} Discussion',
            'description': f'Reviews discussing {main_theme} aspects of the product',
            'sentiment_theme': 'Mixed'
        }
    
    def analyze_product_reviews(self, csv_file, output_dir):
        """Analyze reviews for a single product with LLM enhancement"""
        product_name = os.path.basename(csv_file).replace('Reviews_', '').replace('.csv', '')
        print(f"\n🔍 Analyzing {product_name} reviews with Local LLM...")
        
        # Create output directory
        product_output_dir = os.path.join(output_dir, product_name)
        os.makedirs(product_output_dir, exist_ok=True)
        
        # Load data
        try:
            data = pd.read_csv(csv_file)
        except UnicodeDecodeError:
            data = pd.read_csv(csv_file, encoding='latin1')
        
        # Find review column
        review_col = next((col for col in data.columns if 'review' in col.lower()), data.columns[0])
        data.rename(columns={review_col: 'Reviews'}, inplace=True)
        
        # Preprocess
        print("📝 Preprocessing text...")
        data['Processed_Reviews'] = data['Reviews'].apply(self.preprocess_text)
        
        # Sentiment analysis
        print("😊 Analyzing sentiment...")
        data['Sentiment_Scores'] = data['Reviews'].apply(
            lambda x: self.sia.polarity_scores(x)
        )
        data['Compound_Score'] = data['Sentiment_Scores'].apply(lambda x: x['compound'])
        
        # Generate embeddings
        embeddings = self.get_embeddings(data['Processed_Reviews'].tolist())
        
        # Clustering
        cluster_labels, kmeans_model = self.perform_clustering(embeddings, n_clusters=5)
        
        # Enhanced topic analysis with LLM
        print("🤖 Generating topic descriptions with Local LLM...")
        topic_summaries = []
        
        for topic_idx in range(5):
            top_words = self.get_topic_words(topic_idx, cluster_labels, data['Processed_Reviews'])
            sample_reviews = self.get_sample_reviews(topic_idx, cluster_labels, data['Reviews'])
            
            # Get LLM-generated description
            llm_analysis = self.generate_topic_description_with_llm(
                top_words, sample_reviews, topic_idx
            )
            
            # Calculate statistics
            cluster_reviews = data[cluster_labels == topic_idx]
            num_reviews = len(cluster_reviews)
            avg_sentiment = {
                'compound': cluster_reviews['Compound_Score'].mean(),
                'positive': cluster_reviews['Sentiment_Scores'].apply(lambda x: x['pos']).mean(),
                'negative': cluster_reviews['Sentiment_Scores'].apply(lambda x: x['neg']).mean(),
                'neutral': cluster_reviews['Sentiment_Scores'].apply(lambda x: x['neu']).mean()
            }
            
            topic_summary = {
                'topic': f'Topic {topic_idx + 1}',
                'llm_generated_name': llm_analysis['name'],
                'llm_description': llm_analysis['description'],
                'llm_sentiment_theme': llm_analysis['sentiment_theme'],
                'top_words': top_words,
                'num_reviews': int(num_reviews),
                'sentiment': avg_sentiment,
                'sample_reviews': sample_reviews[:2]  # Include samples
            }
            
            topic_summaries.append(topic_summary)
            
            print(f"✅ Topic {topic_idx + 1}: {llm_analysis['name']}")
        
        # Generate enhanced visualizations
        self.create_enhanced_visualizations(
            topic_summaries, cluster_labels, data, product_output_dir, product_name
        )
        
        # Save enhanced results
        output_data = {
            'product_name': product_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'model_info': {
                'llm_type': self.model_type,
                'llm_model': self.llm_model,
                'embedding_model': 'all-MiniLM-L6-v2'
            },
            'topic_summaries': topic_summaries,
            'overall_stats': {
                'total_reviews': len(data),
                'avg_sentiment': data['Compound_Score'].mean(),
                'sentiment_distribution': {
                    'positive': len(data[data['Compound_Score'] >= 0.05]),
                    'negative': len(data[data['Compound_Score'] <= -0.05]),
                    'neutral': len(data[(data['Compound_Score'] > -0.05) & 
                                      (data['Compound_Score'] < 0.05)])
                }
            }
        }
        
        # Include individual review analysis
        reviews_analysis = []
        for idx, row in data.iterrows():
            topic_idx = cluster_labels[idx]
            reviews_analysis.append({
                'review': row['Reviews'],
                'topic': f'Topic {topic_idx + 1}',
                'topic_name': topic_summaries[topic_idx]['llm_generated_name'],
                'sentiment': row['Sentiment_Scores'],
                'processed_text': row['Processed_Reviews']
            })
        
        output_data['reviews'] = reviews_analysis
        
        # Save to JSON
        json_path = os.path.join(product_output_dir, f'enhanced_analysis_{product_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Enhanced analysis saved to {json_path}")
        return product_name
    
    def create_enhanced_visualizations(self, topic_summaries, cluster_labels, data, output_dir, product_name):
        """Create enhanced visualizations with LLM insights"""
        
        # 1. Enhanced Word Clouds with LLM titles
        plt.figure(figsize=(20, 12))
        for i, topic in enumerate(topic_summaries):
            words = ' '.join(topic['top_words'])
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                colormap='viridis').generate(words)
            
            plt.subplot(2, 3, i + 1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f"{topic['llm_generated_name']}\n({topic['num_reviews']} reviews)", 
                     fontsize=12, fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'enhanced_wordclouds_{product_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Topic Distribution with LLM names
        plt.figure(figsize=(12, 8))
        topic_names = [t['llm_generated_name'] for t in topic_summaries]
        topic_counts = [t['num_reviews'] for t in topic_summaries]
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_names)))
        
        plt.pie(topic_counts, labels=topic_names, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title(f'Topic Distribution for {product_name}\n(Enhanced with Local LLM)', 
                 fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(output_dir, f'enhanced_topic_distribution_{product_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sentiment Analysis by Topic
        plt.figure(figsize=(14, 8))
        sentiments = []
        names = []
        for topic in topic_summaries:
            sentiments.append([
                topic['sentiment']['positive'],
                topic['sentiment']['negative'], 
                topic['sentiment']['neutral']
            ])
            names.append(topic['llm_generated_name'])
        
        sentiments = np.array(sentiments)
        x = np.arange(len(names))
        width = 0.25
        
        plt.bar(x - width, sentiments[:, 0], width, label='Positive', color='green', alpha=0.7)
        plt.bar(x, sentiments[:, 1], width, label='Negative', color='red', alpha=0.7)
        plt.bar(x + width, sentiments[:, 2], width, label='Neutral', color='gray', alpha=0.7)
        
        plt.xlabel('Topics (LLM Generated Names)')
        plt.ylabel('Average Sentiment Score')
        plt.title(f'Sentiment Analysis by Topic - {product_name}')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sentiment_by_topic_{product_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Enhanced visualizations created for {product_name}")


def main():
    """Main function to run enhanced topic modeling with local LLM"""
    
    print("🚀 Starting Enhanced Topic Modeling with Local LLM")
    print("=" * 60)
    
    # Initialize the enhanced system
    # You can change model_type to "huggingface" or "sentence_transformer"
    analyzer = LocalLLMTopicModeling(
        model_type="ollama", 
        llm_model="llama3.2:3b"  # You can also try "mistral", "phi3", etc.
    )
    
    # Create output directory
    output_dir = 'enhanced_analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob('Reviews_*.csv')
    
    if not csv_files:
        print("❌ No CSV files found matching 'Reviews_*.csv' pattern")
        return
    
    processed_products = []
    
    for csv_file in csv_files:
        try:
            product = analyzer.analyze_product_reviews(csv_file, output_dir)
            processed_products.append(product)
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            continue
    
    # Create manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'products_analyzed': len(processed_products),
        'products': processed_products,
        'model_info': {
            'type': analyzer.model_type,
            'llm_model': analyzer.llm_model
        }
    }
    
    with open(os.path.join(output_dir, 'analysis_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n🎉 Enhanced Topic Modeling Complete!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Products analyzed: {len(processed_products)}")


if __name__ == "__main__":
    main()
