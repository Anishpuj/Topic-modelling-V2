import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import glob
import os
from datetime import datetime

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class IntelligentTopicModeling:
    def __init__(self):
        """Initialize the intelligent topic modeling system"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        
        # Load sentence transformer for semantic embeddings
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define product-specific topic themes
        self.product_topic_themes = {
            'desk': {
                'build_quality': ['build', 'quality', 'sturdy', 'solid', 'strong', 'durable', 'construction', 'material', 'frame'],
                'functionality': ['height', 'adjustment', 'adjustable', 'motor', 'control', 'memory', 'preset', 'smooth', 'function'],
                'design_aesthetics': ['design', 'look', 'appearance', 'style', 'classy', 'elegant', 'beautiful', 'aesthetic', 'color'],
                'usability': ['easy', 'setup', 'assembly', 'installation', 'use', 'convenient', 'user', 'friendly', 'simple'],
                'value_service': ['price', 'value', 'money', 'service', 'support', 'delivery', 'customer', 'worth', 'competitive']
            },
            'phone': {
                'performance': ['performance', 'speed', 'fast', 'chip', 'processor', 'smooth', 'lag', 'gaming', 'app'],
                'camera': ['camera', 'photo', 'picture', 'video', 'quality', 'lens', 'zoom', 'capture', 'selfie'],
                'battery_display': ['battery', 'life', 'charge', 'display', 'screen', 'brightness', 'resolution', 'color'],
                'design_build': ['design', 'build', 'look', 'feel', 'premium', 'weight', 'size', 'color', 'finish'],
                'value_experience': ['price', 'value', 'money', 'worth', 'experience', 'overall', 'recommend', 'satisfied']
            },
            'clothing': {
                'fit_comfort': ['fit', 'size', 'comfortable', 'comfort', 'tight', 'loose', 'perfect', 'sizing', 'wear'],
                'quality_material': ['quality', 'material', 'fabric', 'cloth', 'texture', 'soft', 'durable', 'thick', 'thin'],
                'design_style': ['design', 'style', 'look', 'color', 'pattern', 'fashion', 'trendy', 'stylish', 'appearance'],
                'value_pricing': ['price', 'value', 'money', 'cheap', 'expensive', 'worth', 'affordable', 'cost', 'budget'],
                'overall_satisfaction': ['good', 'great', 'excellent', 'satisfied', 'happy', 'recommend', 'love', 'like', 'amazing']
            },
            'footwear': {
                'comfort_fit': ['comfort', 'comfortable', 'fit', 'size', 'cushion', 'support', 'arch', 'heel', 'toe'],
                'quality_durability': ['quality', 'durable', 'strong', 'material', 'leather', 'sole', 'construction', 'build'],
                'design_appearance': ['design', 'look', 'style', 'color', 'appearance', 'beautiful', 'elegant', 'fashionable'],
                'performance_usage': ['walking', 'running', 'sport', 'exercise', 'performance', 'grip', 'traction', 'stability'],
                'value_satisfaction': ['price', 'value', 'money', 'worth', 'satisfied', 'recommend', 'happy', 'excellent']
            },
            'watch': {
                'features_functionality': ['feature', 'function', 'smart', 'fitness', 'health', 'notification', 'app', 'connectivity'],
                'battery_performance': ['battery', 'life', 'charge', 'performance', 'speed', 'responsive', 'smooth', 'fast'],
                'design_build': ['design', 'build', 'look', 'premium', 'quality', 'material', 'strap', 'band', 'display'],
                'usability': ['easy', 'use', 'setup', 'interface', 'user', 'friendly', 'convenient', 'simple', 'intuitive'],
                'value_overall': ['price', 'value', 'money', 'worth', 'overall', 'satisfied', 'recommend', 'excellent', 'great']
            },
            'tv': {
                'picture_quality': ['picture', 'image', 'color', 'quality', 'display', 'screen', 'clarity', 'brightness', 'resolution'],
                'sound_audio': ['sound', 'audio', 'volume', 'speaker', 'music', 'voice', 'clear', 'loud', 'bass'],
                'features_smart': ['smart', 'feature', 'app', 'streaming', 'netflix', 'youtube', 'remote', 'control', 'function'],
                'installation_setup': ['installation', 'setup', 'delivery', 'install', 'mount', 'wall', 'service', 'technician'],
                'value_satisfaction': ['price', 'value', 'money', 'worth', 'satisfied', 'recommend', 'excellent', 'good', 'quality']
            }
        }
        
        # Generic themes for unknown products
        self.generic_themes = {
            'quality_build': ['quality', 'build', 'construction', 'material', 'durable', 'strong', 'solid', 'sturdy'],
            'performance': ['performance', 'speed', 'fast', 'smooth', 'efficient', 'reliable', 'responsive'],
            'design_aesthetics': ['design', 'look', 'appearance', 'style', 'beautiful', 'elegant', 'attractive'],
            'usability': ['easy', 'use', 'convenient', 'user', 'friendly', 'simple', 'intuitive', 'comfortable'],
            'value_satisfaction': ['price', 'value', 'money', 'worth', 'satisfied', 'recommend', 'excellent', 'great']
        }
    
    def identify_product_category(self, product_name):
        """Identify product category from name"""
        product_name = product_name.lower()
        
        if any(word in product_name for word in ['desk', 'table', 'stand', 'workstation']):
            return 'desk'
        elif any(word in product_name for word in ['phone', 'iphone', 'smartphone', 'mobile']):
            return 'phone'
        elif any(word in product_name for word in ['jean', 'pant', 'trouser', 'shirt', 'clothing']):
            return 'clothing'
        elif any(word in product_name for word in ['shoe', 'sneaker', 'boot', 'footwear']):
            return 'footwear'
        elif any(word in product_name for word in ['watch', 'smartwatch', 'timepiece']):
            return 'watch'
        elif any(word in product_name for word in ['tv', 'television', 'samsung', 'final']):
            return 'tv'
        else:
            return 'generic'
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def extract_theme_scores(self, reviews, themes):
        """Calculate theme scores based on word frequency"""
        theme_scores = defaultdict(float)
        all_text = ' '.join(reviews)
        
        for theme_name, theme_words in themes.items():
            score = 0
            for word in theme_words:
                score += all_text.count(' ' + word + ' ') + all_text.count(' ' + word)
            
            # Normalize by theme word count and text length
            normalized_score = score / (len(theme_words) * len(all_text.split()) / 1000)
            theme_scores[theme_name] = normalized_score
        
        return theme_scores
    
    def generate_intelligent_topic_names(self, cluster_labels, processed_reviews, original_reviews, product_category):
        """Generate intelligent topic names based on content analysis"""
        n_topics = len(set(cluster_labels))
        topic_info = []
        
        # Get appropriate themes for the product
        if product_category in self.product_topic_themes:
            themes = self.product_topic_themes[product_category]
        else:
            themes = self.generic_themes
        
        for topic_idx in range(n_topics):
            # Get reviews for this topic
            cluster_indices = np.where(cluster_labels == topic_idx)[0]
            topic_reviews = [processed_reviews.iloc[idx] for idx in cluster_indices]
            topic_original = [original_reviews.iloc[idx] for idx in cluster_indices]
            
            # Calculate theme scores for this topic
            theme_scores = self.extract_theme_scores(topic_reviews, themes)
            
            # Get top words using TF-IDF
            if len(topic_reviews) > 0:
                vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
                try:
                    tfidf_matrix = vectorizer.fit_transform(topic_reviews)
                    feature_names = vectorizer.get_feature_names_out()
                    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    top_indices = mean_scores.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_indices]
                except:
                    # Fallback to simple word counting
                    all_words = ' '.join(topic_reviews).split()
                    word_counts = Counter(all_words)
                    top_words = [word for word, _ in word_counts.most_common(10)]
            else:
                top_words = []
            
            # Determine the best theme for this topic
            if theme_scores:
                primary_theme = max(theme_scores, key=theme_scores.get)
                theme_confidence = theme_scores[primary_theme]
            else:
                primary_theme = list(themes.keys())[0]
                theme_confidence = 0.0
            
            # Generate topic name and description
            topic_name, topic_description = self.create_topic_name_description(
                primary_theme, top_words, topic_original[:3], product_category
            )
            
            # Calculate sentiment
            sentiments = []
            for review in topic_original:
                if review and len(str(review).strip()) > 0:
                    sent_scores = self.sia.polarity_scores(str(review))
                    sentiments.append(sent_scores)
            
            if sentiments:
                avg_sentiment = {
                    'compound': np.mean([s['compound'] for s in sentiments]),
                    'positive': np.mean([s['pos'] for s in sentiments]),
                    'negative': np.mean([s['neg'] for s in sentiments]),
                    'neutral': np.mean([s['neu'] for s in sentiments])
                }
            else:
                avg_sentiment = {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
            
            topic_info.append({
                'topic': topic_name,
                'description': topic_description,
                'primary_theme': primary_theme,
                'theme_confidence': theme_confidence,
                'top_words': top_words,
                'num_reviews': len(cluster_indices),
                'sentiment': avg_sentiment,
                'sample_reviews': topic_original[:2]
            })
        
        return topic_info
    
    def create_topic_name_description(self, theme, top_words, sample_reviews, product_category):
        """Create meaningful topic names and descriptions"""
        
        # Theme-based naming
        theme_names = {
            'build_quality': 'Build Quality & Construction',
            'functionality': 'Features & Functionality', 
            'design_aesthetics': 'Design & Aesthetics',
            'usability': 'Ease of Use & Setup',
            'value_service': 'Value & Customer Service',
            'performance': 'Performance & Speed',
            'camera': 'Camera & Photography',
            'battery_display': 'Battery & Display',
            'design_build': 'Design & Build Quality',
            'value_experience': 'Value & User Experience',
            'fit_comfort': 'Fit & Comfort',
            'quality_material': 'Quality & Material',
            'design_style': 'Design & Style',
            'value_pricing': 'Pricing & Value',
            'overall_satisfaction': 'Overall Satisfaction',
            'comfort_fit': 'Comfort & Fit',
            'quality_durability': 'Quality & Durability',
            'design_appearance': 'Design & Appearance',
            'performance_usage': 'Performance & Usage',
            'value_satisfaction': 'Value & Satisfaction',
            'features_functionality': 'Features & Smart Functions',
            'battery_performance': 'Battery & Performance',
            'quality_build': 'Quality & Build',
            'picture_quality': 'Picture Quality & Display',
            'sound_audio': 'Sound & Audio',
            'features_smart': 'Smart Features & Apps',
            'installation_setup': 'Installation & Setup',
            'value_satisfaction': 'Value & Satisfaction',
        }
        
        topic_name = theme_names.get(theme, theme.replace('_', ' ').title())
        
        # Generate description based on top words and theme
        if 'quality' in theme or 'build' in theme:
            description = f"Reviews discussing product build quality, materials, and construction durability"
        elif 'performance' in theme:
            description = f"Reviews about product performance, speed, and operational efficiency"
        elif 'design' in theme or 'aesthetic' in theme or 'appearance' in theme:
            description = f"Reviews focusing on product design, appearance, and visual appeal"
        elif 'usability' in theme or 'use' in theme:
            description = f"Reviews about ease of use, setup, and user-friendliness"
        elif 'value' in theme or 'price' in theme:
            description = f"Reviews discussing pricing, value for money, and purchase satisfaction"
        elif 'comfort' in theme or 'fit' in theme:
            description = f"Reviews about comfort, fit, and ergonomic aspects"
        elif 'functionality' in theme or 'feature' in theme:
            description = f"Reviews discussing product features and functional capabilities"
        elif 'battery' in theme:
            description = f"Reviews about battery life, charging, and power performance"
        elif 'camera' in theme:
            description = f"Reviews focusing on camera quality and photography features"
        else:
            description = f"Reviews about {theme.replace('_', ' ')} aspects of the product"
        
        return topic_name, description
    
    def analyze_product_reviews(self, csv_file, output_dir):
        """Analyze reviews for a single product with intelligent topic naming"""
        product_name = os.path.basename(csv_file).replace('Reviews_', '').replace('.csv', '')
        print(f"\n🔍 Analyzing {product_name} reviews with intelligent topic modeling...")
        
        # Create output directory
        product_output_dir = os.path.join(output_dir, product_name)
        os.makedirs(product_output_dir, exist_ok=True)
        
        # Load data
        try:
            data = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(csv_file, encoding='latin1')
            except:
                data = pd.read_csv(csv_file, encoding='cp1252')
        
        # Find review column
        review_col = None
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['review', 'feedback', 'comment', 'text']):
                review_col = col
                break
        
        if review_col is None:
            # Take the last column or a column with long text
            text_lengths = [data[col].astype(str).str.len().mean() for col in data.columns]
            review_col = data.columns[np.argmax(text_lengths)]
        
        print(f"Using column '{review_col}' for reviews.")
        data.rename(columns={review_col: 'Reviews'}, inplace=True)
        
        # Remove empty reviews
        data = data.dropna(subset=['Reviews'])
        data = data[data['Reviews'].astype(str).str.strip() != '']
        
        if len(data) < 5:
            print(f"⚠️  Not enough reviews ({len(data)}) for meaningful analysis")
            return None
        
        # Identify product category
        product_category = self.identify_product_category(product_name)
        if 'Product Name' in data.columns:
            product_category = self.identify_product_category(data['Product Name'].iloc[0])
        
        print(f"📂 Detected product category: {product_category}")
        
        # Preprocess
        print("📝 Preprocessing text...")
        data['Processed_Reviews'] = data['Reviews'].apply(self.preprocess_text)
        
        # Remove empty processed reviews
        data = data[data['Processed_Reviews'].str.len() > 0]
        
        # Generate embeddings
        print("🧠 Generating semantic embeddings...")
        embeddings = self.embedding_model.encode(data['Processed_Reviews'].tolist(), show_progress_bar=True)
        
        # Determine optimal number of topics (between 3-6)
        n_topics = min(max(3, len(data) // 15), 6)
        print(f"🎯 Using {n_topics} topics for {len(data)} reviews")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Generate intelligent topic information
        print("🤖 Generating intelligent topic names and descriptions...")
        topic_summaries = self.generate_intelligent_topic_names(
            cluster_labels, data['Processed_Reviews'], data['Reviews'], product_category
        )
        
        # Calculate overall sentiment
        print("😊 Analyzing overall sentiment...")
        data['Sentiment_Scores'] = data['Reviews'].apply(
            lambda x: self.sia.polarity_scores(str(x)) if pd.notna(x) and str(x).strip() else {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
        )
        data['Compound_Score'] = data['Sentiment_Scores'].apply(lambda x: x['compound'])
        
        # Prepare output data
        output_data = {
            'product_name': product_name,
            'product_category': product_category,
            'analysis_timestamp': datetime.now().isoformat(),
            'model_info': {
                'analysis_type': 'intelligent_topic_modeling',
                'embedding_model': 'all-MiniLM-L6-v2',
                'clustering_algorithm': 'K-means',
                'n_topics': n_topics
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
        
        # Add individual review analysis
        reviews_analysis = []
        for idx, row in data.iterrows():
            topic_idx = cluster_labels[list(data.index).index(idx)]
            reviews_analysis.append({
                'review': str(row['Reviews']),
                'topic': topic_summaries[topic_idx]['topic'],
                'topic_theme': topic_summaries[topic_idx]['primary_theme'],
                'sentiment': row['Sentiment_Scores'],
                'processed_text': row['Processed_Reviews']
            })
        
        output_data['reviews'] = reviews_analysis
        
        # Save results
        json_path = os.path.join(product_output_dir, f'intelligent_analysis_{product_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Create visualizations
        self.create_visualizations(topic_summaries, cluster_labels, data, product_output_dir, product_name)
        
        print(f"✅ Analysis complete! Results saved to {json_path}")
        print("\n📊 Topics discovered:")
        for i, topic in enumerate(topic_summaries, 1):
            print(f"   {i}. {topic['topic']} ({topic['num_reviews']} reviews)")
        
        return product_name
    
    def create_visualizations(self, topic_summaries, cluster_labels, data, output_dir, product_name):
        """Create enhanced visualizations"""
        
        # 1. Topic distribution with intelligent names
        plt.figure(figsize=(12, 8))
        topic_names = [t['topic'] for t in topic_summaries]
        topic_counts = [t['num_reviews'] for t in topic_summaries]
        colors = plt.cm.Set3(np.linspace(0, 1, len(topic_names)))
        
        plt.pie(topic_counts, labels=topic_names, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title(f'Intelligent Topic Distribution - {product_name}', 
                 fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(output_dir, f'intelligent_topic_distribution_{product_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Enhanced word clouds with topic names
        plt.figure(figsize=(20, 12))
        for i, topic in enumerate(topic_summaries):
            words = ' '.join(topic['top_words'])
            if words.strip():
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    colormap='viridis').generate(words)
                
                plt.subplot(2, 3, i + 1)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f"{topic['topic']}\n({topic['num_reviews']} reviews)", 
                         fontsize=12, fontweight='bold')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'intelligent_wordclouds_{product_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sentiment by intelligent topics
        plt.figure(figsize=(14, 8))
        sentiments = []
        names = []
        for topic in topic_summaries:
            sentiments.append([
                topic['sentiment']['positive'],
                topic['sentiment']['negative'], 
                topic['sentiment']['neutral']
            ])
            names.append(topic['topic'])
        
        sentiments = np.array(sentiments)
        x = np.arange(len(names))
        width = 0.25
        
        plt.bar(x - width, sentiments[:, 0], width, label='Positive', color='green', alpha=0.7)
        plt.bar(x, sentiments[:, 1], width, label='Negative', color='red', alpha=0.7)
        plt.bar(x + width, sentiments[:, 2], width, label='Neutral', color='gray', alpha=0.7)
        
        plt.xlabel('Intelligent Topics')
        plt.ylabel('Average Sentiment Score')
        plt.title(f'Sentiment Analysis by Intelligent Topics - {product_name}')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'intelligent_sentiment_analysis_{product_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations created for {product_name}")


def main():
    """Main function to run intelligent topic modeling for all products"""
    
    print("🚀 Starting Intelligent Topic Modeling with Descriptive Names")
    print("=" * 70)
    
    # Initialize the system
    analyzer = IntelligentTopicModeling()
    
    # Create output directory
    output_dir = 'intelligent_analysis_output'
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
            if product:
                processed_products.append(product)
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            continue
    
    # Create manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'intelligent_topic_modeling',
        'products_analyzed': len(processed_products),
        'products': processed_products,
        'description': 'Intelligent topic modeling with automatically generated descriptive topic names'
    }
    
    with open(os.path.join(output_dir, 'intelligent_analysis_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n🎉 Intelligent Topic Modeling Complete!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Products analyzed: {len(processed_products)}")
    print("\n✨ Each product now has meaningful topic names instead of generic 'Topic 1', 'Topic 2'!")


if __name__ == "__main__":
    main()
