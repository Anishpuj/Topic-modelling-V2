import json
import os
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_topic_data():
    """Load all topic modeling data from products"""
    products_data = {}
    data_dir = "public/data"
    
    for product_folder in os.listdir(data_dir):
        product_path = os.path.join(data_dir, product_folder)
        if os.path.isdir(product_path):
            json_file = os.path.join(product_path, f"topic_modeling_output_{product_folder}.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    products_data[product_folder] = data
    
    return products_data

def generate_topic_name(top_words, product_name):
    """Generate a meaningful topic name based on top words"""
    # Define semantic categories based on common e-commerce review themes
    categories = {
        'delivery': ['delivery', 'packaging', 'arrived', 'shipping', 'package', 'damaged', 'replaced'],
        'quality': ['quality', 'build', 'material', 'durable', 'sturdy', 'strong', 'solid'],
        'comfort': ['comfortable', 'comfort', 'soft', 'cushion', 'feel', 'ergonomic'],
        'fit': ['fit', 'size', 'fitting', 'tight', 'loose', 'perfect'],
        'design': ['design', 'look', 'style', 'appearance', 'color', 'elegant', 'sleek'],
        'functionality': ['work', 'function', 'operation', 'performance', 'smooth', 'easy'],
        'value': ['price', 'value', 'money', 'worth', 'cost', 'expensive', 'cheap'],
        'service': ['service', 'support', 'company', 'customer', 'help'],
        'durability': ['last', 'lasting', 'wear', 'tear', 'broken', 'ripped'],
        'adjustment': ['adjustment', 'adjustable', 'height', 'control', 'setting'],
        'motor': ['motor', 'quiet', 'silent', 'noise', 'smooth'],
        'space': ['space', 'spacious', 'large', 'ample', 'surface', 'wide'],
        'assembly': ['assembly', 'setup', 'installation', 'assemble', 'install']
    }
    
    # Count matches for each category
    category_scores = defaultdict(int)
    for word in top_words:
        word_lower = word.lower()
        for category, keywords in categories.items():
            if word_lower in keywords:
                category_scores[category] += 1
    
    # Find the best matching category
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        return f"{best_category.title()} & {product_name.split()[0] if product_name else 'Product'}"
    else:
        # Fallback: use the most frequent meaningful word
        meaningful_words = [w for w in top_words if len(w) > 3 and w.lower() not in stopwords.words('english')]
        if meaningful_words:
            return f"{meaningful_words[0].title()} & {product_name.split()[0] if product_name else 'Product'}"
        else:
            return f"General & {product_name.split()[0] if product_name else 'Product'}"

def ensure_unique_topic_names(products_data):
    """Ensure all topic names are unique within each product"""
    updated_data = {}
    
    for product_name, product_data in products_data.items():
        updated_product_data = product_data.copy()
        
        if len(product_data) > 0 and 'topic_summaries' in product_data[0]:
            topic_summaries = product_data[0]['topic_summaries']
            used_names = set()
            
            for i, topic in enumerate(topic_summaries):
                # Generate base topic name
                base_name = generate_topic_name(topic['top_words'], product_name)
                
                # Ensure uniqueness by adding counter if needed
                topic_name = base_name
                counter = 1
                while topic_name in used_names:
                    counter += 1
                    topic_name = f"{base_name} ({counter})"
                
                used_names.add(topic_name)
                topic['topic'] = topic_name
                topic['description'] = f"Reviews related to {topic_name}"
            
            # Update all individual reviews with new topic names
            topic_mapping = {f"Topic {i+1}": topic_summaries[i]['topic'] for i in range(len(topic_summaries))}
            
            for j in range(1, len(updated_product_data)):
                if 'topic' in updated_product_data[j]:
                    old_topic = updated_product_data[j]['topic']
                    if old_topic in topic_mapping:
                        updated_product_data[j]['topic'] = topic_mapping[old_topic]
        
        updated_data[product_name] = updated_product_data
    
    return updated_data

def save_updated_data(updated_data):
    """Save the updated data back to files"""
    data_dir = "public/data"
    
    for product_name, product_data in updated_data.items():
        product_path = os.path.join(data_dir, product_name)
        json_file = os.path.join(product_path, f"topic_modeling_output_{product_name}.json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(product_data, f, indent=4, ensure_ascii=False)
        
        print(f"Updated topics for {product_name}:")
        if len(product_data) > 0 and 'topic_summaries' in product_data[0]:
            for topic in product_data[0]['topic_summaries']:
                print(f"  - {topic['topic']}")
        print()

def main():
    print("Loading topic modeling data...")
    products_data = load_topic_data()
    
    print(f"Found data for {len(products_data)} products")
    
    print("\nGenerating unique topic names...")
    updated_data = ensure_unique_topic_names(products_data)
    
    print("\nSaving updated data...")
    save_updated_data(updated_data)
    
    print("Topic name uniqueness process completed!")

if __name__ == "__main__":
    main()
