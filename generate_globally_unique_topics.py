import json
import os
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

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

def generate_unique_topic_name(top_words, product_name, used_names_global):
    """Generate a globally unique topic name based on top words"""
    
    # Define semantic categories with more specific subcategories
    categories = {
        'delivery_packaging': ['delivery', 'packaging', 'arrived', 'shipping', 'package', 'damaged', 'replaced', 'box'],
        'build_quality': ['quality', 'build', 'material', 'durable', 'sturdy', 'strong', 'solid', 'construction'],
        'comfort_feel': ['comfortable', 'comfort', 'soft', 'cushion', 'feel', 'ergonomic', 'cozy'],
        'size_fit': ['fit', 'size', 'fitting', 'tight', 'loose', 'perfect', 'wide', 'narrow'],
        'design_appearance': ['design', 'look', 'style', 'appearance', 'color', 'elegant', 'sleek', 'beautiful'],
        'functionality_performance': ['work', 'function', 'operation', 'performance', 'smooth', 'easy', 'efficient'],
        'price_value': ['price', 'value', 'money', 'worth', 'cost', 'expensive', 'cheap', 'affordable'],
        'customer_service': ['service', 'support', 'company', 'customer', 'help', 'response'],
        'durability_longevity': ['last', 'lasting', 'wear', 'tear', 'broken', 'ripped', 'durable'],
        'height_adjustment': ['adjustment', 'adjustable', 'height', 'control', 'setting', 'preset'],
        'motor_mechanism': ['motor', 'quiet', 'silent', 'noise', 'smooth', 'mechanism'],
        'space_surface': ['space', 'spacious', 'large', 'ample', 'surface', 'wide', 'area'],
        'assembly_setup': ['assembly', 'setup', 'installation', 'assemble', 'install', 'mounting'],
        'battery_power': ['battery', 'charge', 'power', 'charging', 'backup'],
        'display_screen': ['display', 'screen', 'brightness', 'clarity', 'resolution'],
        'camera_photo': ['camera', 'photo', 'picture', 'image', 'lens'],
        'fabric_material': ['fabric', 'cotton', 'denim', 'material', 'texture', 'cloth'],
        'sole_footbed': ['sole', 'footbed', 'insole', 'outsole', 'grip', 'traction'],
        'strap_band': ['strap', 'band', 'buckle', 'clasp', 'wrist'],
        'water_resistance': ['water', 'waterproof', 'resistant', 'splash', 'sweat']
    }
    
    # Count matches for each category
    category_scores = defaultdict(int)
    for word in top_words[:7]:  # Focus on top 7 words
        word_lower = word.lower()
        for category, keywords in categories.items():
            if word_lower in keywords:
                category_scores[category] += 1
    
    # Generate base name
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        category_display = best_category.replace('_', ' ').title()
        base_name = f"{category_display} - {product_name}"
    else:
        # Fallback: use the most meaningful word
        meaningful_words = [w for w in top_words[:5] if len(w) > 3 and w.lower() not in stopwords.words('english')]
        if meaningful_words:
            base_name = f"{meaningful_words[0].title()} Features - {product_name}"
        else:
            base_name = f"General Aspects - {product_name}"
    
    # Ensure global uniqueness
    final_name = base_name
    counter = 1
    while final_name in used_names_global:
        counter += 1
        final_name = f"{base_name} #{counter}"
    
    used_names_global.add(final_name)
    return final_name

def ensure_globally_unique_topic_names(products_data):
    """Ensure all topic names are globally unique across all products"""
    updated_data = {}
    used_names_global = set()
    
    # Process products in a consistent order
    for product_name in sorted(products_data.keys()):
        product_data = products_data[product_name]
        updated_product_data = product_data.copy()
        
        if len(product_data) > 0 and 'topic_summaries' in product_data[0]:
            topic_summaries = product_data[0]['topic_summaries']
            
            for i, topic in enumerate(topic_summaries):
                # Generate globally unique topic name
                unique_name = generate_unique_topic_name(
                    topic['top_words'], 
                    product_name, 
                    used_names_global
                )
                
                topic['topic'] = unique_name
                topic['description'] = f"Reviews related to {unique_name}"
            
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
    
    print("Updated topic names for all products:")
    print("=" * 50)
    
    for product_name, product_data in updated_data.items():
        product_path = os.path.join(data_dir, product_name)
        json_file = os.path.join(product_path, f"topic_modeling_output_{product_name}.json")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(product_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n📱 {product_name.upper()}:")
        if len(product_data) > 0 and 'topic_summaries' in product_data[0]:
            for i, topic in enumerate(product_data[0]['topic_summaries'], 1):
                print(f"  {i}. {topic['topic']}")

def update_enhanced_analysis_files(updated_data):
    """Update enhanced analysis files with the new globally unique topic names"""
    
    # Create topic mappings for each product
    topic_mappings = {}
    for product_name, product_data in updated_data.items():
        if len(product_data) > 0 and 'topic_summaries' in product_data[0]:
            topic_summaries = product_data[0]['topic_summaries']
            product_mapping = {}
            for i, topic in enumerate(topic_summaries):
                old_name = f"Topic {i+1}"
                new_name = topic['topic']
                product_mapping[old_name] = new_name
            topic_mappings[product_name] = product_mapping
    
    # Update both enhanced analysis directories
    dirs_to_update = ["enhanced_analysis_output", "public/enhanced_analysis_output"]
    
    for enhanced_dir in dirs_to_update:
        if os.path.exists(enhanced_dir):
            for product_folder in os.listdir(enhanced_dir):
                if product_folder == "analysis_manifest.json":
                    continue
                    
                product_path = os.path.join(enhanced_dir, product_folder)
                if os.path.isdir(product_path):
                    json_file = os.path.join(product_path, f"enhanced_analysis_{product_folder}.json")
                    if os.path.exists(json_file) and product_folder in topic_mappings:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        mapping = topic_mappings[product_folder]
                        updated = False
                        
                        # Update all topic references
                        for section in ['topics', 'sentiment_by_topic', 'topic_trends']:
                            if section in data:
                                for topic_key in list(data[section].keys()):
                                    if topic_key in mapping:
                                        new_key = mapping[topic_key]
                                        data[section][new_key] = data[section].pop(topic_key)
                                        updated = True
                        
                        if updated:
                            with open(json_file, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=4, ensure_ascii=False)
                            print(f"✅ Updated {enhanced_dir}/{product_folder}")

def main():
    print("🔄 Generating globally unique topic names...")
    products_data = load_topic_data()
    
    print(f"📊 Found data for {len(products_data)} products")
    
    updated_data = ensure_globally_unique_topic_names(products_data)
    
    print("\n💾 Saving updated data...")
    save_updated_data(updated_data)
    
    print("\n🔧 Updating enhanced analysis files...")
    update_enhanced_analysis_files(updated_data)
    
    print("\n🎉 SUCCESS: All topic names are now globally unique!")
    print("\n💡 Each topic now has a descriptive name that won't repeat across products.")

if __name__ == "__main__":
    main()
