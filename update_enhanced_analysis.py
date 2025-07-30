import json
import os

def update_enhanced_analysis():
    """Update enhanced analysis files with the new topic names"""
    
    # First, load the mapping from the updated topic modeling files
    topic_mappings = {}
    data_dir = "public/data"
    
    for product_folder in os.listdir(data_dir):
        product_path = os.path.join(data_dir, product_folder)
        if os.path.isdir(product_path):
            json_file = os.path.join(product_path, f"topic_modeling_output_{product_folder}.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if len(data) > 0 and 'topic_summaries' in data[0]:
                        # Create mapping from old topic names to new ones
                        topic_summaries = data[0]['topic_summaries']
                        product_mapping = {}
                        for i, topic in enumerate(topic_summaries):
                            old_name = f"Topic {i+1}"
                            new_name = topic['topic']
                            product_mapping[old_name] = new_name
                        topic_mappings[product_folder] = product_mapping
    
    # Update enhanced analysis files
    enhanced_dir = "enhanced_analysis_output"
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
                    
                    # Update topic names in enhanced analysis
                    mapping = topic_mappings[product_folder]
                    updated = False
                    
                    if 'topics' in data:
                        for topic_key in list(data['topics'].keys()):
                            if topic_key in mapping:
                                new_key = mapping[topic_key]
                                data['topics'][new_key] = data['topics'].pop(topic_key)
                                updated = True
                    
                    if 'sentiment_by_topic' in data:
                        for topic_key in list(data['sentiment_by_topic'].keys()):
                            if topic_key in mapping:
                                new_key = mapping[topic_key]
                                data['sentiment_by_topic'][new_key] = data['sentiment_by_topic'].pop(topic_key)
                                updated = True
                    
                    if 'topic_trends' in data:
                        for topic_key in list(data['topic_trends'].keys()):
                            if topic_key in mapping:
                                new_key = mapping[topic_key]
                                data['topic_trends'][new_key] = data['topic_trends'].pop(topic_key)
                                updated = True
                    
                    if updated:
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        print(f"Updated enhanced analysis for {product_folder}")
    
    # Also update public enhanced analysis files
    public_enhanced_dir = "public/enhanced_analysis_output"
    if os.path.exists(public_enhanced_dir):
        for product_folder in os.listdir(public_enhanced_dir):
            if product_folder == "analysis_manifest.json":
                continue
                
            product_path = os.path.join(public_enhanced_dir, product_folder)
            if os.path.isdir(product_path):
                json_file = os.path.join(product_path, f"enhanced_analysis_{product_folder}.json")
                if os.path.exists(json_file) and product_folder in topic_mappings:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Update topic names in enhanced analysis
                    mapping = topic_mappings[product_folder]
                    updated = False
                    
                    if 'topics' in data:
                        for topic_key in list(data['topics'].keys()):
                            if topic_key in mapping:
                                new_key = mapping[topic_key]
                                data['topics'][new_key] = data['topics'].pop(topic_key)
                                updated = True
                    
                    if 'sentiment_by_topic' in data:
                        for topic_key in list(data['sentiment_by_topic'].keys()):
                            if topic_key in mapping:
                                new_key = mapping[topic_key]
                                data['sentiment_by_topic'][new_key] = data['sentiment_by_topic'].pop(topic_key)
                                updated = True
                    
                    if 'topic_trends' in data:
                        for topic_key in list(data['topic_trends'].keys()):
                            if topic_key in mapping:
                                new_key = mapping[topic_key]
                                data['topic_trends'][new_key] = data['topic_trends'].pop(topic_key)
                                updated = True
                    
                    if updated:
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        print(f"Updated public enhanced analysis for {product_folder}")

def main():
    print("Updating enhanced analysis files with new topic names...")
    update_enhanced_analysis()
    print("Enhanced analysis files updated successfully!")

if __name__ == "__main__":
    main()
