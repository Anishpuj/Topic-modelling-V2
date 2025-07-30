import json
import os
from collections import defaultdict

def verify_unique_topics():
    """Verify that all topic names are unique within each product"""
    
    data_dir = "public/data"
    all_results = {}
    
    for product_folder in os.listdir(data_dir):
        product_path = os.path.join(data_dir, product_folder)
        if os.path.isdir(product_path):
            json_file = os.path.join(product_path, f"topic_modeling_output_{product_folder}.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if len(data) > 0 and 'topic_summaries' in data[0]:
                    topic_summaries = data[0]['topic_summaries']
                    topic_names = [topic['topic'] for topic in topic_summaries]
                    
                    # Check for duplicates
                    duplicates = []
                    seen = set()
                    for name in topic_names:
                        if name in seen:
                            duplicates.append(name)
                        seen.add(name)
                    
                    all_results[product_folder] = {
                        'topics': topic_names,
                        'duplicates': duplicates,
                        'unique_count': len(set(topic_names)),
                        'total_count': len(topic_names)
                    }
    
    # Print results
    print("=== TOPIC UNIQUENESS VERIFICATION ===\n")
    
    for product, results in all_results.items():
        print(f"Product: {product}")
        print(f"  Total Topics: {results['total_count']}")
        print(f"  Unique Topics: {results['unique_count']}")
        
        if results['duplicates']:
            print(f"  ❌ DUPLICATES FOUND: {results['duplicates']}")
        else:
            print("  ✅ All topics are unique!")
        
        print("  Topic Names:")
        for topic in results['topics']:
            print(f"    - {topic}")
        print()
    
    # Summary
    total_products = len(all_results)
    products_with_duplicates = sum(1 for r in all_results.values() if r['duplicates'])
    
    print("=== SUMMARY ===")
    print(f"Total Products: {total_products}")
    print(f"Products with Duplicates: {products_with_duplicates}")
    print(f"Products with Unique Topics: {total_products - products_with_duplicates}")
    
    if products_with_duplicates == 0:
        print("🎉 SUCCESS: All products have unique topic names!")
    else:
        print("⚠️  Some products still have duplicate topic names.")

if __name__ == "__main__":
    verify_unique_topics()
