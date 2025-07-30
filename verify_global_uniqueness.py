import json
import os
from collections import Counter

def verify_global_uniqueness():
    """Verify that all topic names are globally unique across all products"""
    
    data_dir = "public/data"
    all_topic_names = []
    product_topics = {}
    
    # Collect all topic names from all products
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
                    product_topics[product_folder] = topic_names
                    all_topic_names.extend(topic_names)
    
    # Count occurrences of each topic name
    topic_counts = Counter(all_topic_names)
    duplicates = {topic: count for topic, count in topic_counts.items() if count > 1}
    
    # Print verification results
    print("🔍 GLOBAL TOPIC NAME UNIQUENESS VERIFICATION")
    print("=" * 60)
    
    print(f"\n📊 STATISTICS:")
    print(f"   Total Products: {len(product_topics)}")
    print(f"   Total Topics: {len(all_topic_names)}")
    print(f"   Unique Topic Names: {len(set(all_topic_names))}")
    print(f"   Duplicate Names Found: {len(duplicates)}")
    
    if duplicates:
        print(f"\n❌ DUPLICATES DETECTED:")
        for topic_name, count in duplicates.items():
            print(f"   '{topic_name}' appears {count} times")
        
        print(f"\n📍 WHERE DUPLICATES APPEAR:")
        for duplicate_topic in duplicates.keys():
            products_with_duplicate = []
            for product, topics in product_topics.items():
                if duplicate_topic in topics:
                    products_with_duplicate.append(product)
            print(f"   '{duplicate_topic}' → {', '.join(products_with_duplicate)}")
    else:
        print(f"\n✅ SUCCESS: All topic names are globally unique!")
    
    print(f"\n📝 ALL TOPIC NAMES BY PRODUCT:")
    print("-" * 40)
    for product, topics in sorted(product_topics.items()):
        print(f"\n🏷️  {product.upper()}:")
        for i, topic in enumerate(topics, 1):
            status = "✅" if topic_counts[topic] == 1 else "❌"
            print(f"   {i}. {status} {topic}")
    
    print(f"\n" + "=" * 60)
    if not duplicates:
        print("🎉 VERIFICATION PASSED: No duplicate topic names found!")
        print("💡 The dashboard will now show completely unique topic names.")
    else:
        print("⚠️  VERIFICATION FAILED: Some topic names are still duplicated.")
        print("🔧 Please run the uniqueness script again.")
    
    return len(duplicates) == 0

if __name__ == "__main__":
    verify_global_uniqueness()
