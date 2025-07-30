#!/usr/bin/env python3
"""
Test script to check original data loading functionality
"""

import json
import os

def test_original_data_loading():
    """Test if original data can be loaded correctly"""
    print("🔍 Testing Original Data Loading...")
    print("=" * 50)
    
    # Check if original data files exist
    products = ["Desk", "Final", "Iphone16", "Jeans", "Shoe", "Watch"]
    
    for product in products:
        file_path = f"public/data/{product}/topic_modeling_output_{product}.json"
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check data structure
                if isinstance(data, list) and len(data) > 0:
                    # First element should be topic summaries
                    topic_summaries = data[0].get('topic_summaries', [])
                    reviews = data[1:] if len(data) > 1 else []
                    
                    print(f"✅ {product}:")
                    print(f"   - Topics: {len(topic_summaries)}")
                    print(f"   - Reviews: {len(reviews)}")
                    
                    # Check topic structure
                    if topic_summaries:
                        sample_topic = topic_summaries[0]
                        print(f"   - Sample topic: {sample_topic.get('topic', 'N/A')}")
                        print(f"   - Top words: {len(sample_topic.get('top_words', []))}")
                        print(f"   - Sentiment: {sample_topic.get('sentiment', {}).get('compound', 'N/A')}")
                    
                    # Check review structure
                    if reviews:
                        sample_review = reviews[0]
                        print(f"   - Sample review topic: {sample_review.get('topic', 'N/A')}")
                        print(f"   - Review sentiment: {sample_review.get('sentiment', {}).get('compound', 'N/A')}")
                
                else:
                    print(f"❌ {product}: Invalid data structure")
                    
            except Exception as e:
                print(f"❌ {product}: Error loading - {e}")
                
        else:
            print(f"❌ {product}: File not found at {file_path}")
        
        print()

def test_url_access():
    """Test URL access patterns"""
    print("🌐 Testing URL Access Patterns...")
    print("=" * 50)
    
    try:
        import requests
        
        # Test products.json
        try:
            response = requests.get("http://localhost:3000/products.json", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ products.json: {len(data.get('products', []))} products")
            else:
                print(f"❌ products.json: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ products.json: {e}")
        
        # Test original data files
        products = ["Desk", "Iphone16"]  # Test a couple
        for product in products:
            try:
                url = f"http://localhost:3000/data/{product}/topic_modeling_output_{product}.json"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {product} original data: {len(data)} items")
                else:
                    print(f"❌ {product} original data: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {product} original data: {e}")
                
    except ImportError:
        print("⚠️  'requests' module not available. Cannot test URL access.")
    
    print()

def simulate_dashboard_processing():
    """Simulate how the dashboard processes original data"""
    print("🧠 Simulating Dashboard Processing...")
    print("=" * 50)
    
    product = "Desk"  # Test with Desk
    file_path = f"public/data/{product}/topic_modeling_output_{product}.json"
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Simulate getTopicSummaries()
        topic_summaries = data[0].get('topic_summaries', []) if data else []
        print(f"📊 Topic Summaries: {len(topic_summaries)}")
        for i, topic in enumerate(topic_summaries[:3]):  # Show first 3
            print(f"   {i+1}. {topic.get('topic')}: {topic.get('num_reviews')} reviews")
        
        # Simulate getReviews()
        reviews = data[1:] if len(data) > 1 else []
        print(f"📝 Reviews: {len(reviews)}")
        if reviews:
            print(f"   Sample: {reviews[0].get('review', '')[:50]}...")
        
        # Simulate getOverallStats()
        total_reviews = len(reviews)
        avg_sentiment = sum(r.get('sentiment', {}).get('compound', 0) for r in reviews) / len(reviews) if reviews else 0
        positive = len([r for r in reviews if r.get('sentiment', {}).get('compound', 0) >= 0.05])
        negative = len([r for r in reviews if r.get('sentiment', {}).get('compound', 0) <= -0.05])
        neutral = len(reviews) - positive - negative
        
        print(f"📈 Overall Stats:")
        print(f"   Total Reviews: {total_reviews}")
        print(f"   Avg Sentiment: {avg_sentiment:.2f}")
        print(f"   Positive: {positive}, Negative: {negative}, Neutral: {neutral}")
    
    print()

def main():
    """Main function"""
    print("🚀 Original Data Test Tool")
    print("=" * 60)
    print()
    
    test_original_data_loading()
    test_url_access()
    simulate_dashboard_processing()
    
    print("💡 Recommendations:")
    print("1. Ensure React server is running (npm start)")
    print("2. Check browser console for errors when switching to 'Original'")
    print("3. Verify data files are accessible via HTTP")
    print("4. Clear browser cache and try again")

if __name__ == "__main__":
    main()
