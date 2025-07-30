#!/usr/bin/env python3
"""
Final verification script for both enhanced and original dashboard modes
"""

import json
import os

def verify_enhanced_data():
    """Verify enhanced data structure"""
    print("🚀 Verifying Enhanced Data...")
    print("=" * 50)
    
    product = "Desk"
    file_path = f"public/enhanced_analysis_output/{product}/enhanced_analysis_{product}.json"
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check enhanced structure
        print(f"✅ Enhanced data for {product}:")
        print(f"   - Product name: {data.get('product_name', 'N/A')}")
        print(f"   - Analysis timestamp: {data.get('analysis_timestamp', 'N/A')}")
        print(f"   - Topic summaries: {len(data.get('topic_summaries', []))}")
        print(f"   - Overall stats: {'✅' if data.get('overall_stats') else '❌'}")
        print(f"   - Model info: {'✅' if data.get('model_info') else '❌'}")
        print(f"   - Reviews: {len(data.get('reviews', []))}")
        
        # Check sample topic
        if data.get('topic_summaries'):
            topic = data['topic_summaries'][0]
            print(f"   - Sample topic LLM name: {topic.get('llm_generated_name', 'N/A')}")
            print(f"   - Sample topic sentiment: {topic.get('sentiment', {})}")
    else:
        print(f"❌ Enhanced data file not found: {file_path}")
    
    print()

def verify_original_data():
    """Verify original data structure"""
    print("🔧 Verifying Original Data...")
    print("=" * 50)
    
    product = "Desk"
    file_path = f"public/data/{product}/topic_modeling_output_{product}.json"
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check original structure
        print(f"✅ Original data for {product}:")
        print(f"   - Data type: {type(data)}")
        print(f"   - Total items: {len(data)}")
        
        if isinstance(data, list) and len(data) > 0:
            # First item should be metadata
            metadata = data[0]
            topic_summaries = metadata.get('topic_summaries', [])
            
            print(f"   - Product name: {metadata.get('product_name', 'N/A')}")
            print(f"   - Topic summaries: {len(topic_summaries)}")
            
            # Check sample topic
            if topic_summaries:
                topic = topic_summaries[0]
                print(f"   - Sample topic: {topic.get('topic', 'N/A')}")
                print(f"   - Sample topic sentiment: {topic.get('sentiment', {})}")
            
            # Reviews should be items 1+
            reviews = data[1:]
            print(f"   - Reviews: {len(reviews)}")
            
            if reviews:
                sample_review = reviews[0]
                print(f"   - Sample review topic: {sample_review.get('topic', 'N/A')}")
                print(f"   - Sample review sentiment: {sample_review.get('sentiment', {})}")
    
    else:
        print(f"❌ Original data file not found: {file_path}")
    
    print()

def test_dashboard_compatibility():
    """Test dashboard data processing compatibility"""
    print("🧪 Testing Dashboard Data Processing...")
    print("=" * 50)
    
    # Test Enhanced Data Processing
    print("📊 Enhanced Data Processing:")
    enhanced_file = "public/enhanced_analysis_output/Desk/enhanced_analysis_Desk.json"
    if os.path.exists(enhanced_file):
        with open(enhanced_file, 'r') as f:
            enhanced_data = json.load(f)
        
        # Simulate dashboard functions
        topic_summaries = enhanced_data.get('topic_summaries', [])
        reviews = enhanced_data.get('reviews', [])
        overall_stats = enhanced_data.get('overall_stats', {})
        
        print(f"   - getTopicSummaries(): {len(topic_summaries)} topics")
        print(f"   - getReviews(): {len(reviews)} reviews")
        print(f"   - getOverallStats(): {len(overall_stats)} stats")
        
        # Test chart data preparation
        if topic_summaries:
            labels = [t.get('llm_generated_name', t.get('topic', '')) for t in topic_summaries]
            data_points = [t.get('num_reviews', 0) for t in topic_summaries]
            print(f"   - Chart labels: {labels[:3]}...")
            print(f"   - Chart data: {data_points[:3]}...")
    
    print()
    
    # Test Original Data Processing
    print("🔧 Original Data Processing:")
    original_file = "public/data/Desk/topic_modeling_output_Desk.json"
    if os.path.exists(original_file):
        with open(original_file, 'r') as f:
            original_data = json.load(f)
        
        # Simulate dashboard functions for original data
        topic_summaries = original_data[0].get('topic_summaries', []) if original_data else []
        reviews = original_data[1:] if len(original_data) > 1 else []
        
        # Simulate getOverallStats() calculation
        total_reviews = len(reviews)
        avg_sentiment = sum(r.get('sentiment', {}).get('compound', 0) for r in reviews) / len(reviews) if reviews else 0
        positive = len([r for r in reviews if r.get('sentiment', {}).get('compound', 0) >= 0.05])
        negative = len([r for r in reviews if r.get('sentiment', {}).get('compound', 0) <= -0.05])
        neutral = len(reviews) - positive - negative
        
        print(f"   - getTopicSummaries(): {len(topic_summaries)} topics")
        print(f"   - getReviews(): {len(reviews)} reviews")
        print(f"   - Calculated stats: Total={total_reviews}, Avg={avg_sentiment:.2f}")
        print(f"   - Sentiment distribution: Pos={positive}, Neg={negative}, Neu={neutral}")
        
        # Test chart data preparation
        if topic_summaries:
            labels = [t.get('topic', '') for t in topic_summaries]
            data_points = [t.get('num_reviews', 0) for t in topic_summaries]
            print(f"   - Chart labels: {labels[:3]}...")
            print(f"   - Chart data: {data_points[:3]}...")
    
    print()

def main():
    """Main verification function"""
    print("🎯 Dashboard Fix Verification")
    print("=" * 60)
    print()
    
    verify_enhanced_data()
    verify_original_data()
    test_dashboard_compatibility()
    
    print("✅ Verification Complete!")
    print()
    print("📋 What to Test Next:")
    print("1. Open http://localhost:3000 in your browser")
    print("2. Verify 'Enhanced' mode loads correctly")
    print("3. Click 'Original' button and verify it switches properly")
    print("4. Check that all charts and data display correctly in both modes")
    print("5. Test switching between different products")
    print("6. Clear browser cache if you see any issues")

if __name__ == "__main__":
    main()
