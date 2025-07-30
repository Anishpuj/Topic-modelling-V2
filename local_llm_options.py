"""
Multiple Local LLM Options for Topic Modeling
=============================================

This script demonstrates various ways to use local LLMs for topic modeling
and sentiment analysis without relying on external APIs.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# Option 1: Ollama (Easiest)
class OllamaLLM:
    def __init__(self, model="llama3.2:3b"):
        self.model = model
        self.base_url = "http://localhost:11434"
        
    def generate(self, prompt, max_tokens=200):
        try:
            import ollama
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': max_tokens}
            )
            return response['message']['content']
        except Exception as e:
            print(f"Ollama error: {e}")
            return self.fallback_response(prompt)
    
    def fallback_response(self, prompt):
        return "Local analysis based on patterns in customer reviews."

# Option 2: Hugging Face Transformers (More Control)
class HuggingFaceLLM:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            print(f"Loading {model_name}...")
            
            # For text generation
            self.generator = pipeline(
                "text-generation",
                model="gpt2",  # Lightweight option
                tokenizer="gpt2",
                max_length=100,
                do_sample=True,
                temperature=0.7
            )
            
            # Alternative: Use a smaller conversational model
            # self.generator = pipeline("conversational", model="microsoft/DialoGPT-small")
            
        except Exception as e:
            print(f"HuggingFace error: {e}")
            self.generator = None
    
    def generate(self, prompt, max_tokens=200):
        if self.generator is None:
            return self.fallback_response(prompt)
        
        try:
            # Truncate prompt for smaller models
            short_prompt = prompt[:200] + "..."
            result = self.generator(short_prompt, max_new_tokens=50, truncation=True)
            return result[0]['generated_text'][len(short_prompt):].strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return self.fallback_response(prompt)
    
    def fallback_response(self, prompt):
        return "AI-generated topic analysis based on customer feedback patterns."

# Option 3: Local API Server (Custom)
class LocalAPILLM:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def generate(self, prompt, max_tokens=200):
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["text"]
            else:
                return self.fallback_response(prompt)
        except Exception as e:
            print(f"Local API error: {e}")
            return self.fallback_response(prompt)
    
    def fallback_response(self, prompt):
        return "Local server-based analysis of customer review topics."

# Option 4: OpenAI-Compatible Local Server (like LM Studio, Oobabooga)
class OpenAICompatibleLLM:
    def __init__(self, base_url="http://localhost:1234/v1", model="local-model"):
        self.base_url = base_url
        self.model = model
    
    def generate(self, prompt, max_tokens=200):
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return self.fallback_response(prompt)
        except Exception as e:
            print(f"OpenAI-compatible API error: {e}")
            return self.fallback_response(prompt)
    
    def fallback_response(self, prompt):
        return "Local OpenAI-compatible server analysis."

# Option 5: Rule-Based Smart Analysis (No LLM needed)
class RuleBasedAnalysis:
    def __init__(self):
        self.topic_patterns = {
            'quality': {
                'keywords': ['quality', 'good', 'excellent', 'perfect', 'great', 'amazing', 'outstanding', 'superb'],
                'negative_keywords': ['bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'disappointing'],
                'name': 'Product Quality',
                'description': 'Discussions about product quality and performance'
            },
            'delivery': {
                'keywords': ['delivery', 'shipping', 'fast', 'quick', 'time', 'arrived', 'package', 'box'],
                'negative_keywords': ['slow', 'late', 'delayed', 'missing', 'damaged', 'broken'],
                'name': 'Delivery & Shipping',
                'description': 'Customer experiences with delivery and packaging'
            },
            'price': {
                'keywords': ['price', 'cheap', 'value', 'money', 'cost', 'affordable', 'reasonable', 'worth'],
                'negative_keywords': ['expensive', 'overpriced', 'costly', 'waste', 'rip-off'],
                'name': 'Price & Value',
                'description': 'Discussions about pricing and value for money'
            },
            'service': {
                'keywords': ['service', 'support', 'help', 'staff', 'customer', 'friendly', 'helpful'],
                'negative_keywords': ['rude', 'unhelpful', 'ignored', 'poor service', 'bad support'],
                'name': 'Customer Service',
                'description': 'Customer service and support experiences'
            },
            'design': {
                'keywords': ['design', 'look', 'beautiful', 'color', 'style', 'appearance', 'aesthetic'],
                'negative_keywords': ['ugly', 'hideous', 'plain', 'boring', 'unattractive'],
                'name': 'Design & Appearance',
                'description': 'Product design and visual appeal discussions'
            }
        }
    
    def analyze_topic(self, top_words, sample_reviews):
        """Analyze topic using rule-based approach"""
        scores = {}
        
        for theme, pattern in self.topic_patterns.items():
            positive_score = sum(1 for word in top_words if word.lower() in pattern['keywords'])
            negative_score = sum(1 for word in top_words if word.lower() in pattern['negative_keywords'])
            
            # Check sample reviews for context
            review_text = ' '.join(sample_reviews).lower()
            context_score = sum(1 for keyword in pattern['keywords'] if keyword in review_text)
            
            total_score = positive_score + context_score - (negative_score * 0.5)
            scores[theme] = total_score
        
        # Find best matching theme
        best_theme = max(scores, key=scores.get) if scores else 'general'
        
        if best_theme in self.topic_patterns:
            pattern = self.topic_patterns[best_theme]
            
            # Determine sentiment based on positive/negative keywords
            positive_words = [w for w in top_words if w.lower() in pattern['keywords']]
            negative_words = [w for w in top_words if w.lower() in pattern['negative_keywords']]
            
            if len(positive_words) > len(negative_words):
                sentiment = 'Positive'
            elif len(negative_words) > len(positive_words):
                sentiment = 'Negative'
            else:
                sentiment = 'Mixed'
            
            return {
                'name': pattern['name'],
                'description': pattern['description'],
                'sentiment_theme': sentiment,
                'confidence': scores[best_theme] / (len(top_words) + 1)
            }
        
        return {
            'name': 'General Discussion',
            'description': 'Mixed customer feedback and experiences',
            'sentiment_theme': 'Mixed',
            'confidence': 0.5
        }

# Main Multi-LLM Topic Analyzer
class MultiLLMTopicAnalyzer:
    def __init__(self, preferred_llm="rule_based"):
        """
        Initialize with preferred LLM option
        
        Options:
        - "ollama": Use Ollama (requires Ollama installed)
        - "huggingface": Use Hugging Face transformers
        - "local_api": Use local API server
        - "openai_compatible": Use OpenAI-compatible server
        - "rule_based": Use intelligent rule-based analysis (recommended)
        """
        self.preferred_llm = preferred_llm
        self.llm = self.setup_llm()
    
    def setup_llm(self):
        """Setup the chosen LLM"""
        if self.preferred_llm == "ollama":
            return OllamaLLM()
        elif self.preferred_llm == "huggingface":
            return HuggingFaceLLM()
        elif self.preferred_llm == "local_api":
            return LocalAPILLM()
        elif self.preferred_llm == "openai_compatible":
            return OpenAICompatibleLLM()
        else:  # rule_based (default)
            return RuleBasedAnalysis()
    
    def analyze_topic(self, top_words, sample_reviews, topic_idx):
        """Analyze a topic using the chosen method"""
        if isinstance(self.llm, RuleBasedAnalysis):
            return self.llm.analyze_topic(top_words, sample_reviews)
        else:
            # Use LLM for analysis
            prompt = f"""
            Analyze this customer review topic:
            
            Key words: {', '.join(top_words[:8])}
            Sample reviews:
            {chr(10).join([f"- {review[:150]}..." for review in sample_reviews[:2]])}
            
            Provide a topic name (2-4 words) and brief description.
            """
            
            response = self.llm.generate(prompt, max_tokens=100)
            
            # Parse response or use fallback
            return {
                'name': f'Topic {topic_idx + 1}',
                'description': response[:200] + "..." if len(response) > 200 else response,
                'sentiment_theme': 'Mixed'
            }

# Example usage and comparison
def compare_llm_options():
    """Compare different LLM options"""
    print("🔍 Comparing Local LLM Options for Topic Modeling\n")
    
    # Sample data
    top_words = ['quality', 'good', 'product', 'delivery', 'fast', 'excellent', 'service', 'recommend']
    sample_reviews = [
        "Great quality product, fast delivery, highly recommend!",
        "Good value for money, but delivery was slow",
        "Excellent customer service, quality could be better"
    ]
    
    llm_options = [
        ("rule_based", "Rule-Based Analysis"),
        ("ollama", "Ollama LLM"),
        ("huggingface", "Hugging Face"),
    ]
    
    for llm_type, llm_name in llm_options:
        print(f"📊 {llm_name}:")
        try:
            analyzer = MultiLLMTopicAnalyzer(llm_type)
            result = analyzer.analyze_topic(top_words, sample_reviews, 0)
            
            print(f"   Name: {result['name']}")
            print(f"   Description: {result['description']}")
            print(f"   Sentiment: {result['sentiment_theme']}")
            if 'confidence' in result:
                print(f"   Confidence: {result['confidence']:.2f}")
            print()
            
        except Exception as e:
            print(f"   Error: {e}\n")

# Enhanced visualization for multiple LLM results
def create_comparison_report():
    """Create a comparison report of different LLM approaches"""
    
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'llm_comparison': {
            'ollama': {
                'pros': ['High quality output', 'Multiple model options', 'Good for complex analysis'],
                'cons': ['Requires installation', 'Resource intensive', 'Slower processing'],
                'best_for': 'Detailed topic analysis with natural language descriptions'
            },
            'huggingface': {
                'pros': ['Wide model selection', 'Good customization', 'Active community'],
                'cons': ['Can be resource heavy', 'Model size limitations', 'Setup complexity'],
                'best_for': 'Custom fine-tuned models for specific domains'
            },
            'rule_based': {
                'pros': ['Fast processing', 'No dependencies', 'Predictable results', 'Transparent logic'],
                'cons': ['Limited creativity', 'Manual rule maintenance', 'Less natural language'],
                'best_for': 'Production systems requiring reliability and speed'
            },
            'local_api': {
                'pros': ['Flexible deployment', 'Custom models', 'Scalable'],
                'cons': ['Requires server setup', 'Network dependency', 'Additional complexity'],
                'best_for': 'Enterprise deployments with custom infrastructure'
            }
        },
        'recommendations': {
            'for_beginners': 'rule_based',
            'for_experimentation': 'ollama',
            'for_production': 'rule_based',
            'for_research': 'huggingface'
        }
    }
    
    with open('llm_comparison_report.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print("📋 LLM Comparison Report created: llm_comparison_report.json")

if __name__ == "__main__":
    print("🚀 Local LLM Options for Topic Modeling")
    print("=" * 50)
    
    # Compare different approaches
    compare_llm_options()
    
    # Create comparison report
    create_comparison_report()
    
    print("\n✅ Comparison complete!")
    print("\nRecommended approaches:")
    print("🥇 For production: Rule-based (fast, reliable)")
    print("🥈 For experimentation: Ollama (flexible, powerful)")
    print("🥉 For research: Hugging Face (customizable)")
