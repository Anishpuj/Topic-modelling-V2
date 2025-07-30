#!/usr/bin/env python3
"""
Enhanced Topic Modeling Project Setup Script
============================================

This script helps set up the enhanced topic modeling project with local LLM support.
It handles installation of dependencies, model downloads, and initial configuration.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔄 {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_python_dependencies():
    """Install required Python packages"""
    packages = [
        "pandas", "numpy", "nltk", "scikit-learn", 
        "matplotlib", "seaborn", "wordcloud",
        "transformers", "torch", "sentence-transformers",
        "ollama"  # Optional, for Ollama integration
    ]
    
    print("📦 Installing Python dependencies...")
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    return True

def check_node_installation():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js {result.stdout.strip()} detected")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
    return False

def install_node_dependencies():
    """Install Node.js dependencies"""
    if not os.path.exists("package.json"):
        print("❌ package.json not found. Make sure you're in the project root directory.")
        return False
    
    print("📦 Installing Node.js dependencies...")
    return run_command("npm install", "Installing React dependencies")

def setup_ollama():
    """Setup Ollama if available"""
    print("🤖 Setting up Ollama (optional)...")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama detected")
            
            # Pull a lightweight model
            print("📥 Pulling lightweight LLM model...")
            if run_command("ollama pull llama3.2:3b", "Downloading Llama 3.2 3B model"):
                print("✅ Ollama model ready!")
            else:
                print("⚠️  Model download failed, but you can try later with: ollama pull llama3.2:3b")
            return True
    except FileNotFoundError:
        print("ℹ️  Ollama not found. Install from https://ollama.ai/ for enhanced LLM features")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    nltk_script = """
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print("✅ NLTK data downloaded")
except Exception as e:
    print(f"❌ NLTK download error: {e}")
"""
    
    with open("temp_nltk_download.py", "w") as f:
        f.write(nltk_script)
    
    success = run_command("python temp_nltk_download.py", "Downloading NLTK data")
    
    # Clean up
    if os.path.exists("temp_nltk_download.py"):
        os.remove("temp_nltk_download.py")
    
    return success

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "analysis_settings": {
            "default_llm_type": "rule_based",  # Options: ollama, huggingface, rule_based
            "ollama_model": "llama3.2:3b",
            "num_topics": 5,
            "min_reviews_per_topic": 3
        },
        "visualization_settings": {
            "chart_colors": ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6"],
            "enable_animations": True
        },
        "server_settings": {
            "port": 3000,
            "enable_cors": True
        }
    }
    
    config_path = "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration file created: {config_path}")
    return True

def check_csv_files():
    """Check if CSV files are present"""
    csv_files = list(Path(".").glob("Reviews_*.csv"))
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"   - {csv_file.name}")
        return True
    else:
        print("⚠️  No CSV files found matching 'Reviews_*.csv' pattern")
        print("   Please add your review CSV files to the project root directory")
        return False

def run_initial_analysis():
    """Run the initial analysis to test the setup"""
    print("🔍 Running initial analysis test...")
    
    if not os.path.exists("enhanced_topic_modeling_llm.py"):
        print("❌ Enhanced analysis script not found")
        return False
    
    # Run a quick test
    return run_command("python -c \"from enhanced_topic_modeling_llm import LocalLLMTopicModeling; print('✅ Analysis modules working')\"", 
                      "Testing analysis modules")

def main():
    """Main setup function"""
    print("🚀 Enhanced Topic Modeling Project Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        return False
    
    if not check_node_installation():
        return False
    
    # Install dependencies
    install_python_dependencies()
    install_node_dependencies()
    
    # Setup additional components
    download_nltk_data()
    setup_ollama()
    create_sample_config()
    
    # Check data files
    check_csv_files()
    
    # Test installation
    run_initial_analysis()
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Add your CSV files (Reviews_*.csv) to the project directory")
    print("2. Run analysis: python enhanced_topic_modeling_llm.py")
    print("3. Start the dashboard: npm start")
    print("4. Open http://localhost:3000 in your browser")
    
    print("\n📖 Usage Options:")
    print("- For basic analysis: Use rule-based approach (fastest)")
    print("- For advanced analysis: Install Ollama and use local LLM")
    print("- For custom models: Use Hugging Face transformers")
    
    print("\n🔧 Configuration:")
    print("- Edit config.json to customize settings")
    print("- Modify enhanced_topic_modeling_llm.py for custom analysis")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
