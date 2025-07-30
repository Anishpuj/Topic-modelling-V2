# 🚀 Enhanced Topic Modeling & Sentiment Analysis Dashboard
#🔗 [Live Dashboard](https://topcmodellingv2.netlify.app/)
An advanced customer review analysis system that combines traditional NLP techniques with modern Local Large Language Models (LLMs) for sophisticated topic modeling and sentiment analysis.

![Dashboard Preview](https://img.shields.io/badge/Status-Enhanced%20with%20Local%20LLM-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-18+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

### 🎯 **Advanced Topic Modeling**
- **BERT-based embeddings** for semantic understanding
- **K-means clustering** for topic identification
- **Local LLM integration** (Ollama, Hugging Face) for topic naming and description
- **Rule-based analysis** as a fast, reliable fallback
- **Multiple embedding models** (Sentence Transformers, BERT)

### 📊 **Comprehensive Sentiment Analysis**
- **VADER sentiment analysis** with compound scoring
- **Context-aware sentiment extraction** handling negations
- **Aspect-based sentiment analysis** by topic
- **Star rating generation** based on sentiment scores
- **Positive/negative word extraction** with context

### 🎨 **Interactive Dashboard**
- **Modern React interface** with Tailwind CSS
- **Real-time data switching** between analysis types
- **Interactive charts** (Pie, Bar, Line) with Chart.js
- **Responsive design** for all devices
- **Animated components** with Framer Motion
- **Tabbed interface** for organized data exploration

### 🤖 **Local LLM Support**
- **Ollama integration** for local LLM inference
- **Hugging Face Transformers** for custom models
- **Rule-based fallback** for production reliability
- **Multiple model options** (Llama, Mistral, GPT-2, etc.)
- **No API costs** - everything runs locally

## 🛠 Installation

### Quick Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd Reviews_LLM_TOPIC-MODELLING

# Run the automated setup script
python setup_enhanced_project.py
```

### Manual Setup

#### Prerequisites
- **Python 3.8+**
- **Node.js 14+**
- **npm or yarn**

#### Python Dependencies
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud
pip install transformers torch sentence-transformers ollama
```

#### React Dependencies
```bash
npm install
```

#### Optional: Ollama Setup
1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model: `ollama pull llama3.2:3b`

## 🚀 Usage

### 1. Prepare Your Data
Place your CSV files in the project root with the naming pattern `Reviews_*.csv`. Each file should contain a column with review text.

Example files:
- `Reviews_Desk.csv`
- `Reviews_Phone.csv`
- `Reviews_Shoes.csv`

### 2. Run Analysis
```bash
# Enhanced analysis with Local LLM
python enhanced_topic_modeling_llm.py

# Compare different LLM approaches
python local_llm_options.py

# Original BERT-based analysis
python topic_modeling.py
```

### 3. Start Dashboard
```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
Reviews_LLM_TOPIC-MODELLING/
├── 📊 Analysis Scripts
│   ├── enhanced_topic_modeling_llm.py    # Main enhanced analysis
│   ├── local_llm_options.py              # LLM comparison tools
│   └── topic_modeling.py                 # Original BERT analysis
├── ⚛️  React Dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── EnhancedDashboard.js      # Main dashboard
│   │   │   ├── ProjectFlow.js            # Project workflow
│   │   │   └── TopicModelingFlow.js      # Technical flow
│   │   └── App.js                        # Main React app
│   └── public/
│       └── enhanced_analysis_output/     # Analysis results
├── 📄 Data Files
│   ├── Reviews_*.csv                     # Input review data
│   └── config.json                       # Configuration settings
├── 🔧 Setup & Configuration
│   └── setup_enhanced_project.py         # Automated setup
└── 📚 Documentation
    └── README.md                         # This file
```

## 🎛 Configuration

Edit `config.json` to customize the analysis:

```json
{
  "analysis_settings": {
    "default_llm_type": "rule_based",
    "ollama_model": "llama3.2:3b",
    "num_topics": 5,
    "min_reviews_per_topic": 3
  },
  "visualization_settings": {
    "chart_colors": ["#3498db", "#2ecc71", "#e74c3c"],
    "enable_animations": true
  }
}
```

## 🤖 Local LLM Options

### 1. **Ollama** (Recommended for experimentation)
```python
analyzer = LocalLLMTopicModeling(model_type="ollama", llm_model="llama3.2:3b")
```

**Pros:**
- High-quality natural language descriptions
- Multiple model options (Llama, Mistral, Phi3)
- Easy installation and management

**Cons:**
- Requires ~4GB+ RAM
- Slower processing
- Requires Ollama installation

### 2. **Hugging Face Transformers**
```python
analyzer = LocalLLMTopicModeling(model_type="huggingface")
```

**Pros:**
- Thousands of pre-trained models
- Fine-tuning capabilities
- Research-friendly

**Cons:**
- Model size limitations
- Complex setup for some models
- Variable performance

### 3. **Rule-Based Analysis** (Recommended for production)
```python
analyzer = LocalLLMTopicModeling(model_type="rule_based")
```

**Pros:**
- Lightning-fast processing
- No dependencies
- Predictable, reliable results
- Transparent logic

**Cons:**
- Less natural language output
- Manual rule maintenance
- Limited creativity

## 📈 Dashboard Features

### Overview Tab
- **Topic Distribution** - Pie chart showing review distribution across topics
- **Sentiment Distribution** - Overall sentiment breakdown
- **Sentiment by Topic** - Bar chart comparing sentiment across topics

### Topic Analysis Tab
- **Enhanced Topic Cards** with LLM-generated names and descriptions
- **Sample Reviews** for each topic
- **Confidence Scores** and sentiment themes
- **Word Analysis** with top keywords

### Review Details Tab
- **Comprehensive Review Table** with sentiment analysis
- **Context-Aware Word Extraction** handling negations
- **Star Ratings** based on sentiment scores
- **Searchable and Filterable** results

## 🔧 Advanced Usage

### Custom LLM Integration
```python
class CustomLLM:
    def generate(self, prompt, max_tokens=200):
        # Your custom LLM logic here
        return "Custom analysis result"

# Use with the analyzer
analyzer = LocalLLMTopicModeling()
analyzer.llm = CustomLLM()
```

### Batch Processing
```python
# Process multiple product categories
csv_files = glob.glob('Reviews_*.csv')
for csv_file in csv_files:
    analyzer.analyze_product_reviews(csv_file, 'output_dir')
```

### API Integration
The dashboard can be extended to work with REST APIs:
```javascript
// Fetch real-time data
const response = await fetch('/api/analysis/product-name');
const data = await response.json();
```

## 🎨 Customization

### Adding New Visualizations
1. Create new chart components in `src/components/`
2. Import Chart.js components as needed
3. Add to the dashboard tabs

### Custom Sentiment Analysis
Modify the sentiment analysis logic in `enhanced_topic_modeling_llm.py`:
```python
def custom_sentiment_analysis(text):
    # Your custom sentiment logic
    return sentiment_scores
```

### Theme Customization
Edit Tailwind CSS classes in React components for custom styling.

## 🚀 Performance Optimization

### For Large Datasets
- Use **batch processing** for embeddings
- Implement **pagination** in the dashboard
- Consider **database storage** instead of JSON files
- Use **caching** for repeated analyses

### Memory Management
- Use **Sentence Transformers** for lighter embedding models
- Implement **lazy loading** for large datasets
- Clear **model cache** between analyses

## 🔍 Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```

**NLTK Data Missing:**
```python
import nltk
nltk.download('all')
```

**React Build Issues:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Memory Issues:**
- Reduce batch size in embedding generation
- Use smaller models (e.g., `all-MiniLM-L6-v2`)
- Process fewer reviews at once

## 📊 Sample Results

The enhanced analysis provides:

- **Topic Names**: "Product Quality", "Delivery Experience", "Customer Service"
- **LLM Descriptions**: Natural language summaries of each topic
- **Sentiment Themes**: "Positive", "Mixed", "Negative" with reasoning
- **Confidence Scores**: Reliability metrics for each analysis
- **Sample Reviews**: Representative examples for each topic

## 🛡 Security & Privacy

- **Local Processing**: All analysis runs on your machine
- **No Data Transmission**: Reviews never leave your system
- **Configurable Models**: Choose your preferred LLM approach
- **Open Source**: Full transparency in analysis methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
npm install --dev
pip install -r requirements-dev.txt

# Run tests
npm test
python -m pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Ollama** for local LLM infrastructure
- **NLTK** for natural language processing
- **React** and **Chart.js** for the dashboard
- **Sentence Transformers** for embeddings

## 📞 Support

- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Documentation**: [Wiki](link-to-wiki)

---

**Made with ❤️ for better customer insight analysis**

*Transform your customer feedback into actionable insights with the power of local LLMs!*
