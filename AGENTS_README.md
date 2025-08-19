# 🤖 Enhanced Agents for YouTube Video Summarizer

This module provides intelligent LangChain agents that enhance the YouTube video summarization process with advanced prompt optimization, content analysis, and insights generation.

## 🚀 Features

### **Core Agent Capabilities**
- **Intelligent Prompt Optimization**: Automatically improves summarization prompts based on content type
- **Content Structure Analysis**: Analyzes video transcripts for key topics, speakers, and structure
- **Content Insights Generation**: Provides actionable insights and recommendations
- **Multi-Model Integration**: Works with HuggingFace models for enhanced performance

### **Advanced Functionality**
- **Category-Aware Processing**: Optimizes prompts for different content types (Educational, Technical, News, etc.)
- **Structured Analysis**: Provides organized insights about video content
- **Error Handling**: Robust fallback mechanisms and error recovery
- **Logging & Monitoring**: Comprehensive logging for debugging and performance tracking

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- HuggingFace account and API token
- Internet connection for model downloads

### **Quick Setup**
1. **Run the setup script**:
   ```bash
   python setup_agents.py
   ```

2. **Manual installation** (if setup script fails):
   ```bash
   pip install python-dotenv langchain langchain-community langchain-core
   ```

3. **Configure environment**:
   - Ensure `hf.env` file exists with your HuggingFace API token
   - Format: `HUGGINGFACEHUB_API_TOKEN=your_token_here`

## 📁 File Structure

```
agents/
├── agents.py              # Basic agents functionality
├── enhanced_agents.py     # Full-featured agents with advanced capabilities
├── setup_agents.py        # Automated setup and installation
├── test_agents.py         # Testing and validation scripts
└── hf.env                 # HuggingFace API configuration
```

## 🎮 Usage

### **Basic Usage**

```python
from enhanced_agents import YouTubeSummarizerAgent

# Create agent
agent = YouTubeSummarizerAgent()

# Optimize a prompt
base_prompt = "Summarize this video about machine learning."
optimized_prompt = agent.optimize_summary_prompt(base_prompt, "Educational")

# Analyze content structure
analysis = agent.analyze_content_structure(transcript_text)

# Generate insights
insights = agent.generate_content_insights(transcript_text, "Technical")
```

### **Advanced Usage**

```python
# Create agent with custom configuration
agent = YouTubeSummarizerAgent()

# Optimize prompt for specific content type
prompts = {
    "Educational": "Explain the concepts in this tutorial",
    "Technical": "Provide step-by-step instructions",
    "News": "Summarize the key events and timeline"
}

for content_type, prompt in prompts.items():
    optimized = agent.optimize_summary_prompt(prompt, content_type)
    print(f"{content_type}: {optimized}")
```

## 🔧 Configuration

### **Environment Variables**
- `HUGGINGFACEHUB_API_TOKEN`: Your HuggingFace API token
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### **Model Configuration**
- **Default Model**: `google/flan-t5-large`
- **Temperature**: 0.5 (balanced creativity)
- **Max Length**: 512 tokens
- **Customizable**: Modify in agent initialization

## 📊 API Reference

### **YouTubeSummarizerAgent Class**

#### **Methods**

##### `__init__()`
Initializes the agent with HuggingFace LLM and LangChain agent.

##### `optimize_summary_prompt(base_prompt: str, content_category: str = "General") -> str`
Optimizes a base prompt for better summarization based on content category.

**Parameters:**
- `base_prompt`: The original prompt to optimize
- `content_category`: Content type (Educational, Technical, News, etc.)

**Returns:**
- Optimized prompt string

##### `analyze_content_structure(transcript: str) -> Dict[str, any]`
Analyzes the structure and key elements of video content.

**Parameters:**
- `transcript`: Video transcript text

**Returns:**
- Dictionary with analysis results and metadata

##### `generate_content_insights(transcript: str, content_category: str) -> str`
Generates insights and recommendations based on content analysis.

**Parameters:**
- `transcript`: Video transcript text
- `content_category`: Content type for context

**Returns:**
- Generated insights string

##### `test_connection() -> bool`
Tests the LLM connection with a simple prompt.

**Returns:**
- True if connection successful, False otherwise

## 🧪 Testing

### **Run Basic Tests**
```bash
python test_agents.py
```

### **Test Enhanced Functionality**
```bash
python enhanced_agents.py
```

### **Test Setup**
```bash
python setup_agents.py
```

## 🔍 Error Handling

### **Common Issues & Solutions**

#### **1. Import Errors**
```
❌ ImportError: No module named 'langchain'
```
**Solution**: Install required packages
```bash
pip install -r requirements.txt
```

#### **2. API Token Issues**
```
❌ HUGGINGFACEHUB_API_TOKEN not found
```
**Solution**: Check `hf.env` file and ensure token is valid

#### **3. Model Loading Errors**
```
❌ Error initializing HuggingFace LLM
```
**Solution**: Check internet connection and API token validity

#### **4. Memory Issues**
```
❌ CUDA out of memory
```
**Solution**: Use smaller models or reduce batch sizes

## 🚀 Performance Optimization

### **Best Practices**
1. **Batch Processing**: Process multiple prompts together when possible
2. **Model Selection**: Choose appropriate model size for your use case
3. **Caching**: Cache frequently used prompts and responses
4. **Error Recovery**: Implement retry mechanisms for failed requests

### **Resource Management**
- **Memory**: Monitor memory usage with large models
- **API Limits**: Respect HuggingFace API rate limits
- **Network**: Ensure stable internet connection for model downloads

## 🔮 Future Enhancements

### **Planned Features**
- **Multi-Model Support**: Integration with additional AI models
- **Custom Training**: Domain-specific model fine-tuning
- **Batch Processing**: Efficient handling of multiple requests
- **Advanced Analytics**: Detailed performance metrics and insights

### **Research Areas**
- **Prompt Engineering**: Advanced prompt optimization techniques
- **Content Understanding**: Better content categorization and analysis
- **Performance Optimization**: Faster processing and reduced latency

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Implement enhancements
4. Add tests and documentation
5. Submit a pull request

### **Code Standards**
- **Python**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for new features
- **Type Hints**: Full type annotation

## 📞 Support

### **Troubleshooting**
- Check the error logs for detailed information
- Verify environment configuration
- Test with simple examples first
- Check HuggingFace service status

### **Getting Help**
- **Issues**: Report bugs and request features
- **Documentation**: Comprehensive usage guides
- **Community**: Share solutions and best practices

## 📝 License

This module is part of the Enhanced YouTube Video Summarizer project and follows the same licensing terms.

---

**Made with ❤️ | Powered by LangChain & HuggingFace**

*Transform your YouTube content analysis with intelligent AI agents.*

