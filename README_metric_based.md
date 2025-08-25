# YouTube Video Summarizer - Metric-Based Version (No FLAN-T5)

This is a modified version of the YouTube Video Summarizer that performs evaluation using standard NLP metrics instead of FLAN-T5, while keeping other transformer models for summarization.

## Key Differences from Original Version

### ❌ Removed Components
- **FLAN-T5 Model**: No longer uses FLAN-T5 for prompt refinement
- **FLAN-T5 Dependencies**: Removed FLAN-T5 specific imports and functions

### ✅ Kept Components
- **BART Models**: Still uses BART for summarization (especially for news and entertainment content)
- **Pegasus Models**: Still uses Pegasus for educational and interview content
- **T5 Models**: Still uses T5 for technical/tutorial content
- **Transformer Pipeline**: Maintains the transformers library for model loading

### ✅ New Components
- **Metric-Based Evaluation**: Implements standard NLP evaluation metrics:
  - **BLEU Score**: Measures n-gram overlap between summary and original text
  - **ROUGE Scores**: ROUGE-1, ROUGE-2, and ROUGE-L for recall-oriented evaluation
  - **METEOR Score**: Metric for Evaluation of Translation with Explicit ORdering
  - **Compression Ratio**: Measures how much the text was condensed
  - **Readability Score**: Flesch Reading Ease score for text complexity
  - **Keyword Coverage**: Percentage of important keywords preserved

## Features

### 🎯 Core Functionality
- **YouTube URL Processing**: Supports multiple YouTube URL formats
- **Transcript Extraction**: Multi-strategy transcript extraction with fallbacks
- **Content Categorization**: Intelligent content type detection
- **Transformer Summarization**: Uses BART, Pegasus, and T5 models based on content type
- **Quality Evaluation**: Comprehensive metric-based assessment
- **Sentiment Analysis**: TextBlob-based sentiment scoring
- **Audio Generation**: Text-to-speech conversion
- **Modern UI**: Glassmorphism design with dark theme

### 📊 Evaluation Metrics
The app provides detailed quality assessment using:

1. **BLEU Score** (0-1): Measures n-gram precision
2. **ROUGE Scores** (0-1): Measures n-gram recall
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap  
   - ROUGE-L: Longest common subsequence
3. **METEOR Score** (0-1): Harmonic mean of precision and recall
4. **Compression Ratio**: Summary length / Original length
5. **Readability Score**: Normalized Flesch Reading Ease
6. **Keyword Coverage**: Important terms preserved in summary
7. **Overall Quality Score**: Weighted average of all metrics

### 🤖 Transformer Models Used
- **BART-large-CNN**: For news, entertainment, and review content
- **Pegasus-XSum**: For educational and interview/podcast content
- **T5-base**: For technical and tutorial content
- **Fallback**: BART-base if primary models fail

### 🎨 User Interface
- **Modern Design**: Glassmorphism UI with dark theme
- **Tabbed Interface**: Organized sections for summary, metrics, sentiment, and export
- **Real-time Feedback**: Progress indicators and status updates
- **Interactive Charts**: Plotly-based visualizations
- **Responsive Layout**: Works on different screen sizes

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_metric_based.txt
   ```

2. **Download NLTK Data** (if not already available):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('stopwords')
   ```

3. **Run the Application**:
   ```bash
   streamlit run Streamlit_without_flanT5.py
   ```

## Usage

1. **Enter YouTube URL**: Paste any YouTube video URL
2. **Configure Options**: Set detail level, format, and processing mode
3. **Generate Summary**: Click "Generate Summary" button
4. **Review Results**: 
   - View the generated summary
   - Check evaluation metrics
   - Analyze sentiment trends
   - Export results

## Technical Details

### Summarization Algorithm
1. **Preprocessing**: Clean and categorize content
2. **Model Selection**: Choose optimal transformer model based on content type
3. **Chunking**: Split text into model-appropriate chunks
4. **Generation**: Use transformer models to generate summaries
5. **Combination**: Merge chunk summaries into final result
6. **Formatting**: Apply user preferences

### Evaluation Process
1. **Text Preparation**: Tokenize and normalize text
2. **Metric Calculation**: Compute all evaluation metrics
3. **Score Normalization**: Convert to 0-1 scale
4. **Weighted Average**: Combine metrics into overall score
5. **Visualization**: Display results in charts

### Content Categories and Model Mapping
- **News/Current Events**: BART-large-CNN (fact-focused)
- **Educational**: Pegasus-XSum (comprehensive)
- **Technical/Tutorial**: T5-base (step-by-step)
- **Entertainment**: BART-large-CNN (engaging)
- **Review/Opinion**: BART-large-CNN (balanced)
- **Interview/Podcast**: Pegasus-XSum (insight-focused)

## Advantages

### 🚀 Performance
- **Faster Processing**: No FLAN-T5 prompt refinement overhead
- **Optimized Models**: Uses best model for each content type
- **Fallback System**: Graceful degradation if models fail

### 📈 Transparency
- **Interpretable Metrics**: Clear quality indicators
- **Reproducible Results**: Deterministic evaluation
- **Model Selection**: Transparent model choice based on content

### 💰 Cost-Effective
- **No FLAN-T5 Costs**: Removes FLAN-T5 specific overhead
- **Efficient Models**: Uses optimized transformer models
- **Open Source**: All dependencies are free

## Limitations

### 📝 Summarization Quality
- **No Prompt Refinement**: Lacks FLAN-T5's prompt optimization
- **Model Dependencies**: Requires transformer models to be available
- **Context Limits**: Limited by model context windows

### 🎯 Evaluation Scope
- **Surface Metrics**: Focuses on n-gram overlap
- **No Semantic Assessment**: Doesn't evaluate meaning preservation
- **Limited Creativity**: Cannot assess novel insights

## Comparison with Original Version

| Feature | Original (with FLAN-T5) | Modified (No FLAN-T5) |
|---------|------------------------|----------------------|
| **Summarization** | BART/Pegasus/T5 + FLAN-T5 refinement | BART/Pegasus/T5 only |
| **Evaluation** | Model-based | Metric-based |
| **Prompt Refinement** | FLAN-T5 | None |
| **Speed** | Slower (FLAN-T5 overhead) | Faster (no FLAN-T5) |
| **Memory** | High (FLAN-T5 + models) | Lower (no FLAN-T5) |
| **Quality** | Higher (with refinement) | Good (without refinement) |
| **Transparency** | Black box | Interpretable metrics |
| **Dependencies** | FLAN-T5 + transformers | Transformers only |

## Future Enhancements

### 🔮 Potential Improvements
- **Alternative Prompt Optimization**: Use other methods for prompt refinement
- **Advanced Metrics**: Add semantic similarity measures
- **Custom Training**: Fine-tune models on specific domains
- **Multi-language**: Support for non-English content
- **Real-time Processing**: Live video analysis

### 🛠️ Technical Roadmap
- **Prompt Engineering**: Manual prompt optimization
- **Model Ensembling**: Combine multiple model outputs
- **Semantic Evaluation**: Add BERTScore or similar metrics
- **Interactive Tuning**: Allow users to adjust model parameters

## Contributing

This version maintains the same code structure as the original but removes FLAN-T5 while keeping other transformer models. Contributions are welcome for:

- **New Evaluation Metrics**: Additional quality measures
- **Improved Model Selection**: Better content-to-model mapping
- **UI Enhancements**: Better user experience
- **Documentation**: Clearer explanations and examples

## License

This project maintains the same license as the original YouTube Video Summarizer.
