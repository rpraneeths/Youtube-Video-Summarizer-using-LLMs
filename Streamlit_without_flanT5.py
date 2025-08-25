import streamlit as st
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import torch
import re
import tempfile
from gtts import gTTS
from plotly.subplots import make_subplots 
import pygame
import time
import requests
from PIL import Image
from io import BytesIO
from textblob import TextBlob

# Import NLTK components with fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    def sent_tokenize(text):
        """Fallback sentence tokenization using simple period splitting"""
        return [s.strip() for s in text.split('.') if s.strip()]

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import urllib.parse
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import standard evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    EVALUATION_METRICS_AVAILABLE = True
except ImportError:
    EVALUATION_METRICS_AVAILABLE = False
    logger.warning("Some evaluation metrics not available. Install nltk and rouge-score for full functionality.")

# Enhanced data structures
class ValidationResult(NamedTuple):
    is_valid: bool
    video_id: Optional[str]
    error: Optional[str]
    url_type: Optional[str]

class VideoMetadata(NamedTuple):
    video_id: str
    title: str
    duration: str
    channel: str
    description: str
    view_count: int
    upload_date: str
    category: str

class ContentCategory(NamedTuple):
    category: str
    confidence: float
    subcategories: List[str]
    preprocessing_strategy: str

class EvaluationMetrics(NamedTuple):
    bleu_score: float
    rouge_scores: Dict[str, float]
    meteor_score: float
    compression_ratio: float
    readability_score: float
    keyword_coverage: float
    overall_score: float

# Enhanced URL validation patterns
YOUTUBE_URL_PATTERNS = [
    r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
]

def validate_youtube_url(url: str) -> ValidationResult:
    """
    Validate YouTube URL with comprehensive checks
    """
    try:
        # Clean and normalize URL
        url = url.strip()
        if not url:
            return ValidationResult(False, None, "URL cannot be empty", None)
        
        # Check if URL matches any YouTube pattern
        video_id = None
        url_type = None
        
        for pattern in YOUTUBE_URL_PATTERNS:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                url_type = pattern
                break
        
        if not video_id:
            return ValidationResult(False, None, "Invalid YouTube URL format", None)
        
        # Validate video ID format (YouTube IDs are 11 characters)
        if len(video_id) != 11:
            return ValidationResult(False, None, "Invalid video ID length", None)
        
        # Check if video is accessible (basic check)
        try:
            # Use YouTube oEmbed API to check video accessibility
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url, timeout=10)
            
            if response.status_code != 200:
                return ValidationResult(False, None, "Video not accessible or private", url_type)
            
            # Check for age restrictions and other issues
            video_info = response.json()
            if not video_info.get('title'):
                return ValidationResult(False, None, "Video information unavailable", url_type)
                
        except requests.RequestException as e:
            logger.warning(f"Could not verify video accessibility: {e}")
            # Continue with basic validation if API check fails
        
        return ValidationResult(True, video_id, None, url_type)
        
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return ValidationResult(False, None, f"Validation error: {str(e)}", None)

def extract_video_metadata(video_id: str) -> Optional[VideoMetadata]:
    """
    Extract comprehensive video metadata using YouTube Data API
    Falls back to oEmbed API if Data API is not available
    """
    try:
        # Try YouTube Data API first (requires API key)
        api_key = os.getenv('YOUTUBE_API_KEY')
        if api_key:
            try:
                data_api_url = f"https://www.googleapis.com/youtube/v3/videos"
                params = {
                    'part': 'snippet,contentDetails,statistics',
                    'id': video_id,
                    'key': api_key
                }
                response = requests.get(data_api_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('items'):
                        item = data['items'][0]
                        snippet = item['snippet']
                        content_details = item['contentDetails']
                        statistics = item['statistics']
                        
                        return VideoMetadata(
                            video_id=video_id,
                            title=snippet.get('title', 'Unknown Title'),
                            duration=content_details.get('duration', 'Unknown'),
                            channel=snippet.get('channelTitle', 'Unknown Channel'),
                            description=snippet.get('description', '')[:500] + '...' if snippet.get('description', '') else '',
                            view_count=int(statistics.get('viewCount', 0)),
                            upload_date=snippet.get('publishedAt', 'Unknown'),
                            category=snippet.get('categoryId', 'Unknown')
                        )
            except Exception as e:
                logger.warning(f"YouTube Data API failed: {e}")
        
        # Fallback to oEmbed API
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return VideoMetadata(
                video_id=video_id,
                title=data.get('title', 'Unknown Title'),
                duration='Unknown',
                channel=data.get('author_name', 'Unknown Channel'),
                description='',
                view_count=0,
                upload_date='Unknown',
                category='Unknown'
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Metadata extraction error: {e}")
        return None

def intelligent_transcript_extraction(video_id: str) -> Tuple[Optional[str], str]:
    """
    Multi-strategy transcript extraction:
    1. Primary: YouTube Transcript API (multiple languages)
    2. Fallback: Auto-generated captions
    3. Alternative: Audio extraction + Speech-to-Text (Whisper)
    4. Final: Inform user about manual transcript upload option
    """
    transcript = None
    method_used = ""
    
    try:
        # Strategy 1: YouTube Transcript API with multiple languages
        ytt_api = YouTubeTranscriptApi()
        
        # Try to get transcript in multiple languages
        languages_to_try = ['en', 'en-US', 'en-GB', 'auto']
        
        for lang in languages_to_try:
            try:
                transcript_data = ytt_api.fetch(video_id, languages=[lang])
                if transcript_data:
                    transcript = " ".join([line.text for line in transcript_data])
                    method_used = f"YouTube Transcript API ({lang})"
                    logger.info(f"Transcript extracted using {method_used}")
                    break
            except Exception as e:
                logger.debug(f"Failed to get transcript in {lang}: {e}")
                continue
        
        # Strategy 2: Try auto-generated captions if no manual transcript
        if not transcript:
            try:
                transcript_data = ytt_api.fetch(video_id, languages=['en'])
                if transcript_data:
                    transcript = " ".join([line.text for line in transcript_data])
                    method_used = "Auto-generated captions"
                    logger.info(f"Transcript extracted using {method_used}")
            except Exception as e:
                logger.debug(f"Auto-generated captions failed: {e}")
        
        # Strategy 3: Check if transcript is available but in different format
        if not transcript:
            try:
                # Try to get available transcripts
                transcript_list = ytt_api.list_transcripts(video_id)
                
                # Try manual transcripts first
                for transcript_obj in transcript_list:
                    if not transcript_obj.is_generated:
                        try:
                            transcript_data = transcript_obj.fetch()
                            transcript = " ".join([line['text'] for line in transcript_data])
                            method_used = f"Manual transcript ({transcript_obj.language})"
                            logger.info(f"Transcript extracted using {method_used}")
                            break
                        except Exception as e:
                            logger.debug(f"Failed to fetch manual transcript: {e}")
                            continue
                
                # If no manual transcript, try auto-generated
                if not transcript:
                    for transcript_obj in transcript_list:
                        if transcript_obj.is_generated:
                            try:
                                transcript_data = transcript_obj.fetch()
                                transcript = " ".join([line['text'] for line in transcript_data])
                                method_used = f"Auto-generated transcript ({transcript_obj.language})"
                                logger.info(f"Transcript extracted using {method_used}")
                                break
                            except Exception as e:
                                logger.debug(f"Failed to fetch auto-generated transcript: {e}")
                                continue
                                
            except Exception as e:
                logger.debug(f"Transcript list check failed: {e}")
        
        if transcript and transcript.strip():
            return transcript, method_used
        else:
            return None, "No transcript available"
            
    except Exception as e:
        logger.error(f"Transcript extraction error: {e}")
        return None, f"Extraction error: {str(e)}"

def _select_optimal_model(content_category: ContentCategory) -> str:
    """Select the optimal model based on content category"""
    model_mapping = {
        "News/Current Events": "facebook/bart-large-cnn",
        "Educational": "google/pegasus-xsum",
        "Technical/Tutorial": "t5-base",
        "Entertainment": "facebook/bart-large-cnn",
        "Review/Opinion": "facebook/bart-large-cnn",
        "Interview/Podcast": "google/pegasus-xsum"
    }
    
    return model_mapping.get(content_category.category, "facebook/bart-large-cnn")

def _get_model_context_window(model_name: str) -> int:
    """Get the context window size for different models"""
    context_windows = {
        "facebook/bart-large-cnn": 1024,
        "google/pegasus-xsum": 1024,
        "t5-base": 512,
        "facebook/bart-base": 1024
    }
    
    return context_windows.get(model_name, 1024)

def transformer_summarization(transcript: str, content_category: ContentCategory, user_preferences: Optional[Dict] = None) -> str:
    """
    Perform summarization using transformer models (BART, Pegasus, T5)
    """
    try:
        if not transcript:
            return "No transcript available for summarization."
        
        # Clean the transcript
        cleaned_transcript = clean_transcript(transcript)
        
        # Select optimal model based on content category
        model_name = _select_optimal_model(content_category)
        
        # Get model context window
        max_chunk_size = _get_model_context_window(model_name)
        
        # Split into chunks that fit within the model's context window
        chunks = []
        for i in range(0, len(cleaned_transcript), max_chunk_size):
            chunk = cleaned_transcript[i:i+max_chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        # Initialize summarizer
        try:
            summarizer = pipeline("summarization", model=model_name)
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to BART-base: {e}")
            summarizer = pipeline("summarization", model="facebook/bart-base")
        
        # Process each chunk
        summarized_text = []
        for i, chunk in enumerate(chunks):
            try:
                # Determine summary length based on user preferences
                if user_preferences and user_preferences.get('detail_level') == 'detailed':
                    max_length = 150
                    min_length = 80
                elif user_preferences and user_preferences.get('detail_level') == 'concise':
                    max_length = 80
                    min_length = 40
                else:
                    max_length = 120
                    min_length = 60
                
                # Generate summary for chunk
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summarized_text.append(summary[0]['summary_text'])
                
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {i+1}: {e}")
                # Fallback: use first few sentences of the chunk
                sentences = sent_tokenize(chunk) if NLTK_AVAILABLE else chunk.split('.')
                fallback_summary = ". ".join(sentences[:3])
                summarized_text.append(fallback_summary)
        
        # Combine summaries
        combined_summary = " ".join(summarized_text)
        
        # Format based on user preferences
        if user_preferences and user_preferences.get('format') == 'bullet_points':
            # Convert to bullet points
            sentences = sent_tokenize(combined_summary) if NLTK_AVAILABLE else combined_summary.split('.')
            combined_summary = "\n• " + "\n• ".join([s.strip() for s in sentences if s.strip()])
        
        return combined_summary
        
    except Exception as e:
        logger.error(f"Transformer summarization error: {e}")
        # Fallback to extractive summarization
        return extractive_summarization(transcript, content_category, user_preferences)

def categorize_content(transcript: str, metadata: Optional[VideoMetadata] = None) -> ContentCategory:
    """
    Intelligent content categorization using NLP techniques
    """
    try:
        # Initialize with default values
        category = "General"
        confidence = 0.5
        subcategories = []
        preprocessing_strategy = "standard"
        
        if not transcript:
            return ContentCategory(category, confidence, subcategories, preprocessing_strategy)
        
        # Convert to lowercase for analysis
        text_lower = transcript.lower()
        
        # Define category keywords and patterns
        category_patterns = {
            "News/Current Events": [
                r'\b(news|breaking|update|announcement|press|official|government|politics|election|vote)\b',
                r'\b(today|yesterday|this week|latest|recent|developing)\b',
                r'\b(report|coverage|investigation|exclusive|interview)\b'
            ],
            "Educational": [
                r'\b(learn|education|tutorial|lesson|course|class|lecture|study|research|academic)\b',
                r'\b(explain|understand|concept|theory|principle|method|technique)\b',
                r'\b(example|demonstration|step-by-step|how to|guide)\b'
            ],
            "Technical/Tutorial": [
                r'\b(programming|coding|software|development|code|script|function|api|database)\b',
                r'\b(install|setup|configure|deploy|build|compile|debug|test)\b',
                r'\b(technology|tech|computer|system|application|platform)\b'
            ],
            "Entertainment": [
                r'\b(entertainment|fun|comedy|humor|joke|laugh|amusing|hilarious)\b',
                r'\b(movie|film|show|series|episode|season|character|plot|story)\b',
                r'\b(music|song|artist|album|concert|performance|dance)\b'
            ],
            "Review/Opinion": [
                r'\b(review|opinion|thoughts|impression|experience|recommendation)\b',
                r'\b(pros|cons|advantages|disadvantages|benefits|drawbacks)\b',
                r'\b(rating|score|grade|evaluation|assessment|analysis)\b'
            ],
            "Interview/Podcast": [
                r'\b(interview|podcast|conversation|discussion|talk|chat|guest|host)\b',
                r'\b(question|answer|response|opinion|viewpoint|perspective)\b',
                r'\b(experience|background|story|journey|career|achievement)\b'
            ]
        }
        
        # Calculate category scores
        category_scores = {}
        for cat, patterns in category_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            category_scores[cat] = score
        
        # Find the category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
            
            if best_score > 0:
                category = best_category
                confidence = min(0.9, best_score / 10)  # Normalize confidence
                
                # Determine preprocessing strategy
                if category == "News/Current Events":
                    preprocessing_strategy = "news_focused"
                elif category == "Educational":
                    preprocessing_strategy = "educational_structured"
                elif category == "Technical/Tutorial":
                    preprocessing_strategy = "technical_stepwise"
                elif category == "Entertainment":
                    preprocessing_strategy = "entertainment_highlight"
                elif category == "Review/Opinion":
                    preprocessing_strategy = "review_balanced"
                elif category == "Interview/Podcast":
                    preprocessing_strategy = "interview_insights"
        
        # Add subcategories based on content analysis
        if "news" in text_lower or "politics" in text_lower:
            subcategories.append("Politics")
        if "technology" in text_lower or "tech" in text_lower:
            subcategories.append("Technology")
        if "science" in text_lower or "research" in text_lower:
            subcategories.append("Science")
        if "business" in text_lower or "finance" in text_lower:
            subcategories.append("Business")
        if "health" in text_lower or "medical" in text_lower:
            subcategories.append("Health")
        
        return ContentCategory(category, confidence, subcategories, preprocessing_strategy)
        
    except Exception as e:
        logger.error(f"Content categorization error: {e}")
        return ContentCategory("General", 0.5, [], "standard")

def content_aware_preprocessing(transcript: str, content_category: ContentCategory, metadata: Optional[VideoMetadata] = None) -> str:
    """
    Intelligent content categorization and preprocessing based on content type
    """
    try:
        if not transcript:
            return transcript
        
        processed_text = transcript
        
        # Apply category-specific preprocessing
        if content_category.preprocessing_strategy == "news_focused":
            processed_text = _preprocess_news_content(processed_text)
        elif content_category.preprocessing_strategy == "educational_structured":
            processed_text = _preprocess_educational_content(processed_text)
        elif content_category.preprocessing_strategy == "technical_stepwise":
            processed_text = _preprocess_technical_content(processed_text)
        elif content_category.preprocessing_strategy == "entertainment_highlight":
            processed_text = _preprocess_entertainment_content(processed_text)
        elif content_category.preprocessing_strategy == "review_balanced":
            processed_text = _preprocess_review_content(processed_text)
        elif content_category.preprocessing_strategy == "interview_insights":
            processed_text = _preprocess_interview_content(processed_text)
        
        # Apply general preprocessing
        processed_text = _apply_general_preprocessing(processed_text)
        
        return processed_text
        
    except Exception as e:
        logger.error(f"Content preprocessing error: {e}")
        return transcript

def _preprocess_news_content(text: str) -> str:
    """Preprocess news content focusing on facts and key information"""
    # Remove common news filler phrases
    filler_phrases = [
        r'\b(breaking news|developing story|stay tuned|more to come)\b',
        r'\b(according to|reports say|sources say|officials say)\b',
        r'\b(meanwhile|additionally|furthermore|moreover)\b'
    ]
    
    for phrase in filler_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    
    return text

def _preprocess_educational_content(text: str) -> str:
    """Preprocess educational content for better structure"""
    # Identify and emphasize key concepts
    text = re.sub(r'\b(important|key|main|primary|essential)\b', r'**\1**', text, flags=re.IGNORECASE)
    
    # Clean up repetitive educational phrases
    text = re.sub(r'\b(as you can see|as we discussed|remember that)\b', '', text, flags=re.IGNORECASE)
    
    return text

def _preprocess_technical_content(text: str) -> str:
    """Preprocess technical content for step-by-step clarity"""
    # Identify step indicators
    text = re.sub(r'\b(step \d+|first|second|third|next|then|finally)\b', r'**\1**', text, flags=re.IGNORECASE)
    
    # Emphasize technical terms
    text = re.sub(r'\b(install|configure|setup|deploy|build|run|execute)\b', r'**\1**', text, flags=re.IGNORECASE)
    
    return text

def _preprocess_entertainment_content(text: str) -> str:
    """Preprocess entertainment content to highlight key moments"""
    # Remove excessive filler words
    filler_words = [
        r'\b(um|uh|like|you know|basically|actually|literally)\b',
        r'\b(so|well|right|okay|yeah|wow|amazing)\b'
    ]
    
    for word in filler_words:
        text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    
    return text

def _preprocess_review_content(text: str) -> str:
    """Preprocess review content for balanced analysis"""
    # Emphasize pros and cons
    text = re.sub(r'\b(pros?|advantages?|benefits?|strengths?)\b', r'**\1**', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(cons?|disadvantages?|drawbacks?|weaknesses?)\b', r'**\1**', text, flags=re.IGNORECASE)
    
    return text

def _preprocess_interview_content(text: str) -> str:
    """Preprocess interview content for key insights"""
    # Identify speaker transitions
    text = re.sub(r'\b(interviewer|host|guest|speaker|question|answer)\b', r'**\1**', text, flags=re.IGNORECASE)
    
    # Clean up interview filler
    text = re.sub(r'\b(that\'s a great question|interesting point|good question)\b', '', text, flags=re.IGNORECASE)
    
    return text

def _apply_general_preprocessing(text: str) -> str:
    """Apply general preprocessing to all content types"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common filler words
    common_fillers = [
        r'\b(um|uh|like|you know|basically|actually|literally)\b',
        r'\b(so|well|right|okay|yeah|wow|amazing)\b'
    ]
    
    for filler in common_fillers:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)
    
    # Normalize punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Clean up timestamps if present
    text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
    
    return text.strip()

def clean_transcript(text: str) -> str:
    """
    Comprehensive transcript cleaning function to remove noise, repeated phrases, and irrelevant content.
    This function prepares clean text for the summarization pipeline.
    """
    import re
    
    if not text or len(text.strip()) < 10:
        return text
    
    # Step 1: Remove stage cues and audience reactions
    stage_cues = [
        r'\([^)]*(?:cheers?|applause|laughter|music|sounds?|noise)[^)]*\)',
        r'\[[^\]]*(?:cheers?|applause|laughter|music|sounds?|noise)[^\]]*\]',
        r'\b(cheers?|applause|laughter|music|sounds?|noise)\b'
    ]
    
    for pattern in stage_cues:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 2: Remove repeated words and phrases
    text = re.sub(r'\b(\w+(?:\s+\w+){0,3})\s*,\s*\1\b', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+(?:\s+\w+){0,3})\s+and\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    
    # Step 3: Remove specific names of journalists, filmmakers, and unrelated people
    irrelevant_names = [
        r'\b(?:Ahmed\s+Rashid|Al\s+Sharpton|Steve\s+Kroft|John\s+Doe|Jane\s+Smith)\b',
        r'\b(?:journalist|reporter|correspondent|anchor|host|presenter)\s+\w+\s+\w+\b',
        r'\b(?:filmmaker|director|producer|writer)\s+\w+\s+\w+\b',
        r'\b(?:columnist|editor|author|contributor)\s+\w+\s+\w+\b'
    ]
    
    for pattern in irrelevant_names:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 4: Remove filler phrases and non-essential parts
    filler_phrases = [
        r'\b(?:letters?\s+from|columnist|series\s+of|part\s+of|episode\s+\d+)\b',
        r'\b(?:as\s+you\s+know|as\s+we\s+discussed|remember\s+that)\b',
        r'\b(?:that\'s\s+a\s+great\s+question|interesting\s+point|good\s+question)\b',
        r'\b(?:um|uh|like|you\s+know|basically|actually|literally)\b',
        r'\b(?:so|well|right|okay|yeah|wow|amazing)\b'
    ]
    
    for pattern in filler_phrases:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Step 5: Clean up punctuation artifacts and extra spaces
    text = re.sub(r'[.!?]{2,}', '.', text)
    text = re.sub(r'[,;]{2,}', ',', text)
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Step 6: Remove empty parentheses and brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # Step 7: Final cleanup - normalize spacing around punctuation
    text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Step 8: Remove sentences that are too short or just noise
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Keep sentences that are meaningful (at least 10 characters and not just noise)
        if len(sentence) >= 10 and not re.match(r'^[^a-zA-Z]*$', sentence):
            cleaned_sentences.append(sentence)
    
    # Rejoin cleaned sentences
    result = ' '.join(cleaned_sentences)
    
    return result

def extractive_summarization(transcript: str, content_category: ContentCategory, user_preferences: Optional[Dict] = None) -> str:
    """
    Perform extractive summarization using TF-IDF and sentence ranking
    """
    try:
        if not transcript:
            return "No transcript available for summarization."
        
        # Clean the transcript
        cleaned_transcript = clean_transcript(transcript)
        
        # Split into sentences
        try:
            sentences = sent_tokenize(cleaned_transcript)
        except LookupError:
            # Fallback: simple sentence splitting by periods
            sentences = [s.strip() for s in cleaned_transcript.split('.') if s.strip()]
        
        if len(sentences) <= 3:
            return cleaned_transcript
        
        # Calculate sentence importance using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Get sentence scores
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # Determine number of sentences to include based on user preferences
        if user_preferences and user_preferences.get('detail_level') == 'detailed':
            num_sentences = min(8, len(sentences))
        elif user_preferences and user_preferences.get('detail_level') == 'concise':
            num_sentences = min(3, len(sentences))
        else:
            num_sentences = min(5, len(sentences))
        
        # Select top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_sentences = [sentences[i] for i in sorted(top_indices)]
        
        # Join sentences
        summary = " ".join(top_sentences)
        
        # Format based on user preferences
        if user_preferences and user_preferences.get('format') == 'bullet_points':
            summary = "\n• " + "\n• ".join(top_sentences)
        
        return summary
        
    except Exception as e:
        logger.error(f"Extractive summarization error: {e}")
        return f"Summarization error: {str(e)}"

def evaluate_summary_quality(original_text: str, summary: str, content_category: ContentCategory) -> EvaluationMetrics:
    """
    Evaluate summary quality using standard metrics
    """
    try:
        if not EVALUATION_METRICS_AVAILABLE:
            # Fallback evaluation without external metrics
            compression_ratio = len(summary) / len(original_text) if len(original_text) > 0 else 0.0
            return EvaluationMetrics(0.0, {}, 0.0, compression_ratio, 0.5, 0.5, 0.5)
        
        # Split texts into sentences for evaluation
        try:
            original_sentences = sent_tokenize(original_text)
            summary_sentences = sent_tokenize(summary)
        except LookupError:
            # Fallback: simple sentence splitting
            original_sentences = [s.strip() for s in original_text.split('.') if s.strip()]
            summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
        
        # Calculate BLEU score
        try:
            # Tokenize sentences for BLEU calculation
            original_tokens = [sentence.lower().split() for sentence in original_sentences]
            summary_tokens = [sentence.lower().split() for sentence in summary_sentences]
            
            # Calculate BLEU score for each summary sentence against all original sentences
            bleu_scores = []
            smoothing = SmoothingFunction().method1
            
            for summary_sent in summary_tokens:
                if summary_sent:  # Skip empty sentences
                    bleu_score = sentence_bleu(original_tokens, summary_sent, smoothing_function=smoothing)
                    bleu_scores.append(bleu_score)
            
            avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            avg_bleu = 0.0
        
        # Calculate ROUGE scores
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(original_text, summary)
            
            rouge_dict = {
                'rouge1_f1': rouge_scores['rouge1'].fmeasure,
                'rouge2_f1': rouge_scores['rouge2'].fmeasure,
                'rougeL_f1': rouge_scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            rouge_dict = {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
        
        # Calculate METEOR score
        try:
            # For METEOR, we need to provide reference and hypothesis as lists of sentences
            meteor = meteor_score([original_sentences], summary_sentences)
        except Exception as e:
            logger.warning(f"METEOR calculation failed: {e}")
            meteor = 0.0
        
        # Calculate compression ratio
        compression_ratio = len(summary) / len(original_text) if len(original_text) > 0 else 0.0
        
        # Calculate readability score (Flesch Reading Ease)
        try:
            def calculate_flesch_reading_ease(text):
                sentences = len(sent_tokenize(text))
                words = len(text.split())
                syllables = sum(1 for word in text.lower().split() 
                              for char in word if char in 'aeiou')
                
                if sentences == 0 or words == 0:
                    return 0.0
                
                return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
            
            readability_score = calculate_flesch_reading_ease(summary)
            # Normalize to 0-1 range
            readability_score = max(0.0, min(1.0, readability_score / 100.0))
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            readability_score = 0.5
        
        # Calculate keyword coverage
        try:
            # Extract keywords from original text using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
            original_tfidf = vectorizer.fit_transform([original_text])
            summary_tfidf = vectorizer.transform([summary])
            
            # Get feature names (keywords)
            keywords = vectorizer.get_feature_names_out()
            
            # Calculate coverage
            original_keywords = set(keywords[original_tfidf.toarray()[0] > 0])
            summary_keywords = set(keywords[summary_tfidf.toarray()[0] > 0])
            
            if original_keywords:
                keyword_coverage = len(summary_keywords.intersection(original_keywords)) / len(original_keywords)
            else:
                keyword_coverage = 0.0
        except Exception as e:
            logger.warning(f"Keyword coverage calculation failed: {e}")
            keyword_coverage = 0.0
        
        # Calculate overall score (weighted average)
        weights = {
            'bleu': 0.2,
            'rouge': 0.3,
            'meteor': 0.2,
            'compression': 0.1,
            'readability': 0.1,
            'keyword_coverage': 0.1
        }
        
        avg_rouge = np.mean(list(rouge_dict.values()))
        
        # Normalize compression ratio (prefer 0.1-0.3 range)
        normalized_compression = 1.0 - abs(compression_ratio - 0.2) / 0.2
        normalized_compression = max(0.0, min(1.0, normalized_compression))
        
        overall_score = (
            weights['bleu'] * avg_bleu +
            weights['rouge'] * avg_rouge +
            weights['meteor'] * meteor +
            weights['compression'] * normalized_compression +
            weights['readability'] * readability_score +
            weights['keyword_coverage'] * keyword_coverage
        )
        
        return EvaluationMetrics(
            bleu_score=avg_bleu,
            rouge_scores=rouge_dict,
            meteor_score=meteor,
            compression_ratio=compression_ratio,
            readability_score=readability_score,
            keyword_coverage=keyword_coverage,
            overall_score=overall_score
        )
        
    except Exception as e:
        logger.error(f"Summary evaluation error: {e}")
        return EvaluationMetrics(0.0, {}, 0.0, 0.0, 0.0, 0.0, 0.0)

def enhanced_summarization_pipeline(transcript: str, content_category: ContentCategory, user_preferences: Optional[Dict] = None) -> Tuple[str, EvaluationMetrics]:
    """
    Enhanced summarization pipeline using transformer models and metric-based evaluation
    """
    try:
        if not transcript:
            return "No transcript available for summarization.", EvaluationMetrics(0.0, {}, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Stage 1: Content Analysis and Preprocessing
        processed_transcript = content_aware_preprocessing(transcript, content_category)
        
        # Stage 2: Pre-summarization cleanup
        with st.status("Preprocessing transcript for cleaner summarization...", expanded=True) as status:
            cleaned_transcript = clean_transcript(processed_transcript)
            logger.info(f"After transcript cleaning: {len(cleaned_transcript)} characters")
            
            # Limit input text length to 2000 characters for more focused summarization
            if len(cleaned_transcript) > 2000:
                cleaned_transcript = cleaned_transcript[:2000]
                logger.info(f"Transcript truncated to {len(cleaned_transcript)} characters for focused summarization")
            
            status.update(label="Transcript preprocessed successfully!", state="complete")
        
        # Stage 3: Transformer-based Summarization
        with st.status("Generating summary using transformer models...", expanded=True) as status:
            try:
                summary = transformer_summarization(cleaned_transcript, content_category, user_preferences)
                status.update(label="Summary generated successfully!", state="complete")
            except Exception as e:
                logger.warning(f"Transformer summarization failed: {e}")
                st.warning("Transformer models failed, falling back to extractive summarization...")
                summary = extractive_summarization(cleaned_transcript, content_category, user_preferences)
                status.update(label="Extractive summary generated!", state="complete")
        
        # Stage 4: Evaluate Summary Quality
        with st.status("Evaluating summary quality...", expanded=True) as status:
            evaluation_metrics = evaluate_summary_quality(cleaned_transcript, summary, content_category)
            status.update(label="Evaluation complete!", state="complete")
        
        # Stage 5: Format summary based on category
        formatted_summary = _format_summary_by_category(summary, content_category)
        
        return formatted_summary, evaluation_metrics
        
    except Exception as e:
        logger.error(f"Enhanced summarization pipeline error: {e}")
        error_summary = f"Summarization error: {str(e)}"
        error_metrics = EvaluationMetrics(0.0, {}, 0.0, 0.0, 0.0, 0.0, 0.0)
        return error_summary, error_metrics

def _format_summary_by_category(summary: str, content_category: ContentCategory) -> str:
    """Format summary based on content category"""
    if content_category.preprocessing_strategy == "news_focused":
        return f"📰 **News Summary**\n\n{summary}"
    elif content_category.preprocessing_strategy == "educational_structured":
        return f"📚 **Educational Summary**\n\n{summary}"
    elif content_category.preprocessing_strategy == "technical_stepwise":
        return f"⚙️ **Technical Summary**\n\n{summary}"
    elif content_category.preprocessing_strategy == "entertainment_highlight":
        return f"🎬 **Entertainment Summary**\n\n{summary}"
    elif content_category.preprocessing_strategy == "review_balanced":
        return f"⭐ **Review Summary**\n\n{summary}"
    elif content_category.preprocessing_strategy == "interview_insights":
        return f"🎤 **Interview Summary**\n\n{summary}"
    else:
        return f"📋 **Summary**\n\n{summary}"

def handle_transcript_unavailable(video_id: str, metadata: Optional[VideoMetadata] = None) -> Dict:
    """
    Intelligent fallback when transcript is unavailable
    """
    fallback_options = {
        'auto_captions_available': False,
        'audio_extraction_possible': False,
        'manual_upload_suggested': True,
        'alternative_methods': [],
        'error_message': "No transcript available for this video."
    }
    
    try:
        # Check for auto-generated captions
        ytt_api = YouTubeTranscriptApi()
        try:
            transcript_list = ytt_api.list_transcripts(video_id)
            for transcript_obj in transcript_list:
                if transcript_obj.is_generated:
                    fallback_options['auto_captions_available'] = True
                    fallback_options['alternative_methods'].append("Auto-generated captions available")
                    break
        except Exception as e:
            logger.debug(f"Could not check auto-captions: {e}")
        
        # Suggest manual transcript upload
        fallback_options['alternative_methods'].extend([
            "Manual transcript upload",
            "Video description analysis",
            "Comments analysis (if enabled)",
            "Audio analysis (if legally permissible)"
        ])
        
        # Update error message with helpful information
        if fallback_options['auto_captions_available']:
            fallback_options['error_message'] = "Manual transcript not available, but auto-generated captions may be accessible."
        else:
            fallback_options['error_message'] = "No transcript available. Consider uploading a manual transcript or using alternative analysis methods."
        
        return fallback_options
        
    except Exception as e:
        logger.error(f"Fallback analysis error: {e}")
        return fallback_options

# Set page config
st.set_page_config(
    page_title="YouTube Video Summarizer (Metric-Based)",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize pygame for audio
pygame.mixer.init()

# File paths for persistence
FAVORITES_FILE = "favorites.json"
SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.json"

def load_data(file_path, default_data):
    """Load data from JSON file or return default if file doesn't exist"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
    return default_data

def save_data(file_path, data):
    """Save data to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Error saving {file_path}: {str(e)}")

def init_session_states():
    """Initialize session states with loaded data"""
    # Load saved data
    favorites = load_data(FAVORITES_FILE, [])
    settings = load_data(SETTINGS_FILE, {
        'theme': 'Dark',
        'font_size': 'Medium',
        'language': 'en',
        'auto_play': False
    })
    history = load_data(HISTORY_FILE, [])
    
    # Initialize session states
    if 'favorites' not in st.session_state:
        st.session_state.favorites = favorites
    if 'settings' not in st.session_state:
        st.session_state.settings = settings
    if 'history' not in st.session_state:
        st.session_state.history = history
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False
    if 'execution_time' not in st.session_state:
        st.session_state.execution_time = 0
    if 'sentiment_fig' not in st.session_state:
        st.session_state.sentiment_fig = None
    if 'avg_polarity' not in st.session_state:
        st.session_state.avg_polarity = 0
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "analyzer"
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'evaluation_metrics' not in st.session_state:
        st.session_state.evaluation_metrics = None

def set_modern_style(dark_mode):
    """Set modern glassmorphism style for the app"""
    if isinstance(dark_mode, str):
        dark_mode = dark_mode.lower() == 'true' or dark_mode.lower() == 'dark'
    st.markdown("""
        <style>
        /* Base styles */
        .stApp {
            background: linear-gradient(135deg, 
                rgba(37, 38, 89, 0.95) 0%,
                rgba(74, 21, 131, 0.95) 35%,
                rgba(37, 38, 89, 0.95) 100%);
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Glass card styles */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .glass-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        /* Metric card styles */
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        }
        
        /* Text styles */
        .title-text {
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            font-size: 2.5rem;
            background: linear-gradient(135deg, #ffffff, #e0e0e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .summary-text {
            line-height: 1.8;
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.9);
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            transition: all 0.3s ease;
        }
        
        .summary-text:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        /* Button styles */
        .stButton > button {
            background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
            border: none;
            border-radius: 16px;
            color: white;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 24px rgba(110, 72, 170, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

def text_to_speech(text, filename="summary.mp3"):
    """Convert text to speech"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts = gTTS(text=text, lang='en')
            tts.save(tmpfile.name)
            return tmpfile.name
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def play_audio(file_path):
    """Play audio file"""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        st.session_state.audio_playing = True
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")
        st.session_state.audio_playing = False

def stop_audio():
    """Stop audio playback"""
    try:
        pygame.mixer.music.stop()
        st.session_state.audio_playing = False
    except:
        pass

def cleanup_audio():
    """Clean up audio files"""
    if st.session_state.get('audio_file') and os.path.exists(st.session_state.audio_file):
        try:
            if st.session_state.audio_playing:
                stop_audio()
            os.remove(st.session_state.audio_file)
            st.session_state.audio_file = None
        except:
            pass

def enhanced_process_video(url, user_preferences=None):
    """
    Enhanced video processing pipeline with metric-based evaluation
    """
    try:
        cleanup_audio()
        process_start_time = time.time()
        
        # Stage 1: Enhanced URL Validation
        with st.status("Validating YouTube URL...", expanded=True) as status:
            validation_result = validate_youtube_url(url)
            if not validation_result.is_valid:
                st.error(f"URL validation failed: {validation_result.error}")
                return False
            status.update(label="URL validated successfully!", state="complete")
        
        # Stage 2: Extract Video Metadata
        with st.status("Extracting video metadata...", expanded=True) as status:
            metadata = extract_video_metadata(validation_result.video_id)
            if metadata:
                st.info(f"📺 **{metadata.title}** by **{metadata.channel}**")
            status.update(label="Metadata extracted!", state="complete")
        
        # Stage 3: Intelligent Transcript Extraction
        with st.status("Extracting transcript using multiple strategies...", expanded=True) as status:
            transcript, method_used = intelligent_transcript_extraction(validation_result.video_id)
            
            if not transcript:
                # Handle transcript unavailability intelligently
                fallback_options = handle_transcript_unavailable(validation_result.video_id, metadata)
                st.error(fallback_options['error_message'])
                
                if fallback_options['alternative_methods']:
                    st.info("**Alternative methods available:**")
                    for method in fallback_options['alternative_methods']:
                        st.write(f"• {method}")
                
                return False
            
            st.session_state.transcript = transcript
            st.success(f"✅ Transcript extracted using: {method_used}")
            status.update(label="Transcript extracted successfully!", state="complete")
        
        # Stage 4: Content Categorization
        with st.status("Analyzing content type...", expanded=True) as status:
            content_category = categorize_content(transcript, metadata)
            st.info(f"🎯 **Content Category:** {content_category.category} (Confidence: {content_category.confidence:.1%})")
            
            if content_category.subcategories:
                st.write(f"**Subcategories:** {', '.join(content_category.subcategories)}")
            
            status.update(label="Content categorized!", state="complete")
        
        # Stage 5: Enhanced Summarization with Metric-Based Evaluation
        with st.status("Generating summary with metric-based evaluation...", expanded=True) as status:
            try:
                summary, evaluation_metrics = enhanced_summarization_pipeline(transcript, content_category, user_preferences)
                st.session_state.summary = summary
                st.session_state.evaluation_metrics = evaluation_metrics
                status.update(label="Summary generated and evaluated successfully!", state="complete")
            except Exception as e:
                st.error(f"Enhanced summarization failed: {str(e)}")
                return False
        
        # Stage 6: Generate Audio Summary
        with st.status("Generating audio summary...", expanded=True) as status:
            audio_file = text_to_speech(st.session_state.summary)
            if audio_file:
                st.session_state.audio_file = audio_file
                status.update(label="Audio summary generated!", state="complete")
        
        # Stage 7: Sentiment Analysis
        with st.status("Analyzing sentiment...", expanded=True) as status:
            analyze_sentiment(url)
            status.update(label="Sentiment analysis complete!", state="complete")
        
        # Calculate and store execution time
        st.session_state.execution_time = time.time() - process_start_time
        st.session_state.analysis_complete = True
        
        # Store enhanced metadata in session state
        st.session_state.video_metadata = metadata
        st.session_state.content_category = content_category
        st.session_state.transcript_method = method_used
        
        # Add to history with enhanced information
        video_title = metadata.title if metadata else f"Video {validation_result.video_id[:8]}..."
        add_to_history(
            validation_result.video_id,
            video_title,
            st.session_state.summary,
            st.session_state.avg_polarity
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced video processing error: {e}")
        st.error(f"Processing error: {str(e)}")
        return False

def analyze_sentiment(url):
    """Analyze sentiment of video transcript"""
    try:
        video_id = url.split("watch?v=")[-1].split('&')[0]
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
        except Exception as e:
            from youtube_transcript_api._errors import NoTranscriptFound
            if isinstance(e, NoTranscriptFound):
                st.error("No transcript available for this video.")
                return
            else:
                st.error(f"Error fetching transcript: {str(e)}")
                return
        
        # Calculate video duration in minutes
        try:
            if hasattr(transcript[-1], 'start'):
                duration_sec = transcript[-1].start + transcript[-1].duration
            else:
                duration_sec = transcript[-1]['start'] + transcript[-1]['duration']
        except (AttributeError, KeyError, IndexError):
            duration_sec = len(transcript) * 10
        duration_min = duration_sec / 60
        
        # Determine appropriate interval
        interval = get_sentiment_interval(duration_min)
        
        # Group transcript by time intervals
        sentiments = []
        time_labels = []
        current_text = ""
        current_start = 0
        
        for segment in transcript:
            try:
                if hasattr(segment, 'start'):
                    segment_min = segment.start / 60
                    segment_text = segment.text
                else:
                    segment_min = segment['start'] / 60
                    segment_text = segment['text']
            except (AttributeError, KeyError):
                continue
            
            if segment_min - current_start >= interval:
                if current_text:
                    try:
                        blob = TextBlob(current_text)
                        sentiments.append(blob.sentiment.polarity)
                        time_labels.append(f"{int(current_start)}-{int(current_start)+interval}min")
                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for segment: {e}")
                        sentiments.append(0.0)
                        time_labels.append(f"{int(current_start)}-{int(current_start)+interval}min")
                
                current_text = segment_text
                current_start = interval * (segment_min // interval)
            else:
                current_text += " " + segment_text
        
        # Analyze the last segment
        if current_text:
            try:
                blob = TextBlob(current_text)
                sentiments.append(blob.sentiment.polarity)
                time_labels.append(f"{int(current_start)}-{int(current_start)+interval}min")
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for last segment: {e}")
                sentiments.append(0.0)
                time_labels.append(f"{int(current_start)}-{int(current_start)+interval}min")
        
        # Store average polarity
        st.session_state.avg_polarity = np.mean(sentiments)
        
        # Create and store the figure
        st.session_state.sentiment_fig = create_modern_sentiment_analysis(
            sentiments, 
            time_labels, 
            duration_min,
            interval
        )
        
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        st.session_state.sentiment_fig = None
        st.session_state.avg_polarity = 0

def get_sentiment_interval(duration_min):
    """Determine sentiment analysis interval based on video duration"""
    if duration_min <= 5:
        return 1
    elif duration_min <= 10:
        return 2
    elif duration_min <= 20:
        return 3
    elif duration_min <= 30:
        return 4
    elif duration_min <= 60:
        return 5
    else:
        return 10

def create_modern_sentiment_analysis(sentiments, time_labels, duration_min, interval):
    """Create a modern sentiment analysis visualization"""
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=sentiments,
            mode='lines+markers',
            line=dict(
                color='rgba(110, 72, 170, 0.8)',
                width=3,
                shape='spline'
            ),
            name="Sentiment",
            marker=dict(
                size=8,
                color=sentiments,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Sentiment",
                        font=dict(color='#ffffff')
                    ),
                    tickfont=dict(color='#ffffff')
                )
            )
        )
    )
    
    # Add area fill
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=sentiments,
            fill='tozeroy',
            mode='none',
            fillcolor='rgba(110, 72, 170, 0.2)',
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Sentiment Analysis (Every {interval} min)",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(
                text="Video Timeline (minutes)",
                font=dict(color='#ffffff')
            ),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            title=dict(
                text="Sentiment Score",
                font=dict(color='#ffffff')
            ),
            range=[-1, 1],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            tickfont=dict(color='#ffffff')
        ),
        height=500,
        margin=dict(t=80, l=50, r=50, b=50)
    )
    
    # Add horizontal line at zero
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.5)"
    )
    
    return fig

def add_to_history(video_id, title, summary, sentiment):
    """Add a video to history with timestamp"""
    history_entry = {
        'video_id': video_id,
        'title': title or f"Video {video_id[:8]}...",
        'summary': summary or "",
        'sentiment': sentiment or 0,
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.history.append(history_entry)
    save_data(HISTORY_FILE, st.session_state.history)

def toggle_favorite(video_id, title, summary):
    """Toggle favorite status of a video"""
    favorite = {
        'video_id': video_id,
        'title': title,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check if already in favorites
    for fav in st.session_state.favorites:
        if fav['video_id'] == video_id:
            st.session_state.favorites.remove(fav)
            save_data(FAVORITES_FILE, st.session_state.favorites)
            return False
    
    # Add to favorites if not present
    st.session_state.favorites.append(favorite)
    save_data(FAVORITES_FILE, st.session_state.favorites)
    return True

def display_modern_results():
    """Display results in a modern UI"""
    if not st.session_state.analysis_complete:
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    tabs = st.tabs(["📝 Summary", "📊 Evaluation Metrics", "📈 Sentiment Analysis", "💾 Export"])
    
    with tabs[0]:  # Summary Tab
        st.markdown("""
            <h3 style='margin-bottom: 1.5rem;'>Video Summary</h3>
            """, unsafe_allow_html=True)
        
        st.markdown(
            f'<div class="glass-card summary-text">{st.session_state.summary}</div>', 
            unsafe_allow_html=True
        )
        
        if st.session_state.audio_file:
            st.markdown("<h3 style='margin: 2rem 0 1rem;'>🎧 Audio Summary</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("▶️ Play", key="play_button"):
                    play_audio(st.session_state.audio_file)
            with col2:
                if st.button("⏹ Stop", key="stop_button"):
                    stop_audio()
                    
            # Add to favorites button
            if st.button("⭐ Add to Favorites", type="secondary"):
                video_id = st.session_state.url.split("watch?v=")[-1].split('&')[0]
                if toggle_favorite(video_id, f"Video {video_id[:8]}...", st.session_state.summary):
                    st.success("Added to favorites!")
                else:
                    st.success("Removed from favorites!")
    
    with tabs[1]:  # Evaluation Metrics Tab
        st.markdown("""
            <h3 style='margin-bottom: 1.5rem;'>Summary Quality Evaluation</h3>
            """, unsafe_allow_html=True)
        
        if st.session_state.evaluation_metrics:
            metrics = st.session_state.evaluation_metrics
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Overall Quality Score",
                    f"{metrics.overall_score:.3f}",
                    help="Weighted average of all metrics (0-1 scale)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "BLEU Score",
                    f"{metrics.bleu_score:.3f}",
                    help="Bilingual Evaluation Understudy score"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "METEOR Score",
                    f"{metrics.meteor_score:.3f}",
                    help="Metric for Evaluation of Translation with Explicit ORdering"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Compression Ratio",
                    f"{metrics.compression_ratio:.1%}",
                    help="Summary length relative to original text"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Readability Score",
                    f"{metrics.readability_score:.3f}",
                    help="Flesch Reading Ease score (normalized)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Keyword Coverage",
                    f"{metrics.keyword_coverage:.1%}",
                    help="Percentage of important keywords preserved"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ROUGE Scores
            if metrics.rouge_scores:
                st.markdown("### ROUGE Scores")
                rouge_col1, rouge_col2, rouge_col3 = st.columns(3)
                
                with rouge_col1:
                    st.metric("ROUGE-1 F1", f"{metrics.rouge_scores.get('rouge1_f1', 0):.3f}")
                with rouge_col2:
                    st.metric("ROUGE-2 F1", f"{metrics.rouge_scores.get('rouge2_f1', 0):.3f}")
                with rouge_col3:
                    st.metric("ROUGE-L F1", f"{metrics.rouge_scores.get('rougeL_f1', 0):.3f}")
        else:
            st.info("Evaluation metrics not available")
    
    with tabs[2]:  # Sentiment Analysis Tab
        st.markdown("### 📊 Sentiment Analysis")
        if st.session_state.sentiment_fig is not None:
            st.plotly_chart(st.session_state.sentiment_fig, use_container_width=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.metric(
                "Overall Sentiment Score",
                f"{st.session_state.avg_polarity:.2f}",
                delta=None,
                help="Range: -1 (most negative) to +1 (most positive)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        elif hasattr(st.session_state, 'avg_polarity') and st.session_state.avg_polarity != 0:
            st.info("Sentiment analysis completed. Chart will be displayed after analysis.")
    
    with tabs[3]:  # Export Tab
        st.markdown("""
            <h3 style='margin-bottom: 1.5rem;'>Export Options</h3>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <h4 style='margin-bottom: 1rem;'>Summary Export</h4>
                """, unsafe_allow_html=True)
            st.download_button(
                "📥 Download Summary (TXT)",
                st.session_state.summary,
                "youtube_summary.txt",
                help="Download the generated summary as a text file"
            )
            
            if st.session_state.audio_file:
                with open(st.session_state.audio_file, "rb") as f:
                    st.download_button(
                        "🎵 Download Audio Summary",
                        f.read(),
                        "summary_audio.mp3",
                        mime="audio/mpeg",
                        help="Download the audio version of the summary"
                    )
        
        with col2:
            st.markdown("""
                <h4 style='margin-bottom: 1rem;'>Transcript Export</h4>
                """, unsafe_allow_html=True)
            st.download_button(
                "📄 Download Transcript (TXT)",
                st.session_state.transcript,
                "youtube_transcript.txt",
                help="Download the full video transcript"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_analyzer_view():
    """Create the main analyzer view"""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='margin-bottom: 1rem;'>🎥 YouTube Video Summarizer</h1>
            <p style='opacity: 0.8;'>Transform your video content into concise, actionable insights with metric-based evaluation</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Analysis Options
    with st.expander("🔧 Advanced Analysis Options", expanded=False):
        st.markdown("### Customize Your Analysis")
        
        # Metric-based evaluation info
        st.info("📊 **Quality Evaluation:** This app uses standard NLP metrics (BLEU, ROUGE, METEOR) to evaluate summary quality automatically.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            detail_level = st.selectbox(
                "Summary Detail Level",
                ["Concise", "Balanced", "Detailed"],
                help="Choose how detailed you want your summaries to be"
            )
            
            summary_format = st.selectbox(
                "Summary Format",
                ["Paragraph", "Bullet Points"],
                help="Choose the format for your summary"
            )
            
            include_timestamps = st.toggle(
                "Include Timestamps", 
                help="Add timestamps for key points in the summary"
            )
        
        with col2:
            processing_mode = st.selectbox(
                "Processing Mode",
                ["Fast", "Balanced", "High Quality"],
                help="Choose processing speed vs quality trade-off"
            )
            
            content_priority = st.multiselect(
                "Content Priorities",
                ["News/Current Events", "Educational", "Technical/Tutorial", "Entertainment", "Review/Opinion", "Interview/Podcast"],
                default=["Educational", "Technical/Tutorial"],
                help="Select content types to prioritize"
            )
    
    # Input section
    st.markdown('<div class="glass-card input-container">', unsafe_allow_html=True)
    url = st.text_input(
        "🔗 Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube video URL to get started"
    )
    
    if st.button("🚀 Generate Summary", type="primary"):
        if url:
            st.session_state.url = url
            
            # Prepare user preferences for enhanced processing
            user_preferences = {
                'detail_level': detail_level.lower(),
                'format': summary_format.lower().replace(' ', '_'),
                'include_timestamps': include_timestamps,
                'processing_mode': processing_mode.lower(),
                'content_priorities': content_priority
            }
            
            if enhanced_process_video(url, user_preferences):
                st.balloons()
                st.success("✨ Summary generated successfully!")
        else:
            st.warning("⚠️ Please enter a YouTube URL")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results
    display_modern_results()

def main():
    """Main function"""
    # Initialize
    init_session_states()
    
    # Set theme
    set_modern_style(True)  # Always use dark theme for consistency
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.title("⚙️ Navigation")
        
        # Navigation
        nav_selection = st.radio(
            "Go to",
            ["Analyzer", "History", "Favorites", "Settings"],
            key="navigation"
        )
        
        st.markdown("---")
        st.markdown("""
        <h3 style='margin-bottom: 1rem;'>📱 Features</h3>
        <ul style='list-style-type: none; padding: 0;'>
            <li>📝 Extractive summaries</li>
            <li>📊 Metric-based evaluation</li>
            <li>📈 Sentiment analysis</li>
            <li>🎧 Audio playback</li>
            <li>📋 Quality assessment</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on navigation
    if nav_selection == "Analyzer":
        create_analyzer_view()
    elif nav_selection == "History":
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 class='title-text'>📜 Analysis History</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.history:
            for i, entry in enumerate(reversed(st.session_state.history)):
                if not isinstance(entry, dict):
                    continue
                    
                title = entry.get('title', 'Untitled Video')
                timestamp = entry.get('timestamp', datetime.now().isoformat())
                summary = entry.get('summary', 'No summary available')
                sentiment = entry.get('sentiment', 0)
                
                with st.expander(f"{title} - {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}"):
                    st.markdown(f"**Summary:**\n{summary}")
                    st.markdown(f"**Average Sentiment:** {sentiment:.2f}")
        else:
            st.info("No analysis history yet")
    elif nav_selection == "Favorites":
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 class='title-text'>⭐ Favorite Summaries</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.favorites:
            for fav in reversed(st.session_state.favorites):
                with st.expander(fav['title']):
                    st.markdown(fav['summary'])
                    st.markdown(f"*Added on: {datetime.fromisoformat(fav['timestamp']).strftime('%Y-%m-%d %H:%M')}*")
                    if st.button("Remove from Favorites", key=f"remove_{fav['video_id']}"):
                        toggle_favorite(fav['video_id'], fav['title'], fav['summary'])
                        st.rerun()
        else:
            st.info("No favorite summaries yet")
    elif nav_selection == "Settings":
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 class='title-text'>⚙️ Settings</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("Settings functionality can be extended as needed.")
        
        # Clean history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            save_data(HISTORY_FILE, st.session_state.history)
            st.success("History cleared!")

if __name__ == "__main__":
    main()
