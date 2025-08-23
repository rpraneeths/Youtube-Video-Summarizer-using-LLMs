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

# Import FLAN-T5 for prompt refinement
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

# Load FLAN-T5 model for prompt refinement
refiner_model_name = "google/flan-t5-base"  # or flan-t5-small for low memory
try:
    refiner_tokenizer = AutoTokenizer.from_pretrained(refiner_model_name)
    refiner_model = AutoModelForSeq2SeqLM.from_pretrained(refiner_model_name)
    FLAN_T5_AVAILABLE = True
    logger.info("FLAN-T5 model loaded successfully for prompt refinement")
except Exception as e:
    logger.warning(f"Failed to load FLAN-T5 model: {e}")
    refiner_tokenizer = None
    refiner_model = None
    FLAN_T5_AVAILABLE = False

def refine_prompt(base_prompt: str) -> str:
    """Refine a base prompt using FLAN-T5 model."""
    # Check if FLAN-T5 is available
    if not FLAN_T5_AVAILABLE or refiner_tokenizer is None or refiner_model is None:
        logger.warning("FLAN-T5 model not available, skipping prompt refinement")
        return base_prompt
    
    try:
        # Check if the prompt is too long and truncate if necessary
        if len(base_prompt) > 400:  # Leave room for the instruction
            base_prompt = base_prompt[:400] + "..."
        
        input_text = f"Improve this instruction for better clarity and detail:\n\n{base_prompt}"
        inputs = refiner_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate refined prompt
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = refiner_model.generate(
                **inputs, 
                max_length=128, 
                min_length=20,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )
        
        refined_prompt = refiner_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Validate the refined prompt
        if not refined_prompt or len(refined_prompt.strip()) < 10:
            logger.warning("Refined prompt too short, using original")
            return base_prompt
            
        logger.info(f"Prompt refined successfully. Original length: {len(base_prompt)}, Refined length: {len(refined_prompt)}")
        return refined_prompt
        
    except Exception as e:
        logger.warning(f"Prompt refinement failed: {e}")
        return base_prompt  # Return original prompt if refinement fails

def remove_redundant_sentences(text: str) -> str:
    """Remove duplicate sentences from summary."""
    import re
    seen = set()
    sentences = re.split(r'(?<=[.!?]) +', text)
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean.lower() not in seen and len(s_clean) > 3:
            cleaned.append(s_clean)
            seen.add(s_clean.lower())
    return " ".join(cleaned)

def clean_summary_text(text: str) -> str:
    """Remove duplicate or highly similar sentences and repeated phrases with enhanced deduplication and noise removal."""
    import re
    from difflib import SequenceMatcher
    
    if not text or len(text.strip()) < 10:
        return text
    
    # Remove common noise patterns and repetitive phrases
    noise_patterns = [
        r'\b(letters?\s+from\s+\w+)\b',
        r'\b(film-maker\s+and\s+columnist)\b',
        r'\b(columnist\s+and\s+\w+)\b',
        r'\b(journalist\s+and\s+\w+)\b',
        r'\b(\w+\s+and\s+columnist)\b',
        r'\b(\w+\s+and\s+journalist)\b',
        r'\b(repeated\s+names?\s+like\s+\w+\s+\w+)\b',
        r'\b(um|uh|like|you\s+know|basically|actually|literally)\b',
        r'\b(so|well|right|okay|yeah|wow|amazing)\b'
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # First, remove repeated phrases within the same sentence (more aggressive)
    text = re.sub(r'\b(\w+\s+){1,6}\1', r'\1', text)
    
    # Split into sentences using more robust pattern
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    seen_sentences = set()
    
    for s in sentences:
        s_clean = s.strip()
        # Filter out extremely short or insignificant sentences
        if len(s_clean) < 8:
            continue
            
        # Skip sentences that are just noise patterns
        if re.search(r'^(letters?\s+from|film-maker|columnist|journalist)', s_clean, re.IGNORECASE):
            continue
            
        # Normalize sentence for comparison (lowercase, remove extra spaces, punctuation)
        s_normalized = re.sub(r'\s+', ' ', s_clean.lower()).strip()
        s_normalized = re.sub(r'[^\w\s]', '', s_normalized)  # Remove punctuation for comparison
        
        # Skip if exact duplicate (case-insensitive)
        if s_normalized in seen_sentences:
            continue
            
        # Skip if too similar to any previous sentence (very aggressive similarity check)
        is_similar = False
        for prev_sentence in cleaned:
            prev_normalized = re.sub(r'\s+', ' ', prev_sentence.lower()).strip()
            prev_normalized = re.sub(r'[^\w\s]', '', prev_normalized)
            
            # Use very aggressive threshold for similarity (0.7 as requested)
            ratio = SequenceMatcher(None, prev_normalized, s_normalized).ratio()
            if ratio > 0.7:
                is_similar = True
                break
                
        if not is_similar:
            cleaned.append(s_clean)
            seen_sentences.add(s_normalized)
    
    # Join sentences and do final cleanup
    result = " ".join(cleaned)
    
    # Remove any remaining repeated phrases across sentence boundaries (more aggressive)
    result = re.sub(r'\b(\w+\s+){2,8}\1', r'\1', result)
    
    # Remove repeated phrases that appear more than once in the entire summary
    words = result.split()
    for i in range(len(words) - 3):
        for j in range(4, min(10, len(words) - i)):
            phrase = ' '.join(words[i:i+j])
            if result.count(phrase) > 1:
                # Keep only the first occurrence
                result = result.replace(phrase, '', result.count(phrase) - 1)
    
    # Final cleanup: remove excessive whitespace and normalize punctuation
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\.{2,}', '.', result)
    result = re.sub(r'!{2,}', '!', result)
    result = re.sub(r'\?{2,}', '?', result)
    
    return result.strip()

def remove_repeated_lines(text: str) -> str:
    """Remove repeated line patterns that commonly occur in summaries with enhanced similarity checking and noise removal."""
    import re
    from difflib import SequenceMatcher
    
    if not text:
        return text
    
    # Remove common noise patterns from lines
    noise_patterns = [
        r'\b(letters?\s+from\s+\w+)\b',
        r'\b(film-maker\s+and\s+columnist)\b',
        r'\b(columnist\s+and\s+\w+)\b',
        r'\b(journalist\s+and\s+\w+)\b',
        r'\b(\w+\s+and\s+columnist)\b',
        r'\b(\w+\s+and\s+journalist)\b',
        r'\b(repeated\s+names?\s+like\s+\w+\s+\w+)\b'
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Split into lines
    lines = text.split('\n')
    cleaned_lines = []
    seen_lines = set()
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # Skip lines that are just noise patterns
        if re.search(r'^(letters?\s+from|film-maker|columnist|journalist)', line_clean, re.IGNORECASE):
            continue
            
        # Normalize line for comparison
        line_normalized = re.sub(r'\s+', ' ', line_clean.lower()).strip()
        line_normalized = re.sub(r'[^\w\s]', '', line_normalized)  # Remove punctuation
        
        # Skip if exact duplicate
        if line_normalized in seen_lines:
            continue
            
        # Skip if too similar to previous lines (check last 5 lines for better pattern detection)
        is_similar = False
        for prev_line in cleaned_lines[-3:]:  # Check last 3 lines as requested
            prev_normalized = re.sub(r'\s+', ' ', prev_line.lower()).strip()
            prev_normalized = re.sub(r'[^\w\s]', '', prev_normalized)
            
            # Use similarity threshold of 0.8 for lines as requested
            ratio = SequenceMatcher(None, prev_normalized, line_normalized).ratio()
            if ratio > 0.8:
                is_similar = True
                break
                
        if not is_similar:
            cleaned_lines.append(line_clean)
            seen_lines.add(line_normalized)
    
    return '\n'.join(cleaned_lines)

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
    - Support youtube.com/watch?v=, youtu.be/, youtube.com/embed/, m.youtube.com
    - Extract video ID reliably
    - Check for live streams, private videos, age-restricted content
    - Validate URL format and accessibility
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

def _filter_unrelated_topics(text: str) -> str:
    """Filter out unrelated topics to focus on brain, learning, neuroplasticity, stroke, and recovery content"""
    import re
    
    # Define patterns for unrelated topics that should be filtered out
    unrelated_patterns = [
        r'\b(climate\s+change|global\s+warming|carbon\s+emissions|greenhouse\s+gases)\b',
        r'\b(cancer|tumor|oncology|chemotherapy|radiation)\b',
        r'\b(politics|election|government|policy|legislation)\b',
        r'\b(sports|football|basketball|baseball|soccer)\b',
        r'\b(celebrity|gossip|entertainment\s+news|movie\s+reviews)\b',
        r'\b(technology\s+news|gadget\s+reviews|software\s+updates)\b',
        r'\b(journalists?\s+and\s+\w+|columnists?\s+and\s+\w+)\b',
        r'\b(letters?\s+from\s+\w+)\b',
        r'\b(film-maker\s+and\s+columnist)\b',
        r'\b(unrelated\s+names?\s+and\s+titles)\b'
    ]
    
    # Remove sentences containing unrelated topics
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if sentence contains unrelated topics
        contains_unrelated = False
        for pattern in unrelated_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                contains_unrelated = True
                break
        
        # Keep sentence if it doesn't contain unrelated topics
        if not contains_unrelated:
            filtered_sentences.append(sentence)
    
    # Rejoin filtered sentences
    filtered_text = ' '.join(filtered_sentences)
    
    # If too much content was filtered out, return original text
    if len(filtered_text) < len(text) * 0.3:  # If less than 30% remains
        logger.warning("Too much content filtered out, returning original text")
        return text
    
    return filtered_text

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
    
    # Step 2: Remove repeated words and phrases (e.g., "stroke recovery, stroke recovery")
    # Find and remove immediate repetitions
    text = re.sub(r'\b(\w+(?:\s+\w+){0,3})\s*,\s*\1\b', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+(?:\s+\w+){0,3})\s+and\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    
    # Step 3: Remove specific names of journalists, filmmakers, and unrelated people
    # Common names that often appear in transcripts but aren't relevant to content
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
    # Remove multiple consecutive punctuation marks
    text = re.sub(r'[.!?]{2,}', '.', text)
    text = re.sub(r'[,;]{2,}', ',', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    
    # Remove multiple consecutive spaces
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

def generate_contextual_prompt(transcript: str, content_category: ContentCategory, user_preferences: Optional[Dict] = None) -> str:
    """
    Generate optimized prompts based on content type and user preferences
    """
    base_prompt = f"""Summarize the following transcript in 5-6 sentences. Focus on:
         - How the brain changes (neuroplasticity)
         - Why people learn differently
         - How this knowledge helps with learning and stroke recovery
         Ignore names, unrelated topics, and repeated phrases.
         
         Please provide a comprehensive summary of the following {content_category.category.lower()} content:"""
    
    # Add category-specific instructions
    if content_category.preprocessing_strategy == "news_focused":
        base_prompt += """
        
        Focus on:
        - Key events and timeline
        - Main stakeholders and their roles
        - Factual information and verified details
        - Implications and potential impact
        - Include specific dates, numbers, and locations
        
        Ensure accuracy and objectivity in the summary."""
        
    elif content_category.preprocessing_strategy == "educational_structured":
        base_prompt += """
        
        Structure the summary to emphasize:
        - Main concepts and learning objectives
        - Key definitions and terminology
        - Examples and practical applications
        - Step-by-step explanations if applicable
        - Important takeaways and conclusions
        
        Make it easy to understand and follow."""
        
    elif content_category.preprocessing_strategy == "technical_stepwise":
        base_prompt += """
        
        Organize the summary around:
        - Tools and technologies mentioned
        - Step-by-step processes and procedures
        - Prerequisites and requirements
        - Expected outcomes and results
        - Common issues and troubleshooting tips
        - Best practices and recommendations
        
        Focus on actionable information."""
        
    elif content_category.preprocessing_strategy == "entertainment_highlight":
        base_prompt += """
        
        Highlight:
        - Main themes and storylines
        - Key characters and their roles
        - Memorable moments and highlights
        - Entertainment value and appeal
        - Overall tone and atmosphere
        
        Keep it engaging and fun."""
        
    elif content_category.preprocessing_strategy == "review_balanced":
        base_prompt += """
        
        Provide a balanced summary covering:
        - Product/service overview
        - Pros and advantages
        - Cons and limitations
        - Overall rating or recommendation
        - Target audience and use cases
        - Value for money assessment
        
        Present both positive and negative aspects fairly."""
        
    elif content_category.preprocessing_strategy == "interview_insights":
        base_prompt += """
        
        Extract and organize:
        - Key insights from each speaker
        - Main discussion topics and themes
        - Expert opinions and perspectives
        - Actionable advice and recommendations
        - Memorable quotes and statements
        - Background context and expertise
        
        Focus on valuable insights and takeaways."""
    
    # Add user preference customization if available
    if user_preferences:
        if user_preferences.get('detail_level') == 'detailed':
            base_prompt += "\n\nProvide a detailed summary with comprehensive coverage."
        elif user_preferences.get('detail_level') == 'concise':
            base_prompt += "\n\nProvide a concise summary focusing on key points only."
        
        if user_preferences.get('include_timestamps'):
            base_prompt += "\n\nInclude relevant timestamps for key points."
        
        if user_preferences.get('format') == 'bullet_points':
            base_prompt += "\n\nFormat the summary as bullet points for easy reading."
    
    base_prompt += f"\n\nContent to summarize:\n{transcript[:2000]}..."
    
    return base_prompt

def enhanced_summarization_pipeline(transcript: str, content_category: ContentCategory, user_preferences: Optional[Dict] = None) -> str:
    """
    Multi-stage summarization with intelligent model selection and enhanced deduplication
    """
    try:
        if not transcript:
            return "No transcript available for summarization."
        
        # Stage 1: Content Analysis and Preprocessing
        processed_transcript = content_aware_preprocessing(transcript, content_category)
        
        # Stage 1.5: Pre-summarization cleanup to remove noise and redundancy
        with st.status("Preprocessing transcript for cleaner summarization...", expanded=True) as status:
            # Step 1: Apply comprehensive transcript cleaning to remove noise, repeated phrases, and irrelevant names
            cleaned_transcript = clean_transcript(processed_transcript)
            logger.info(f"After transcript cleaning: {len(cleaned_transcript)} characters")
            
            # Step 2: Remove repeated phrases and irrelevant text using existing functions
            cleaned_transcript = clean_summary_text(cleaned_transcript)
            cleaned_transcript = remove_repeated_lines(cleaned_transcript)
            
            # Step 3: Filter out unrelated topics like climate change or cancer
            cleaned_transcript = _filter_unrelated_topics(cleaned_transcript)
            
            # Step 4: Limit input text length to 2000 characters for more focused summarization
            if len(cleaned_transcript) > 2000:
                # Keep the most relevant content (first 2000 characters)
                cleaned_transcript = cleaned_transcript[:2000]
                logger.info(f"Transcript truncated to {len(cleaned_transcript)} characters for focused summarization")
            
            status.update(label="Transcript preprocessed successfully!", state="complete")
        
        # Stage 2: Generate contextual prompt
        llm_prompt = generate_contextual_prompt(cleaned_transcript, content_category, user_preferences)
        
        # Stage 2.5: Refine prompt using FLAN-T5
        if FLAN_T5_AVAILABLE:
            with st.status("Refining prompt using FLAN-T5...", expanded=True) as status:
                refined_prompt = refine_prompt(llm_prompt)
                # Track if prompt refinement was used
                st.session_state.prompt_refinement_used = refined_prompt != llm_prompt
                status.update(label="Prompt refined successfully!", state="complete")
        else:
            # Skip refinement if FLAN-T5 is not available
            refined_prompt = llm_prompt
            st.session_state.prompt_refinement_used = False
            st.info("🤖 FLAN-T5 not available - using standard prompt optimization")
        
        # Stage 3: Model Selection based on content type
        model_name = _select_optimal_model(content_category)
        
        # Stage 4: Primary Summarization
        with st.status(f"Generating summary using {model_name}...", expanded=True) as status:
            try:
                summarizer = pipeline("summarization", model=model_name)
                
                # Split into chunks that fit within the model's context window
                max_chunk_size = _get_model_context_window(model_name)
                # Use non-overlapping chunks to prevent repetition
                chunks = []
                for i in range(0, len(processed_transcript), max_chunk_size):
                    chunk = processed_transcript[i:i+max_chunk_size]
                    if chunk.strip():  # Only add non-empty chunks
                        chunks.append(chunk)
                
                summarized_text = []
                for i, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        status.update(label=f"Processing chunk {i+1}/{len(chunks)}...")
                    
                    # Use refined prompt with chunk content
                    summary_input = f"{refined_prompt}\n\nContent:\n{chunk}"
                    summary = summarizer(summary_input, max_length=120, min_length=60, do_sample=False)
                    summarized_text.append(summary[0]['summary_text'])
                
                # Stage 5: Combine and refine summaries
                combined_summary = " ".join(summarized_text)
                
                # Log for debugging
                logger.info(f"Combined summary length: {len(combined_summary)}")
                logger.info(f"Number of chunks processed: {len(chunks)}")
                
                # Stage 5.5: Enhanced cleanup - apply both functions in sequence for comprehensive deduplication
                with st.status("Cleaning summary text...", expanded=True) as status:
                    # First pass: clean summary text (remove duplicates, near-duplicates, and repeated phrases)
                    cleaned_summary = clean_summary_text(combined_summary)
                    logger.info(f"After sentence deduplication length: {len(cleaned_summary)}")
                    
                    # Second pass: remove repeated lines with enhanced similarity checking
                    cleaned_summary = remove_repeated_lines(cleaned_summary)
                    logger.info(f"After line deduplication length: {len(cleaned_summary)}")
                    
                    # Third pass: additional cleanup for any remaining repetition
                    cleaned_summary = clean_summary_text(cleaned_summary)
                    logger.info(f"Final cleaned summary length: {len(cleaned_summary)}")
                    
                    # Fourth pass: final line cleanup to catch any remaining repetition
                    cleaned_summary = remove_repeated_lines(cleaned_summary)
                    logger.info(f"After final line cleanup length: {len(cleaned_summary)}")
                    
                    status.update(label="Summary text cleaned!", state="complete")
                
                # Stage 6: Post-processing and formatting
                final_summary = _format_summary_by_category(cleaned_summary, content_category)
                
                status.update(label="Summary generated successfully!", state="complete")
                return final_summary
                
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                # Fallback to basic summarization
                return _fallback_summarization(processed_transcript)
                
    except Exception as e:
        logger.error(f"Enhanced summarization pipeline error: {e}")
        return f"Summarization error: {str(e)}"

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

def _fallback_summarization(text: str) -> str:
    """Fallback summarization when primary method fails"""
    try:
        # Simple extractive summarization using TF-IDF
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback: simple sentence splitting by periods
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        # Calculate sentence importance using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Get sentence scores
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # Select top sentences
        top_indices = sentence_scores.argsort()[-3:][::-1]
        top_sentences = [sentences[i] for i in sorted(top_indices)]
        
        # Clean fallback summary text using enhanced deduplication
        fallback_summary = " ".join(top_sentences)
        cleaned_fallback = clean_summary_text(fallback_summary)
        cleaned_fallback = remove_repeated_lines(cleaned_fallback)
        # Final pass to catch any remaining repetition
        cleaned_fallback = clean_summary_text(cleaned_fallback)
        return cleaned_fallback
        
    except Exception as e:
        logger.error(f"Fallback summarization error: {e}")
        # Return first few sentences as last resort
        try:
            sentences = sent_tokenize(text)
            return " ".join(sentences[:3])
        except LookupError:
            # Final fallback: simple text truncation
            return text[:500] + "..." if len(text) > 500 else text

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
    page_title="YouTube Video Summarizer",
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

def install_packages():
    """Install required packages if not already installed"""
    required_packages = [
        'streamlit',
        'youtube_transcript_api',
        'transformers',
        'torch',
        'Pillow',
        'requests',
        'textblob',
        'matplotlib',
        'plotly',
        'numpy',
        'gtts',
        'pygame',
        'nltk',
        'scikit-learn',
        'pandas'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            st.warning(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Download required NLTK data
    try:
        # Download punkt tokenizer (essential for sentence tokenization)
        nltk.download('punkt', quiet=True)
        
        # Download additional NLTK resources
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Verify punkt is available
        try:
            from nltk.data import find
            find('tokenizers/punkt')
        except LookupError:
            st.error("NLTK punkt tokenizer not found. Please restart the app.")
            
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        st.warning("Some features may not work properly without NLTK data.")

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
        'theme': 'Dark',  # Changed from 'dark' to 'Dark'
        'font_size': 'Medium',  # Changed from 'medium' to 'Medium'
        'language': 'en',
        'auto_play': False
    })
    history = load_data(HISTORY_FILE, [])
    
    # Normalize settings values to ensure consistency
    if 'detail_level' in settings:
        detail_level = settings['detail_level']
        if detail_level.lower() == 'balanced':
            settings['detail_level'] = 'Balanced'
        elif detail_level.lower() == 'concise':
            settings['detail_level'] = 'Concise'
        elif detail_level.lower() == 'detailed':
            settings['detail_level'] = 'Detailed'
    
    if 'summary_format' in settings:
        summary_format = settings['summary_format']
        if summary_format.lower() == 'paragraph':
            settings['summary_format'] = 'Paragraph'
        elif summary_format.lower() == 'bullet_points':
            settings['summary_format'] = 'Bullet Points'
        elif summary_format.lower() == 'structured':
            settings['summary_format'] = 'Structured'
        elif summary_format.lower() == 'mixed':
            settings['summary_format'] = 'Mixed'
    
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
    if 'prompt_refinement_used' not in st.session_state:
        st.session_state.prompt_refinement_used = False

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
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.1) 0%,
                rgba(255, 255, 255, 0.05) 100%);
            z-index: -1;
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
        
        .metric-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #6e48aa, #9d50bb);
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
        
        /* Input styles */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: rgba(255, 255, 255, 0.4);
            box-shadow: 0 0 0 3px rgba(110, 72, 170, 0.2);
            background: rgba(255, 255, 255, 0.15);
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
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 24px rgba(110, 72, 170, 0.3);
        }
        
        .stButton > button:hover::before {
            opacity: 1;
        }
        
        /* Tab styles */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            padding: 0.5rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            color: rgba(255, 255, 255, 0.8);
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.15);
            color: #ffffff;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(110, 72, 170, 0.3);
        }
        
        /* Sidebar styles */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Radio button styles */
        .stRadio > div {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .stRadio > div:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        /* Select box styles */
        .stSelectbox > div > div > div {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ffffff;
            border-radius: 16px;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div > div:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        /* Slider styles */
        .stSlider > div > div > div {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        .stSlider > div > div > div > div {
            background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
        }
        
        /* Expander styles */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            color: #ffffff;
            padding: 1rem 1.5rem;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        /* Scrollbar styles */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        /* Highlight styles */
        .highlight {
            background: rgba(110, 72, 170, 0.2);
            padding: 2px 8px;
            border-radius: 8px;
            color: #ffffff;
            transition: all 0.3s ease;
        }
        
        .highlight:hover {
            background: rgba(110, 72, 170, 0.3);
            transform: translateY(-1px);
        }
        
        /* Footer styles */
        .footer {
            text-align: center;
            padding: 2rem;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            margin-top: 2rem;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .glass-card {
                padding: 1.5rem;
            }
            
            .metric-card {
                padding: 1.5rem;
            }
            
            .title-text {
                font-size: 2rem;
            }
        }
        
        /* Animation keyframes */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .floating {
            animation: float 3s ease-in-out infinite;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top: 3px solid #6e48aa;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    """, unsafe_allow_html=True)

def extract_key_sentences(text, num_sentences=5):
    """Extract key sentences from the text using TF-IDF"""
    try:
        # Ensure NLTK data is available first
        ensure_nltk_data()
        
        # Try to use NLTK sent_tokenize
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback: simple sentence splitting by periods
            st.warning("NLTK punkt not available, using fallback sentence splitting")
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = np.sum(tfidf_matrix[i].toarray())
            sentence_scores.append((score, sentence))
        
        # Get top sentences
        key_sentences = sorted(sentence_scores, reverse=True)[:num_sentences]
        key_sentences = [sentence for _, sentence in key_sentences]
        
        return key_sentences
        
    except Exception as e:
        st.error(f"Error extracting key sentences: {str(e)}")
        # Return simple fallback
        return [text[:200] + "..." if len(text) > 200 else text]

def highlight_key_sentences(summary):
    """Highlight key sentences in the summary"""
    key_sentences = extract_key_sentences(summary)
    highlighted_summary = summary
    
    for sentence in key_sentences:
        highlighted_sentence = f'<span style="background-color: rgba(110, 72, 170, 0.2); padding: 2px 4px; border-radius: 4px;">{sentence}</span>'
        highlighted_summary = highlighted_summary.replace(sentence, highlighted_sentence)
    
    return highlighted_summary

def get_sentiment_interval(duration_min):
    """Determine sentiment analysis interval based on video duration"""
    if duration_min <= 5:
        return 1  # Every 1 minute
    elif duration_min <= 10:
        return 2  # Every 2 minutes
    elif duration_min <= 20:
        return 3  # Every 3 minutes
    elif duration_min <= 30:
        return 4  # Every 4 minutes
    elif duration_min <= 60:
        return 5  # Every 5 minutes
    else:
        return 10  # Every 10 minutes

def create_analysis_distribution_chart():
    """Create a chart showing the distribution of analyses"""
    try:
        if not st.session_state.history or len(st.session_state.history) < 2:
            return None
            
        # Prepare data for the chart
        df = pd.DataFrame(st.session_state.history)
        
        # Ensure timestamp column exists and is valid
        if 'timestamp' not in df.columns:
            return None
            
        # Convert timestamp to datetime and extract hour
        try:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        except Exception as e:
            logger.warning(f"Could not parse timestamps: {e}")
            return None
        
        # Count analyses by hour
        hour_counts = df['hour'].value_counts().sort_index()
        
        # Ensure we have data to plot
        if hour_counts.empty or len(hour_counts) < 2:
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add bar trace
        fig.add_trace(
            go.Bar(
                x=hour_counts.index,
                y=hour_counts.values,
                name="Analyses",
                marker_color='rgba(110, 72, 170, 0.8)',
                hovertemplate="Hour: %{x}<br>Count: %{y}<extra></extra>"
            )
        )
        
        # Add line trace for trend
        fig.add_trace(
            go.Scatter(
                x=hour_counts.index,
                y=hour_counts.values,
                mode='lines',
                name="Trend",
                line=dict(
                    color='rgba(255, 255, 255, 0.7)',
                    width=3,
                    shape='spline'
                ),
                hovertemplate="Hour: %{x}<br>Count: %{y}<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Analysis Activity by Hour of Day",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'color': '#ffffff'}
            },
            xaxis=dict(
                title=dict(text="Hour of Day", font=dict(color='#ffffff')),
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[-0.5, 23.5],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff')
            ),
            yaxis=dict(
                title=dict(text="Number of Analyses", font=dict(color='#ffffff')),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#ffffff')
            ),
            hoverlabel=dict(
                bgcolor='rgba(37, 38, 89, 0.9)',
                font_size=14,
                font_color='white'
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating analysis distribution chart: {str(e)}")
        return None

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
                print(f"No transcript available for this video.{e}")
                st.error("No transcript available for this video.")
                return
            else:
                st.error(f"Error fetching transcript: {str(e)}")
                return
        
        # Calculate video duration in minutes
        # Handle both dictionary and object access for transcript segments
        try:
            if hasattr(transcript[-1], 'start'):
                # FetchedTranscriptSnippet object
                duration_sec = transcript[-1].start + transcript[-1].duration
            else:
                # Dictionary format
                duration_sec = transcript[-1]['start'] + transcript[-1]['duration']
        except (AttributeError, KeyError, IndexError):
            # Fallback: estimate duration from transcript length
            duration_sec = len(transcript) * 10  # Assume 10 seconds per segment
        duration_min = duration_sec / 60
        
        # Determine appropriate interval
        interval = get_sentiment_interval(duration_min)
        
        # Group transcript by time intervals
        sentiments = []
        time_labels = []
        current_text = ""
        current_start = 0
        
        for segment in transcript:
            # Handle both dictionary and object access for transcript segments
            try:
                if hasattr(segment, 'start'):
                    # FetchedTranscriptSnippet object
                    segment_min = segment.start / 60
                    segment_text = segment.text
                else:
                    # Dictionary format
                    segment_min = segment['start'] / 60
                    segment_text = segment['text']
            except (AttributeError, KeyError):
                # Skip malformed segments
                continue
            
            # If we've reached a new interval, analyze the accumulated text
            if segment_min - current_start >= interval:
                if current_text:
                    try:
                        blob = TextBlob(current_text)
                        sentiments.append(blob.sentiment.polarity)
                        time_labels.append(f"{int(current_start)}-{int(current_start)+interval}min")
                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for segment: {e}")
                        sentiments.append(0.0)  # Neutral sentiment as fallback
                        time_labels.append(f"{int(current_start)}-{int(current_start)+interval}min")
                
                # Reset for next interval
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
                sentiments.append(0.0)  # Neutral sentiment as fallback
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

def create_modern_sentiment_analysis(sentiments, time_labels, duration_min, interval):
    """Create a modern sentiment analysis visualization"""
    # Create figure
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
    
    # Calculate metrics for annotations
    avg_sentiment = np.mean(sentiments)
    max_sentiment = np.max(sentiments)
    min_sentiment = np.min(sentiments)
    
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
    
    # Add annotations for key points
    fig.add_annotation(
        x=time_labels[np.argmax(sentiments)],
        y=max_sentiment,
        text=f"Most Positive: {max_sentiment:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        font=dict(color="#ffffff")
    )
    
    fig.add_annotation(
        x=time_labels[np.argmin(sentiments)],
        y=min_sentiment,
        text=f"Most Negative: {min_sentiment:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=40,
        font=dict(color="#ffffff")
    )
    
    return fig

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
    Enhanced video processing pipeline with intelligent features
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
        
        # Stage 5: Content-Aware Preprocessing
        with st.status("Preprocessing content for optimal summarization...", expanded=True) as status:
            processed_transcript = content_aware_preprocessing(transcript, content_category, metadata)
            status.update(label="Content preprocessed!", state="complete")
        
        # Stage 6: Enhanced Summarization
        with st.status("Generating intelligent summary...", expanded=True) as status:
            try:
                summary = enhanced_summarization_pipeline(processed_transcript, content_category, user_preferences)
                st.session_state.summary = summary
                status.update(label="Summary generated successfully!", state="complete")
            except Exception as e:
                st.error(f"Enhanced summarization failed: {str(e)}")
                # Fallback to basic summarization
                st.warning("Falling back to basic summarization...")
                summary = _fallback_summarization(processed_transcript)
                st.session_state.summary = summary
                status.update(label="Basic summary generated!", state="complete")
        
        # Stage 7: Generate Audio Summary
        with st.status("Generating audio summary...", expanded=True) as status:
            audio_file = text_to_speech(st.session_state.summary)
            if audio_file:
                st.session_state.audio_file = audio_file
                status.update(label="Audio summary generated!", state="complete")
        
        # Stage 8: Enhanced Sentiment Analysis
        with st.status("Analyzing sentiment and content insights...", expanded=True) as status:
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

def process_video(url):
    """Legacy process_video function - now calls enhanced version"""
    return enhanced_process_video(url)

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

def update_settings(new_settings):
    """Update settings and save to file"""
    st.session_state.settings.update(new_settings)
    save_data(SETTINGS_FILE, st.session_state.settings)

def clean_history():
    """Remove history entries older than 30 days"""
    thirty_days_ago = datetime.now() - timedelta(days=30)
    st.session_state.history = [
        entry for entry in st.session_state.history
        if datetime.fromisoformat(entry['timestamp']) > thirty_days_ago
    ]
    save_data(HISTORY_FILE, st.session_state.history)

def display_modern_results():
    """Display results in a modern UI"""
    if not st.session_state.analysis_complete:
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    tabs = st.tabs(["📝 Summary", "📊 Analysis", "⚡ Performance", "💾 Export"])
    
    with tabs[0]:  # Summary Tab
        st.markdown("""
            <h3 style='margin-bottom: 1.5rem;'>Video Summary</h3>
            """, unsafe_allow_html=True)
        
        try:
            highlighted_summary = highlight_key_sentences(st.session_state.summary)
        except Exception as e:
            st.warning(f"Could not highlight key sentences: {str(e)}")
            highlighted_summary = st.session_state.summary
        
        st.markdown(
            f'<div class="glass-card summary-text">{highlighted_summary}</div>', 
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
    
    with tabs[1]:  # Analysis Tab
        st.markdown("""
            <h3 style='margin-bottom: 1.5rem;'>Content Analysis & Insights</h3>
            """, unsafe_allow_html=True)
        
        # Enhanced content information
        if hasattr(st.session_state, 'content_category') and st.session_state.content_category:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Content Category",
                    st.session_state.content_category.category,
                    f"Confidence: {st.session_state.content_category.confidence:.1%}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.session_state.content_category.subcategories:
                    st.markdown("**Subcategories:**")
                    for subcat in st.session_state.content_category.subcategories:
                        st.write(f"• {subcat}")
            
            with col2:
                if hasattr(st.session_state, 'video_metadata') and st.session_state.video_metadata:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Channel",
                        st.session_state.video_metadata.channel,
                        f"Upload: {st.session_state.video_metadata.upload_date}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.session_state.video_metadata.view_count > 0:
                        st.markdown(f"**Views:** {st.session_state.video_metadata.view_count:,}")
        
        # Transcript method used
        if hasattr(st.session_state, 'transcript_method') and st.session_state.transcript_method:
            st.info(f"📝 **Transcript Source:** {st.session_state.transcript_method}")
        
        # Prompt refinement indicator
        if st.session_state.prompt_refinement_used:
            st.success("🤖 **AI Prompt Refinement:** FLAN-T5 was used to enhance the summarization prompt for better results")
        
        # Sentiment analysis
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
            st.markdown("### 📊 Sentiment Analysis")
            st.info("Sentiment analysis completed. Chart will be displayed after analysis.")
    
    with tabs[2]:  # Performance Tab
        st.markdown("""
            <h3 style='margin-bottom: 1.5rem;'>Performance Metrics</h3>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Original Length",
                f"{len(st.session_state.transcript):,}",
                "characters"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Summary Length",
                f"{len(st.session_state.summary):,}",
                "characters"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if st.session_state.execution_time is not None:
                st.metric(
                    "Processing Time",
                    f"{st.session_state.execution_time:.2f}",
                    "seconds"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.transcript and st.session_state.summary:
            compression_ratio = (len(st.session_state.summary) / len(st.session_state.transcript)) * 100
            st.markdown('<div class="glass-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.metric(
                "Compression Ratio",
                f"{compression_ratio:.1f}%",
                help="Percentage of original text length"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
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

def create_dashboard():
    """Create the main dashboard view"""
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 class='title-text'>📊 Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    # Quick Actions Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🚀 Quick Actions")
    cols = st.columns(4)
    with cols[0]:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.current_view = "analyzer"
    with cols[1]:
        if st.button("View History", use_container_width=True):
            st.session_state.current_view = "history"
    with cols[2]:
        if st.button("Favorites", use_container_width=True):
            st.session_state.current_view = "favorites"
    with cols[3]:
        if st.button("Settings", use_container_width=True):
            st.session_state.current_view = "settings"
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Metrics
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Key Metrics")
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric(
            "Total Videos Analyzed",
            len(st.session_state.history),
            delta="+1" if len(st.session_state.history) > 0 else None
        )
    
    with metric_cols[1]:
        # Modified this section to handle missing sentiment values
        if st.session_state.history:
            total_sentiment = 0
            valid_entries = 0
            for entry in st.session_state.history:
                if isinstance(entry, dict) and 'sentiment' in entry:
                    total_sentiment += entry['sentiment']
                    valid_entries += 1
            avg_sentiment = total_sentiment / valid_entries if valid_entries > 0 else 0
        else:
            avg_sentiment = 0
            
        st.metric(
            "Average Sentiment",
            f"{avg_sentiment:.2f}",
            delta=None
        )
    
    with metric_cols[2]:
        st.metric(
            "Favorite Summaries",
            len(st.session_state.favorites),
            delta=None
        )
    
    with metric_cols[3]:
        if st.session_state.history:
            # Get the most recent valid timestamp
            last_timestamp = None
            for entry in reversed(st.session_state.history):
                if isinstance(entry, dict) and 'timestamp' in entry:
                    try:
                        last_timestamp = datetime.fromisoformat(entry['timestamp'])
                        break
                    except:
                        continue
            
            if last_timestamp:
                st.metric("Last Analysis", last_timestamp.strftime("%H:%M:%S"))
            else:
                st.metric("Last Analysis", "Unknown")
        else:
            st.metric("Last Analysis", "Never")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Recent Activity
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🕒 Recent Activity")
    
    if st.session_state.history:
        # Show last 5 valid entries
        shown_count = 0
        for entry in reversed(st.session_state.history):
            if shown_count >= 5:
                break
                
            if isinstance(entry, dict) and 'title' in entry and 'timestamp' in entry:
                with st.container():
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.markdown(f"**{entry.get('title', 'Untitled Video')}**")
                    with cols[1]:
                        st.markdown(f"Sentiment: {entry.get('sentiment', 0):.2f}")
                    with cols[2]:
                        try:
                            timestamp = datetime.fromisoformat(entry['timestamp'])
                            st.markdown(f"Processed: {timestamp.strftime('%Y-%m-%d %H:%M')}")
                        except:
                            st.markdown("Processed: Unknown")
                shown_count += 1
    else:
        st.info("No recent activity")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis Distribution
    if st.session_state.history and len(st.session_state.history) > 1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Distribution")
        
        # Create visualization of analysis types
        fig = create_analysis_distribution_chart()
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analysis data available for visualization")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Favorites Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⭐ Favorite Summaries")
    
    if st.session_state.favorites:
        for fav in st.session_state.favorites:
            with st.expander(fav.get('title', 'Untitled Video')):
                st.markdown(fav.get('summary', 'No summary available'))
                st.markdown(f"*Added on: {datetime.fromisoformat(fav.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M')}*")
    else:
        st.info("No favorite summaries yet")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_history_view():
    """Show history view"""
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 class='title-text'>📜 Analysis History</h1>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            if not isinstance(entry, dict):
                continue
                
            # Safely get values with defaults
            title = entry.get('title', 'Untitled Video')
            timestamp = entry.get('timestamp', datetime.now().isoformat())
            summary = entry.get('summary', 'No summary available')
            sentiment = entry.get('sentiment', 0)
            
            with st.expander(f"{title} - {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(f"**Summary:**\n{summary}")
                st.markdown(f"**Average Sentiment:** {sentiment:.2f}")
    else:
        st.info("No analysis history yet")

def show_favorites_view():
    """Show favorites view"""
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

def show_settings_view():
    """Show settings view with comprehensive options"""
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 class='title-text'>⚙️ Settings</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different settings categories
    tabs = st.tabs(["Appearance", "Audio", "Export", "Privacy", "Performance"])
    
    with tabs[0]:  # Appearance Settings
        st.markdown("### 🎨 Appearance Settings")
        with st.container():
            # Theme Selection
            theme = st.select_slider(
                "Color Theme",
                options=["Dark", "Light", "System", "Custom"],
                value=st.session_state.settings.get('theme', 'Dark')
            )
            
            # Custom Colors (if Custom theme selected)
            if theme == "Custom":
                primary_color = st.color_picker(
                    "Primary Color", 
                    st.session_state.settings.get('primary_color', '#6e48aa')
                )
                secondary_color = st.color_picker(
                    "Secondary Color", 
                    st.session_state.settings.get('secondary_color', '#9d50bb')
                )
                st.session_state.settings['primary_color'] = primary_color
                st.session_state.settings['secondary_color'] = secondary_color
            
            # Font Settings
            font_family = st.selectbox(
                "Font Family",
                ["Poppins", "Inter", "Roboto", "Open Sans", "Custom"],
                index=["Poppins", "Inter", "Roboto", "Open Sans", "Custom"].index(
                    st.session_state.settings.get('font_family', 'Poppins')
                )
            )
            
            if font_family == "Custom":
                custom_font = st.text_input(
                    "Custom Font Name",
                    value=st.session_state.settings.get('custom_font', '')
                )
                st.session_state.settings['custom_font'] = custom_font
            
            # Text Size Controls
            font_size = st.select_slider(
                "Font Size",
                options=["Very Small", "Small", "Medium", "Large", "Very Large"],
                value=st.session_state.settings.get('font_size', 'Medium')
            )
            
            # Animation Settings
            animations_enabled = st.toggle(
                "Enable Animations", 
                st.session_state.settings.get('animations_enabled', True)
            )
            animation_speed = st.slider(
                "Animation Speed", 
                0.1, 2.0, 
                st.session_state.settings.get('animation_speed', 1.0)
            )
    with tabs[1]:  # Analysis Tab
        st.markdown("### 🔍 Analysis Settings")
        with st.container():
            # Content Analysis Preferences
            st.markdown("#### Content Analysis Options")
            detail_level = st.selectbox(
                "Summary Detail Level",
                ["Concise", "Balanced", "Detailed"],
                index=["Concise", "Balanced", "Detailed"].index(
                    st.session_state.settings.get('detail_level', 'Balanced')
                ),
                help="Choose how detailed you want your summaries to be"
            )
            
            include_timestamps = st.toggle(
                "Include Timestamps in Summary", 
                st.session_state.settings.get('include_timestamps', False),
                help="Add timestamps for key points in the summary"
            )
            
            # Get the stored summary format and convert to title case for matching
            stored_summary_format_1 = st.session_state.settings.get('summary_format', 'Paragraph')
            # Convert from snake_case to title case to match the options list
            if stored_summary_format_1 == 'paragraph':
                stored_summary_format_1 = 'Paragraph'
            elif stored_summary_format_1 == 'bullet_points':
                stored_summary_format_1 = 'Bullet Points'
            elif stored_summary_format_1 == 'structured':
                stored_summary_format_1 = 'Structured'
            elif stored_summary_format_1 == 'mixed':
                stored_summary_format_1 = 'Mixed'
            
            summary_format = st.selectbox(
                "Summary Format",
                ["Paragraph", "Bullet Points", "Structured", "Mixed"],
                index=["Paragraph", "Bullet Points", "Structured", "Mixed"].index(stored_summary_format_1)
            )
            
            # Model Selection Preferences
            st.markdown("#### Model Preferences")
            preferred_model = st.selectbox(
                "Preferred Summarization Model",
                ["Auto-select (Recommended)", "BART", "Pegasus", "T5", "Fast"],
                index=["Auto-select (Recommended)", "BART", "Pegasus", "T5", "Fast"].index(
                    st.session_state.settings.get('preferred_model', 'Auto-select (Recommended)')
                ),
                help="Choose your preferred model or let the system auto-select based on content"
            )
            
            # Content Type Preferences
            st.markdown("#### Content Type Preferences")
            content_preferences = st.multiselect(
                "Prioritize These Content Types",
                ["News/Current Events", "Educational", "Technical/Tutorial", "Entertainment", "Review/Opinion", "Interview/Podcast"],
                default=st.session_state.settings.get('content_preferences', ["Educational", "Technical/Tutorial"]),
                help="Select content types you're most interested in for better summarization"
            )
            
            # Save preferences
            if st.button("Save Analysis Preferences"):
                st.session_state.settings.update({
                    'detail_level': detail_level.lower(),
                    'include_timestamps': include_timestamps,
                    'summary_format': summary_format.lower().replace(' ', '_'),
                    'preferred_model': preferred_model,
                    'content_preferences': content_preferences
                })
                save_data(SETTINGS_FILE, st.session_state.settings)
                st.success("Analysis preferences saved!")
        
        # Show current sentiment analysis if available
        if st.session_state.sentiment_fig is not None:
            st.markdown("### 📊 Sentiment Analysis")
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
            st.markdown("### 📊 Sentiment Analysis")
            st.info("Sentiment analysis completed. Chart will be displayed after analysis.")

    
    with tabs[2]:  # Export Settings
        st.markdown("### 💾 Export Settings")
        with st.container():
            # File Format Preferences
            formats = st.multiselect(
                "Default Export Formats",
                ["TXT", "PDF", "DOCX", "HTML", "JSON"],
                default=st.session_state.settings.get('export_formats', ["TXT", "PDF"])
            )
            
            # Export Options
            include_timestamps = st.toggle(
                "Include Timestamps", 
                st.session_state.settings.get('include_timestamps', True)
            )
            include_analytics = st.toggle(
                "Include Analytics", 
                st.session_state.settings.get('include_analytics', True)
            )
            include_visualizations = st.toggle(
                "Include Visualizations", 
                st.session_state.settings.get('include_visualizations', True)
            )
    
    with tabs[3]:  # Privacy Settings
        st.markdown("### 🔒 Privacy Settings")
        with st.container():
            # Data Collection
            save_history = st.toggle(
                "Save Analysis History",
                st.session_state.settings.get('save_history', True)
            )
            share_analytics = st.toggle(
                "Share Anonymous Analytics",
                st.session_state.settings.get('share_analytics', False)
            )
            
            # Data Retention
            retention_days = st.slider(
                "Data Retention Period (days)",
                1, 365,
                st.session_state.settings.get('retention_days', 30)
            )
    
    with tabs[4]:  # Performance Settings
        st.markdown("### ⚡ Performance Settings")
        with st.container():
            # Processing Mode
            mode = st.selectbox(
                "Processing Mode",
                ["Fast", "Balanced", "High Quality"],
                index=["Fast", "Balanced", "High Quality"].index(
                    st.session_state.settings.get('processing_mode', 'Balanced')
                )
            )
            
            # Hardware Acceleration
            use_gpu = st.toggle(
                "Use GPU Acceleration",
                st.session_state.settings.get('use_gpu', True)
            )
            
            # Cache Settings
            cache_enabled = st.toggle(
                "Enable Caching",
                st.session_state.settings.get('cache_enabled', True)
            )
            cache_size = st.slider(
                "Cache Size (MB)",
                100, 1000,
                st.session_state.settings.get('cache_size', 500)
            )
    
    # Save button for all settings
    if st.button("Save All Settings", type="primary"):
        # Update settings dictionary
        st.session_state.settings.update({
            'theme': theme,
            'font_family': font_family,
            'font_size': font_size,
            'animations_enabled': animations_enabled,
            'animation_speed': animation_speed,
            'export_formats': formats,
            'include_timestamps': include_timestamps,
            'include_analytics': include_analytics,
            'include_visualizations': include_visualizations,
            'save_history': save_history,
            'share_analytics': share_analytics,
            'retention_days': retention_days,
            'processing_mode': mode,
            'use_gpu': use_gpu,
            'cache_enabled': cache_enabled,
            'cache_size': cache_size
        })
        
        # Save settings to file
        save_data(SETTINGS_FILE, st.session_state.settings)
        
        # Apply theme changes
        set_modern_style(st.session_state.settings['theme'] == 'Dark')
        
        st.success("Settings saved successfully!")
        st.rerun()

def create_analyzer_view():
    """Create the main analyzer view with enhanced options"""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='margin-bottom: 1rem;'>🎥 YouTube Video Summarizer</h1>
            <p style='opacity: 0.8;'>Transform your video content into concise, actionable insights with AI-powered FLAN-T5 prompt refinement</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Analysis Options
    with st.expander("🔧 Advanced Analysis Options", expanded=False):
        st.markdown("### Customize Your Analysis")
        
        # FLAN-T5 integration info
        st.info("🤖 **AI Enhancement:** This app uses FLAN-T5 to automatically refine and improve summarization prompts for better results. This happens automatically during processing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get the stored detail level and convert to title case for matching
            stored_detail_level = st.session_state.settings.get('detail_level', 'Balanced')
            # Convert to title case to match the options list
            if stored_detail_level == 'detailed':
                stored_detail_level = 'Detailed'
            elif stored_detail_level == 'concise':
                stored_detail_level = 'Concise'
            elif stored_detail_level == 'balanced':
                stored_detail_level = 'Balanced'
            
            detail_level = st.selectbox(
                "Summary Detail Level",
                ["Concise", "Balanced", "Detailed"],
                index=["Concise", "Balanced", "Detailed"].index(stored_detail_level),
                help="Choose how detailed you want your summaries to be"
            )
            
            # Get the stored summary format and convert to title case for matching
            stored_summary_format_2 = st.session_state.settings.get('summary_format', 'Paragraph')
            # Convert from snake_case to title case to match the options list
            if stored_summary_format_2 == 'paragraph':
                stored_summary_format_2 = 'Paragraph'
            elif stored_summary_format_2 == 'bullet_points':
                stored_summary_format_2 = 'Bullet Points'
            elif stored_summary_format_2 == 'structured':
                stored_summary_format_2 = 'Structured'
            elif stored_summary_format_2 == 'mixed':
                stored_summary_format_2 = 'Mixed'
            
            summary_format = st.selectbox(
                "Summary Format",
                ["Paragraph", "Bullet Points", "Structured", "Mixed"],
                index=["Paragraph", "Bullet Points", "Structured", "Mixed"].index(stored_summary_format_2)
            )
            
            include_timestamps = st.toggle(
                "Include Timestamps", 
                st.session_state.settings.get('include_timestamps', False),
                help="Add timestamps for key points in the summary"
            )
        
        with col2:
            preferred_model = st.selectbox(
                "Preferred Model",
                ["Auto-select (Recommended)", "BART", "Pegasus", "T5", "Fast"],
                index=["Auto-select (Recommended)", "BART", "Pegasus", "T5", "Fast"].index(
                    st.session_state.settings.get('preferred_model', 'Auto-select (Recommended)')
                )
            )
            
            content_priority = st.multiselect(
                "Content Priorities",
                ["News/Current Events", "Educational", "Technical/Tutorial", "Entertainment", "Review/Opinion", "Interview/Podcast"],
                default=st.session_state.settings.get('content_preferences', ["Educational", "Technical/Tutorial"]),
                help="Select content types to prioritize"
            )
            
            processing_mode = st.selectbox(
                "Processing Mode",
                ["Fast", "Balanced", "High Quality"],
                index=["Fast", "Balanced", "High Quality"].index(
                    st.session_state.settings.get('processing_mode', 'Balanced')
                )
            )
        
        # Save preferences button
        if st.button("💾 Save Preferences", type="secondary"):
            st.session_state.settings.update({
                'detail_level': detail_level.lower(),
                'summary_format': summary_format.lower().replace(' ', '_'),
                'include_timestamps': include_timestamps,
                'preferred_model': preferred_model,
                'content_preferences': content_priority,
                'processing_mode': processing_mode
            })
            save_data(SETTINGS_FILE, st.session_state.settings)
            st.success("Preferences saved! They will be applied to your next analysis.")
    
    # Input section
    st.markdown('<div class="glass-card input-container">', unsafe_allow_html=True)
    url = st.text_input(
        "🔗 Enter YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube video URL to get started"
    )
    
    # Show supported URL formats
    with st.expander("📋 Supported URL Formats"):
        st.markdown("""
        **Supported YouTube URL formats:**
        - `https://www.youtube.com/watch?v=VIDEO_ID`
        - `https://youtu.be/VIDEO_ID`
        - `https://www.youtube.com/embed/VIDEO_ID`
        - `https://m.youtube.com/watch?v=VIDEO_ID`
        - `https://www.youtube.com/shorts/VIDEO_ID`
        """)
    
    if st.button("🚀 Generate Summary", type="primary"):
        if url:
            st.session_state.url = url
            
            # Prepare user preferences for enhanced processing
            user_preferences = {
                'detail_level': detail_level.lower(),
                'format': summary_format.lower().replace(' ', '_'),
                'include_timestamps': include_timestamps,
                'preferred_model': preferred_model,
                'content_priorities': content_priority,
                'processing_mode': processing_mode
            }
            
            if enhanced_process_video(url, user_preferences):
                st.balloons()
                st.success("✨ Summary generated successfully!")
        else:
            st.warning("⚠️ Please enter a YouTube URL")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results
    display_modern_results()
    
    # Footer
    st.markdown("""
    <div class="glass-card" style="text-align: center; margin-top: 3rem;">
        <p style="margin: 0;">Made with ❤️ | Powered by Enhanced AI Models</p>
    </div>
    """, unsafe_allow_html=True)

def ensure_nltk_data():
    """Ensure NLTK data is available, download if needed"""
    if not NLTK_AVAILABLE:
        st.warning("NLTK is not available. Using fallback text processing.")
        return False
        
    try:
        # Try to import and use punkt
        import nltk
        from nltk.data import find
        
        # Check if punkt is available
        try:
            find('tokenizers/punkt')
            find('tokenizers/punkt_tab')
            return True
        except LookupError:
            st.warning("NLTK punkt tokenizer not found. Downloading...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                nltk.download('stopwords', quiet=True)
                
                # Verify downloads
                find('tokenizers/punkt')
                find('tokenizers/punkt_tab')
                st.success("NLTK data downloaded successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to download NLTK data: {str(e)}")
                return False
    except Exception as e:
        st.error(f"NLTK import error: {str(e)}")
        return False

def main():
    """Main function"""
    # Install required packages
    # install_packages()
    
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    # Initialize
    init_session_states()
    
    # Ensure settings has a default theme if not set
    if not isinstance(st.session_state.settings, dict):
        st.session_state.settings = {
            'theme': 'dark',
            'font_size': 'medium',
            'language': 'en',
            'auto_play': False
        }
    elif 'theme' not in st.session_state.settings:
        st.session_state.settings['theme'] = 'dark'
    
    if st.session_state.settings.get('theme') == 'dark':
        st.session_state.settings['theme'] = 'Dark'
    if st.session_state.settings.get('font_size') == 'medium':
        st.session_state.settings['font_size'] = 'Medium'
    
    set_modern_style(st.session_state.settings['theme'].lower() == 'dark')

    
    # Clean up old history entries
    clean_history()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.title("⚙️ Navigation")
        
        # Navigation
        nav_selection = st.radio(
            "Go to",
            ["Dashboard", "Analyzer", "History", "Favorites", "Settings"],
            key="navigation"
        )
        
        # YouTube logo
        try:
            logo_url = "https://www.youtube.com/img/desktop/yt_1200.png"
            response = requests.get(logo_url)
            logo = Image.open(BytesIO(response.content))
            st.image(logo, width=150)
        except Exception as e:
            st.warning(f"Couldn't load YouTube logo: {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        <h3 style='margin-bottom: 1rem;'>📱 Features</h3>
        <ul style='list-style-type: none; padding: 0;'>
            <li>📝 AI-powered summaries</li>
            <li>🤖 FLAN-T5 prompt refinement</li>
            <li>📊 Sentiment analysis</li>
            <li>🎧 Audio playback</li>
            <li>📈 Performance metrics</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # History section
        st.markdown("---")
        st.subheader("Recent Summaries")
        if st.session_state.history:
            for i, entry in enumerate(st.session_state.history[-5:]):  # Show last 5 entries
                # Ensure entry has all required fields
                if not isinstance(entry, dict):
                    continue
                    
                # Get values with defaults
                title = entry.get('title', 'Untitled Video')
                timestamp = entry.get('timestamp', datetime.now().isoformat())
                summary = entry.get('summary', 'No summary available')
                video_id = entry.get('video_id', '')
                
                # Only show expander if we have valid data
                if title and timestamp and summary:
                    with st.expander(f"{title} ({datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')})"):
                        st.text_area(
                            "Summary", 
                            value=summary, 
                            height=100, 
                            key=f"history_{video_id}_{i}",  # Added index to make key unique
                            disabled=True
                        )
        else:
            st.write("No history yet")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on navigation
    if nav_selection == "Dashboard":
        create_dashboard()
    elif nav_selection == "Analyzer":
        create_analyzer_view()
    elif nav_selection == "History":
        show_history_view()
    elif nav_selection == "Favorites":
        show_favorites_view()
    elif nav_selection == "Settings":
        show_settings_view()

if __name__ == "__main__":
    main()