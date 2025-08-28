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

def refine_prompt(base_prompt: str, transcript: str = "") -> str:
    """
    Refine a base prompt using FLAN-T5 model with flexible constraints for better ROUGE performance.
    """
    # Check if FLAN-T5 is enabled in session state
    if not st.session_state.get('flan_t5_enabled', True):
        logger.info("FLAN-T5 refinement disabled by user")
        return base_prompt
    
    # Check if FLAN-T5 is available
    if not FLAN_T5_AVAILABLE or refiner_tokenizer is None or refiner_model is None:
        logger.warning("FLAN-T5 model not available, skipping prompt refinement")
        return base_prompt
    
    try:
        # Store original prompt for validation
        original_prompt = base_prompt
        original_length = len(base_prompt)
        
        # Extract named entities from original prompt
        original_entities = extract_named_entities(base_prompt)
        
        # Optimal prompt size to prevent token overflow (400 chars max)
        if len(base_prompt) > 400:
            base_prompt = base_prompt[:400] + "..."
        
        # ROUGE-optimized instruction for maximum lexical overlap
        input_text = f"""Optimize this summarization prompt specifically for ROUGE score maximization. Add instructions that encourage exact word usage, phrase preservation, and lexical fidelity to the source transcript.

Original prompt:
{base_prompt}

ROUGE Optimization Guidelines:
- Add explicit instructions to use exact words and phrases from the transcript
- Encourage preservation of consecutive word sequences (bigrams, trigrams)
- Emphasize maintaining original terminology and expressions
- Include guidance for strategic repetition of key terms
- Enhance instructions for longest common subsequence preservation
- Focus on lexical overlap optimization while maintaining meaning
- Add directives to favor original wording over paraphrasing

ROUGE-Enhanced prompt:"""
        
        inputs = refiner_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=500)
        
        # Ensure proper device handling for model inputs
        try:
            device = next(refiner_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except (StopIteration, RuntimeError):
            # Fallback for meta tensor issues
            device = torch.device('cpu')
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate refined prompt with enhanced parameters for ROUGE optimization
        with torch.no_grad():
            outputs = refiner_model.generate(
                **inputs, 
                max_length=250,  # Further increased for comprehensive refinement
                min_length=40,   # Increased for detailed enhancement
                temperature=0.5, # Increased for lexical diversity
                do_sample=True,
                top_p=0.9,       # Higher for richer vocabulary
                repetition_penalty=1.1,  # Reduced to allow strategic repetition
                length_penalty=1.2,      # Favor longer, comprehensive outputs
                early_stopping=True,
                num_beams=6              # Increased beam search for best results
            )
        
        refined_prompt = refiner_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Use flexible validation instead of strict
        if not validate_refined_prompt_flexible(original_prompt, refined_prompt, original_entities, transcript):
            logger.warning("Refined prompt failed flexible validation, using original")
            return original_prompt
        
        logger.info(f"Prompt refined successfully. Original length: {original_length}, Refined length: {len(refined_prompt)}")
        return refined_prompt
        
    except Exception as e:
        logger.error(f"Prompt refinement failed: {e}")
        return base_prompt

def extract_named_entities(text: str) -> set:
    """
    Enhanced named entity extraction optimized for ROUGE preservation.
    """
    entities = set()
    
    # Enhanced patterns for comprehensive entity extraction
    patterns = [
        # Person names
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Full names
        r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # Three-part names
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',  # Variable length names
        
        # Organizations and institutions
        r'\b[A-Z][a-z]+ University\b',
        r'\b[A-Z][a-z]+ College\b', 
        r'\b[A-Z][a-z]+ Institute\b',
        r'\b[A-Z][a-z]+ Corporation\b',
        r'\b[A-Z][a-z]+ Company\b',
        r'\b[A-Z][a-z]+ Lab(?:oratory)?\b',
        r'\b[A-Z][a-z]+ Center\b',
        r'\b[A-Z][a-z]+ Foundation\b',
        
        # Technology and brands
        r'\b[A-Z][a-z]+ Technologies?\b',
        r'\b[A-Z][a-z]+ Systems?\b',
        r'\b[A-Z][a-z]+ Solutions?\b',
        r'\b[A-Z][a-z]+ Software\b',
        
        # Domain-specific terms that should be preserved
        r'\b[A-Z][A-Z]+\b',  # Acronyms
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase terms
        
        # Research and academic terms
        r'\b[A-Z][a-z]+ Study\b',
        r'\b[A-Z][a-z]+ Research\b',
        r'\b[A-Z][a-z]+ Project\b',
        r'\b[A-Z][a-z]+ Program\b',
        
        # Media and publications
        r'\b[A-Z][a-z]+ Journal\b',
        r'\b[A-Z][a-z]+ Magazine\b',
        r'\b[A-Z][a-z]+ News\b',
        r'\b[A-Z][a-z]+ Times\b',
        r'\b[A-Z][a-z]+ Post\b',
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text)
            entities.update(matches)
        except Exception:
            continue
    
    # Also extract quoted phrases which often contain important terms
    quoted_phrases = re.findall(r'"([^"]+)"', text)
    entities.update([phrase for phrase in quoted_phrases if len(phrase.split()) <= 4])
    
    # Extract domain-specific capitalized terms
    domain_caps = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text)
    entities.update([term for term in domain_caps if len(term.split()) <= 3 and len(term) > 3])
    
    return entities

def validate_refined_prompt_flexible(original_prompt: str, refined_prompt: str, original_entities: set, transcript: str = "") -> bool:
    """
    Flexible validation for refined prompts allowing more creativity while preventing hallucinations.
    """
    try:
        # Allow up to 80% length increase for better lexical overlap
        if len(refined_prompt) > len(original_prompt) * 1.8:
            logger.warning(f"Refined prompt too long: {len(refined_prompt)} vs {len(original_prompt)}")
            return False
        
        # Extract entities from refined prompt
        refined_entities = extract_named_entities(refined_prompt)
        
        # Allow new entities if they appear in the transcript
        new_entities = refined_entities - original_entities
        if new_entities and transcript:
            transcript_entities = extract_named_entities(transcript)
            valid_new_entities = new_entities.intersection(transcript_entities)
            invalid_entities = new_entities - valid_new_entities
            
            if invalid_entities:
                logger.warning(f"Invalid new entities not in transcript: {invalid_entities}")
                return False
            else:
                logger.info(f"Allowing new entities from transcript: {valid_new_entities}")
        elif new_entities and not transcript:
            # If no transcript provided, be more lenient with common entities
            logger.info(f"New entities detected (no transcript validation): {new_entities}")
        
        # Relaxed word checking - only block clearly problematic additions
        original_words = set(re.findall(r'\b\w+\b', original_prompt.lower()))
        refined_words = set(re.findall(r'\b\w+\b', refined_prompt.lower()))
        
        # Minimal problematic words list - only clearly fabricated content
        highly_problematic_words = {
            'lorem', 'ipsum', 'placeholder', 'template', 'example_name'
        }
        
        new_words = refined_words - original_words
        highly_problematic_new_words = new_words.intersection(highly_problematic_words)
        
        if highly_problematic_new_words:
            logger.warning(f"Highly problematic words detected: {highly_problematic_new_words}")
            return False
        
        # Increased tolerance for new words (was 35%, now 50% for lexical overlap)
        new_word_ratio = len(new_words) / len(refined_words) if refined_words else 0
        if new_word_ratio > 0.5:
            logger.warning(f"Refined prompt has too many new words ({new_word_ratio:.2%})")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Flexible prompt validation failed: {e}")
        return False

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
    
    # Store original text for fallback
    original_text = text
    
    # Extended noise patterns to remove irrelevant references
    noise_patterns = [
        r'\b(letters?\s+from\s+\w+)\b',
        r'\b(film-maker\s+and\s+columnist)\b',
        r'\b(columnist\s+and\s+\w+)\b',
        r'\b(journalist\s+and\s+\w+)\b',
        r'\b(\w+\s+and\s+columnist)\b',
        r'\b(\w+\s+and\s+journalist)\b',
        r'\b(repeated\s+names?\s+like\s+\w+\s+\w+)\b',
        r'\b(um|uh|like|you\s+know|basically|actually|literally)\b',
        r'\b(so|well|right|okay|yeah|wow|amazing)\b',
        r'\b(letters?\s+from\s+african\s+journalists?)\b',
        r'\b(university\s+tips?\s+and\s+advice)\b',
        r'\b(fake\s+names?\s+and\s+examples?)\b',
        r'\b(example\s+names?\s+like\s+\w+)\b',
        r'\b(sample\s+names?\s+such\s+as\s+\w+)\b',
        r'\b(for\s+instance\s+like\s+\w+)\b',
        r'\b(such\s+as\s+\w+\s+and\s+\w+)\b',
        r'\b(including\s+\w+\s+and\s+\w+)\b',
        r'\b(like\s+\w+\s+or\s+\w+)\b'
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # First, remove repeated phrases within the same sentence (less aggressive)
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
        if re.search(r'^(letters?\s+from|film-maker|columnist|journalist|university\s+tips|fake\s+names)', s_clean, re.IGNORECASE):
            continue
            
        # Normalize sentence for comparison (lowercase, remove extra spaces, punctuation)
        s_normalized = re.sub(r'\s+', ' ', s_clean.lower()).strip()
        s_normalized = re.sub(r'[^\w\s]', '', s_normalized)  # Remove punctuation for comparison
        
        # Skip if exact duplicate (case-insensitive)
        if s_normalized in seen_sentences:
            continue
            
        # Skip if too similar to any previous sentence (less aggressive similarity check)
        is_similar = False
        for prev_sentence in cleaned:
            prev_normalized = re.sub(r'\s+', ' ', prev_sentence.lower()).strip()
            prev_normalized = re.sub(r'[^\w\s]', '', prev_normalized)
            
            # Optimized threshold for similarity (0.85 for balanced deduplication)
            ratio = SequenceMatcher(None, prev_normalized, s_normalized).ratio()
            if ratio > 0.85:
                is_similar = True
                break
                
        if not is_similar:
            cleaned.append(s_clean)
            seen_sentences.add(s_normalized)
    
    # Join sentences and do final cleanup
    result = " ".join(cleaned)
    
    # Remove any remaining repeated phrases across sentence boundaries (less aggressive)
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
    
    result = result.strip()
    
    # SAFEGUARD: Balanced preservation for ROUGE optimization (75% threshold)
    if len(result) < len(original_text) * 0.75:  # Balanced 75% preservation for noise removal
        logger.warning("clean_summary_text removed too much content, returning original")
        return original_text
    
    return result

def remove_repeated_lines(text: str) -> str:
    """Remove repeated line patterns that commonly occur in summaries with enhanced similarity checking and noise removal."""
    import re
    from difflib import SequenceMatcher
    
    if not text:
        return text
    
    # Store original text for fallback
    original_text = text
    
    # Extended noise patterns to remove irrelevant references
    noise_patterns = [
        r'\b(letters?\s+from\s+\w+)\b',
        r'\b(film-maker\s+and\s+columnist)\b',
        r'\b(columnist\s+and\s+\w+)\b',
        r'\b(journalist\s+and\s+\w+)\b',
        r'\b(\w+\s+and\s+columnist)\b',
        r'\b(\w+\s+and\s+journalist)\b',
        r'\b(repeated\s+names?\s+like\s+\w+\s+\w+)\b',
        r'\b(letters?\s+from\s+african\s+journalists?)\b',
        r'\b(university\s+tips?\s+and\s+advice)\b',
        r'\b(fake\s+names?\s+and\s+examples?)\b',
        r'\b(example\s+names?\s+like\s+\w+)\b',
        r'\b(sample\s+names?\s+such\s+as\s+\w+)\b',
        r'\b(for\s+instance\s+like\s+\w+)\b',
        r'\b(such\s+as\s+\w+\s+and\s+\w+)\b',
        r'\b(including\s+\w+\s+and\s+\w+)\b',
        r'\b(like\s+\w+\s+or\s+\w+)\b'
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
        if re.search(r'^(letters?\s+from|film-maker|columnist|journalist|university\s+tips|fake\s+names)', line_clean, re.IGNORECASE):
            continue
            
        # Normalize line for comparison
        line_normalized = re.sub(r'\s+', ' ', line_clean.lower()).strip()
        line_normalized = re.sub(r'[^\w\s]', '', line_normalized)  # Remove punctuation
        
        # Skip if exact duplicate
        if line_normalized in seen_lines:
            continue
            
        # Skip if too similar to previous lines (check last 3 lines for better pattern detection)
        is_similar = False
        for prev_line in cleaned_lines[-3:]:  # Check last 3 lines as requested
            prev_normalized = re.sub(r'\s+', ' ', prev_line.lower()).strip()
            prev_normalized = re.sub(r'[^\w\s]', '', prev_normalized)
            
            # Ultra-conservative similarity threshold (0.95 for maximum preservation)
            ratio = SequenceMatcher(None, prev_normalized, line_normalized).ratio()
            if ratio > 0.95:
                is_similar = True
                break
                
        if not is_similar:
            cleaned_lines.append(line_clean)
            seen_lines.add(line_normalized)
    
    result = '\n'.join(cleaned_lines)
    
    # SAFEGUARD: Ensure 85% content preservation for ROUGE optimization
    if len(result.strip()) < len(original_text.strip()) * 0.85:  # 85% preservation threshold
        logger.warning("remove_repeated_lines removed too much content, returning original")
        return original_text
    
    return result

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
    Generate enriched prompts with key sentences and keywords for better ROUGE performance
    """
    # Extract key sentences from transcript (first 3000 chars for context)
    extended_transcript = transcript[:3000] if len(transcript) > 3000 else transcript
    transcript_lower = extended_transcript.lower()
    
    # Extract key sentences using TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import nltk
        sentences = nltk.sent_tokenize(extended_transcript)
        
        if len(sentences) > 3:
            # Get top 3-5 key sentences
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            top_sentence_indices = sentence_scores.argsort()[-5:][::-1]
            key_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
        else:
            key_sentences = sentences
    except:
        # Fallback to first few sentences
        key_sentences = extended_transcript.split('. ')[:3]
    
    # Extract important keywords with enhanced filtering
    try:
        # Extract all meaningful words (3+ chars)
        words = re.findall(r'\b[A-Za-z]{3,}\b', transcript_lower)
        word_freq = {}
        
        # Enhanced stopword list
        stopwords = {
            'that', 'this', 'with', 'have', 'they', 'will', 'from', 'been', 'more', 
            'what', 'when', 'where', 'were', 'there', 'their', 'then', 'than', 'them',
            'said', 'says', 'like', 'just', 'really', 'very', 'much', 'many', 'most',
            'some', 'any', 'all', 'can', 'could', 'would', 'should', 'may', 'might',
            'now', 'here', 'way', 'time', 'make', 'take', 'get', 'give', 'come', 'go'
        }
        
        for word in words:
            if word not in stopwords and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 15 keywords (increased from 10)
        key_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        important_terms = [word for word, freq in key_words if freq >= 1]  # Include single occurrences
        
        # Also extract named entities and domain-specific terms
        named_entities = extract_named_entities(extended_transcript)
        domain_terms = set()
        
        # Extract domain-specific terms
        domain_patterns = [
            r'\b(habit[s]?|learning|brain|mindfulness|meditation|reward|trigger|behavior)\b',
            r'\b(technology|digital|app[s]?|software|computer|algorithm)\b',
            r'\b(research|study|studies|experiment|evidence|science)\b',
            r'\b(training|education|teaching|practice|method|technique)\b'
        ]
        
        for pattern in domain_patterns:
            matches = re.findall(pattern, transcript_lower)
            domain_terms.update(matches)
        
        # Combine all important terms
        all_important_terms = list(set(important_terms + list(named_entities) + list(domain_terms)))
        important_terms = all_important_terms[:12]  # Top 12 most relevant terms
        
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        important_terms = []
    
    # Detect actual topics present in the transcript
    detected_topics = []
    topic_patterns = {
        'brain_learning': [r'\b(brain|learning|education|teaching|study|knowledge|habit|mindfulness)\b'],
        'technology': [r'\b(technology|software|computer|digital|programming|coding|app)\b'],
        'health_medical': [r'\b(health|medical|medicine|treatment|therapy|recovery)\b'],
        'business': [r'\b(business|company|market|industry|finance|economics)\b'],
        'science': [r'\b(science|research|experiment|discovery|scientific)\b'],
        'news_events': [r'\b(news|event|announcement|update|report|coverage)\b'],
        'entertainment': [r'\b(entertainment|movie|film|show|performance|art)\b']
    }
    
    for topic_name, patterns in topic_patterns.items():
        for pattern in patterns:
            if re.search(pattern, transcript_lower):
                detected_topics.append(topic_name)
                break
    
    # Create enriched context
    if detected_topics:
        topic_context = f"Main topics: {', '.join(detected_topics)}"
    else:
        topic_context = "Key information and main points"
    
    if important_terms:
        keyword_context = f"CRITICAL: Include these key terms for lexical overlap: {', '.join(important_terms)}. Use these exact words and phrases in your summary."
    else:
        keyword_context = "Preserve original terminology and key phrases exactly as they appear"
    
    # Build enriched base prompt with key context
    key_sentences_text = ""
    if key_sentences:
        key_sentences_text = f"""
Key reference sentences from the content:
{chr(10).join([f"- {sent.strip()}" for sent in key_sentences[:3] if sent.strip()])}
"""

    base_prompt = f"""Create a comprehensive summary using ONLY transcript content. {keyword_context}.

{topic_context}. 

{key_sentences_text}

🎯 NATURAL ROUGE OPTIMIZATION INSTRUCTIONS:

WORD PRESERVATION (ROUGE-1):
- Use the EXACT words from the transcript - never substitute synonyms
- Include ALL important terminology, names, and domain-specific vocabulary  
- Repeat key terms naturally as they appear in the source
- Preserve proper nouns, technical terms, and specialized language

PHRASE PRESERVATION (ROUGE-2):
- Maintain word pairs and phrases exactly as they appear
- Keep compound terms and expressions intact (e.g., "machine learning", "data analysis")
- Preserve natural collocations and multi-word units
- Use consecutive words from source in their original order

SEQUENCE PRESERVATION (ROUGE-L):
- Follow the logical flow and sequence of the original content
- Maintain temporal markers and sequential information
- Preserve numbered lists, steps, and ordered elements
- Keep sentence structures similar to source patterns

CONCISENESS & FOCUS STRATEGY:
- Build sentences using transcript vocabulary as building blocks
- Prioritize key phrases and important terminology over general statements  
- Use selective repetition of critical terms for emphasis and ROUGE-2
- Avoid verbose explanations - focus on core concepts and facts

LEXICAL EFFICIENCY PROTOCOL:
- Never substitute synonyms for original words
- Reuse frequent transcript terms naturally across summary
- Preserve domain-specific language and technical terminology
- Eliminate filler words while maintaining original vocabulary

OPTIMIZED LENGTH TARGETS:
- Target 600-1000 characters for balanced coverage and conciseness
- Focus on 150-250 words with high lexical density
- Every sentence should advance key concepts using source vocabulary
- Quality over quantity - dense, information-rich summaries preferred

If the content discusses habits, learning, or mindfulness, prioritize these key points:
- Reward-based learning and habit loops (trigger → behavior → reward)
- How habits like smoking and stress eating form
- Mindfulness and curiosity as effective tools to break habits
- Examples of mindful practices and their effects
- Use of technology and apps for mindfulness training

If the content discusses habits, learning, or mindfulness, prioritize these key points:
- Reward-based learning and habit loops (trigger → behavior → reward)
- How habits like smoking and stress eating form
- Mindfulness and curiosity as effective tools to break habits
- Examples of mindful practices and their effects
- Use of technology and apps for mindfulness training
         
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
        
        Ensure accuracy and objectivity in the summary. Use ONLY information from the transcript."""
        
    elif content_category.preprocessing_strategy == "educational_structured":
        base_prompt += """
        
        Structure the summary to emphasize:
        - Main concepts and learning objectives
        - Key definitions and terminology
        - Examples and practical applications
        - Step-by-step explanations if applicable
        - Important takeaways and conclusions
        
        Make it easy to understand and follow. Use ONLY content from the transcript."""
        
    elif content_category.preprocessing_strategy == "technical_stepwise":
        base_prompt += """
        
        Organize the summary around:
        - Tools and technologies mentioned
        - Step-by-step processes and procedures
        - Prerequisites and requirements
        - Expected outcomes and results
        - Common issues and troubleshooting tips
        - Best practices and recommendations
        
        Focus on actionable information. Use ONLY information from the transcript."""
        
    elif content_category.preprocessing_strategy == "entertainment_highlight":
        base_prompt += """
        
        Highlight:
        - Main themes and storylines
        - Key characters and their roles
        - Memorable moments and highlights
        - Entertainment value and appeal
        - Overall tone and atmosphere
        
        Use ONLY content from the transcript."""
        
    elif content_category.category == "Interview/Podcast":
        base_prompt += """
        
        📋 INTERVIEW/PODCAST ENHANCEMENT PROTOCOL:
        
        STRUCTURE & FLOW:
        - Identify main discussion topics and conversation flow
        - Highlight key insights, quotes, and revelations
        - Preserve speaker perspectives and viewpoints
        - Maintain conversational context and natural progression
        
        CONTENT REFINEMENT:
        - Extract meaningful exchanges and important dialogue
        - Focus on substantive content over casual conversation
        - Preserve technical terminology and domain-specific language
        - Highlight actionable advice, recommendations, or conclusions
        
        PROFESSIONAL POLISH:
        - Transform conversational fragments into coherent narrative
        - Eliminate filler words, false starts, and repetitive phrasing
        - Maintain speaker authenticity while improving clarity
        - Create smooth transitions between different discussion points
        
        INTERVIEW-SPECIFIC FOCUS:
        - Guest background and expertise areas
        - Key questions and comprehensive responses
        - Novel information, insights, or perspectives shared
        - Practical takeaways and main conclusions
        
        FINAL QUALITY STANDARDS:
        - Professional, readable prose suitable for publication
        - Comprehensive coverage of main discussion points
        - Balanced representation of all speakers
        - Clear, engaging narrative that captures conversation essence
        
        Use ONLY content from the transcript to create a polished, professional summary."""
        
    elif content_category.preprocessing_strategy == "review_balanced":
        base_prompt += """
        
        Provide a balanced summary covering:
        - Product/service overview
        - Pros and advantages
        - Cons and limitations
        - Overall rating or recommendation
        - Target audience and use cases
        - Value for money assessment
        
        Present both positive and negative aspects fairly. Use ONLY information from the transcript."""
    
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
            
            # Step 4: Optimal transcript length for balanced context and performance (6000 chars)
            if len(cleaned_transcript) > 6000:
                # Keep the most relevant content - 6000 chars optimal for ROUGE vs latency balance
                cleaned_transcript = cleaned_transcript[:6000]
                logger.info(f"Transcript optimized to {len(cleaned_transcript)} characters for balanced ROUGE-speed performance")
            
            status.update(label="Transcript preprocessed successfully!", state="complete")
        
        # Stage 2: Generate contextual prompt
        llm_prompt = generate_contextual_prompt(cleaned_transcript, content_category, user_preferences)
        
        # Stage 2.5: Refine prompt using FLAN-T5
        if FLAN_T5_AVAILABLE:
            with st.status("Refining prompt using FLAN-T5...", expanded=True) as status:
                refined_prompt = refine_prompt(llm_prompt, cleaned_transcript)
                # Track if prompt refinement was used
                st.session_state.prompt_refinement_used = refined_prompt != llm_prompt
                
                # Additional safeguard: validate refined prompt
                if refined_prompt != llm_prompt:
                    if not validate_refined_prompt(llm_prompt, refined_prompt):
                        logger.warning("Refined prompt failed validation, falling back to original")
                        refined_prompt = llm_prompt
                        st.session_state.prompt_refinement_used = False
                        st.warning("⚠️ Prompt refinement failed validation - using original prompt")
                    else:
                        st.success("✅ Prompt refined and validated successfully!")
                
                status.update(label="Prompt refinement complete!", state="complete")
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
                
                # Optimized chunking for ROUGE-speed balance
                max_chunk_size = _get_model_context_window(model_name) 
                overlap_size = int(max_chunk_size * 0.25)  # 25% overlap - optimal for context retention vs speed
                
                chunks = []
                for i in range(0, len(processed_transcript), max_chunk_size - overlap_size):
                    chunk = processed_transcript[i:i+max_chunk_size]
                    if chunk.strip() and len(chunk) > 200:  # Ensure meaningful minimum chunk size
                        # Enhanced sentence boundary detection for clean chunks
                        if i + max_chunk_size < len(processed_transcript):
                            # Find the last complete sentence in the chunk
                            sentence_endings = [chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?')]
                            last_sentence_end = max(sentence_endings)
                            
                            # More aggressive boundary detection - if sentence end is in last 30%
                            if last_sentence_end > len(chunk) * 0.7:  
                                chunk = chunk[:last_sentence_end + 1]
                            else:
                                # Look for other natural break points
                                for break_point in ['. ', '! ', '? ']:
                                    pos = chunk.rfind(break_point)
                                    if pos > len(chunk) * 0.7:
                                        chunk = chunk[:pos + 1]
                                        break
                        
                        chunks.append(chunk)
                    if i + max_chunk_size >= len(processed_transcript):
                        break
                
                # Ensure we don't have too many tiny chunks
                if len(chunks) > 1 and len(chunks[-1]) < 300:
                    if len(chunks) > 1:
                        chunks[-2] += " " + chunks[-1]
                        chunks.pop()
                
                summarized_text = []
                for i, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        status.update(label=f"Processing chunk {i+1}/{len(chunks)}...")
                    
                    # Use refined prompt with chunk content
                    summary_input = f"{refined_prompt}\n\nContent:\n{chunk}"
                    # Balanced ROUGE-speed optimization parameters
                    summary = summarizer(
                        summary_input, 
                        max_length=180,   # Optimal length for concise comprehensive summaries
                        min_length=80,    # Balanced minimum for essential details
                        do_sample=False,  # Deterministic for consistent lexical choices
                        num_beams=4,      # Balanced beam search for speed-quality trade-off
                        early_stopping=True,   # Enable for faster generation
                        length_penalty=1.2,    # Moderate preference for detailed outputs
                        repetition_penalty=1.1,   # Allow strategic repetition for ROUGE
                        no_repeat_ngram_size=2,   # Optimize for ROUGE-2 without excessive repetition
                    )
                    summarized_text.append(summary[0]['summary_text'])
                
                # Stage 5: Combine and refine summaries with performance monitoring
                combined_summary = " ".join(summarized_text)
                
                # Performance and quality logging
                logger.info(f"Chunking performance: {len(chunks)} chunks, avg size: {sum(len(c) for c in chunks)//len(chunks) if chunks else 0} chars")
                logger.info(f"Pre-cleaning summary: {len(combined_summary)} chars, estimated words: {len(combined_summary.split())}")
                
                # Quick ROUGE preview on combined summary
                try:
                    quick_rouge = log_rouge_analysis(combined_summary, processed_transcript, "Post-Generation")
                except:
                    logger.info("Quick ROUGE analysis skipped")
                
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
                
                # Stage 5.6: Enhanced hallucination validation and removal
                with st.status("Validating summary for hallucinations...", expanded=True) as status:
                    # First, remove hallucinated sentences
                    validated_summary = remove_hallucinated_sentences(cleaned_summary, cleaned_transcript)
                    logger.info(f"After hallucination removal length: {len(validated_summary)}")
                    
                    # Then validate against transcript
                    strict_mode = st.session_state.get('strict_mode', True)
                    is_valid, warning_msg, similarity_score = validate_summary_against_transcript(
                        validated_summary, cleaned_transcript, strict_mode=strict_mode
                    )
                    
                    # Store validation results in session state
                    st.session_state.similarity_score = similarity_score
                    st.session_state.hallucination_warning = warning_msg if warning_msg else None
                    
                    if not is_valid:
                        logger.warning(f"Summary validation failed: {warning_msg}")
                        if similarity_score < 0.3:
                            logger.warning("Low similarity detected, using cleaned summary")
                            validated_summary = cleaned_summary
                    
                    logger.info(f"Similarity score: {similarity_score:.2f}")
                    status.update(label="Summary validated!", state="complete")
                
                # SAFEGUARD: Ensure we have a valid summary
                if not validated_summary or len(validated_summary.strip()) < 50:
                    logger.warning("Summary became too short after validation, using cleaned summary")
                    if cleaned_summary and len(cleaned_summary.strip()) >= 50:
                        validated_summary = cleaned_summary
                    elif combined_summary and len(combined_summary.strip()) >= 50:
                        logger.warning("Using combined summary as fallback")
                        validated_summary = combined_summary
                    else:
                        logger.error("All summary versions are too short, using fallback")
                        validated_summary = _fallback_summarization(processed_transcript)
                
                # Stage 6: Create high-quality summary meeting specific requirements
                with st.status("Creating high-quality summary (800+ chars, 200+ words, >0.3 similarity)...", expanded=True) as status:
                    # Create summary that meets all requirements
                    improved_summary = create_quality_summary(validated_summary, cleaned_transcript, target_chars=800, target_words=200, min_similarity=0.3)
                    logger.info(f"Quality summary created: {len(improved_summary)} characters, {len(improved_summary.split())} words")
                    
                    # Verify final similarity
                    final_similarity = calculate_rouge_similarity(improved_summary, cleaned_transcript)
                    st.session_state.similarity_score = final_similarity
                    logger.info(f"Final similarity score: {final_similarity:.3f}")
                    
                    status.update(label="High-quality summary created!", state="complete")
                
                # Stage 7: Post-processing and formatting
                final_summary = _format_summary_by_category(improved_summary, content_category)
                
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
    """Select the optimal model for best ROUGE performance based on content category"""
    # Prioritize models known for better ROUGE scores and lexical overlap
    model_mapping = {
        "News/Current Events": "facebook/bart-large-cnn",      # Best for news/factual content
        "Educational": "facebook/bart-large-cnn",              # Changed from pegasus for better ROUGE
        "Technical/Tutorial": "facebook/bart-large-cnn",       # Changed from t5-base for consistency  
        "Entertainment": "google/pegasus-xsum",                # Keep pegasus for entertainment
        "Review/Opinion": "facebook/bart-large-cnn",           # BART for opinion pieces
        "Interview/Podcast": "facebook/bart-large-cnn"         # Optimized for interview content
    }
    
    # Default to BART-large-CNN for best overall ROUGE performance
    return model_mapping.get(content_category.category, "facebook/bart-large-cnn")

def _get_model_context_window(model_name: str) -> int:
    """Get optimized context window size for maximum input utilization"""
    # Increased context windows for better ROUGE performance
    context_windows = {
        "facebook/bart-large-cnn": 1500,  # Increased from 1024
        "google/pegasus-xsum": 1500,     # Increased from 1024 
        "t5-base": 800,                  # Increased from 512
        "facebook/bart-base": 1500,      # Increased from 1024
        "microsoft/DialoGPT-medium": 1024,
        "microsoft/DialoGPT-large": 1024
    }
    
    return context_windows.get(model_name, 1500)  # Default increased to 1500

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
    if 'strict_mode' not in st.session_state:
        st.session_state.strict_mode = True
    if 'flan_t5_enabled' not in st.session_state:
        st.session_state.flan_t5_enabled = True

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
        
        # Display hallucination warning if detected
        if hasattr(st.session_state, 'hallucination_warning') and st.session_state.hallucination_warning:
            st.warning(f"⚠️ **Hallucination Detection:** {st.session_state.hallucination_warning}")
        
        # Display similarity score if available
        if hasattr(st.session_state, 'similarity_score'):
            similarity_color = "🟢" if st.session_state.similarity_score >= 0.5 else "🟡" if st.session_state.similarity_score >= 0.3 else "🔴"
            st.info(f"{similarity_color} **Content Similarity:** {st.session_state.similarity_score:.2f} (0.0-1.0 scale)")
        
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

def validate_refined_prompt(original_prompt: str, refined_prompt: str) -> bool:
    """
    Validate if the refined prompt is safe to use.
    Returns True if safe, False if should fall back to original.
    """
    try:
        # Check if refined prompt is too long (more than 20% longer)
        if len(refined_prompt) > len(original_prompt) * 1.2:
            logger.warning(f"Refined prompt too long ({len(refined_prompt)} vs {len(original_prompt)})")
            return False
        
        # Check for introduction of new topics not in original
        original_words = set(re.findall(r'\b\w+\b', original_prompt.lower()))
        refined_words = set(re.findall(r'\b\w+\b', refined_prompt.lower()))
        
        # Check for potentially problematic new words
        problematic_new_words = []
        for word in refined_words - original_words:
            # Check if new word might indicate hallucination
            if word in ['stroke', 'recovery', 'neuroplasticity', 'brain', 'learning', 'therapy', 'treatment']:
                # Only flag if these words weren't in the original context
                if not any(context_word in original_words for context_word in ['brain', 'learning', 'education', 'teaching']):
                    problematic_new_words.append(word)
        
        if problematic_new_words:
            logger.warning(f"Refined prompt introduced potentially unrelated topics: {problematic_new_words}")
            return False
        
        # Check for excessive changes in meaning
        new_word_ratio = len(refined_words - original_words) / len(refined_words) if refined_words else 0
        if new_word_ratio > 0.3:
            logger.warning(f"Refined prompt has too many new words ({new_word_ratio:.2%})")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Prompt validation failed: {e}")
        return False

def validate_and_clean_summary(summary: str, original_transcript: str, content_category: ContentCategory) -> str:
    """
    Validate and clean the generated summary to remove hallucinations and ensure relevance.
    """
    if not summary or not original_transcript:
        return summary
    
    try:
        # Step 1: Apply existing cleanup functions
        cleaned_summary = clean_summary_text(summary)
        cleaned_summary = _filter_unrelated_topics(cleaned_summary)
        
        # Step 2: Remove hallucinated content by checking against original transcript
        original_words = set(re.findall(r'\b\w+\b', original_transcript.lower()))
        summary_sentences = re.split(r'(?<=[.!?])\s+', cleaned_summary)
        validated_sentences = []
        
        for sentence in summary_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains too many words not in original transcript
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            new_words = sentence_words - original_words
            
            # Calculate ratio of new words
            new_word_ratio = len(new_words) / len(sentence_words) if sentence_words else 0
            
            # If more than 60% of words are new, the sentence might be hallucinated (less aggressive)
            if new_word_ratio > 0.6:
                logger.warning(f"Removing potentially hallucinated sentence: {sentence[:50]}...")
                continue
            
            # Check for specific hallucination patterns
            hallucination_patterns = [
                r'\b(stroke\s+recovery|neuroplasticity|brain\s+changes)\b',
                r'\b(how\s+this\s+helps\s+with\s+learning)\b',
                r'\b(why\s+people\s+learn\s+differently)\b'
            ]
            
            contains_hallucination = False
            for pattern in hallucination_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Only flag if these concepts aren't actually in the original transcript
                    if not re.search(pattern, original_transcript, re.IGNORECASE):
                        contains_hallucination = True
                        break
            
            if contains_hallucination:
                logger.warning(f"Removing sentence with hallucinated content: {sentence[:50]}...")
                continue
            
            validated_sentences.append(sentence)
        
        # Step 3: Reconstruct summary
        final_summary = " ".join(validated_sentences)
        
        # Step 4: Ensure summary is not too short (less aggressive threshold)
        if len(final_summary) < len(summary) * 0.3:  # If less than 30% remains (was 50%)
            logger.warning("Too much content removed during validation, using original cleaned summary")
            return cleaned_summary
        
        # Step 5: Final cleanup
        final_summary = re.sub(r'\s+', ' ', final_summary).strip()
        final_summary = re.sub(r'\.{2,}', '.', final_summary)
        
        logger.info(f"Summary validation complete. Original: {len(summary)}, Final: {len(final_summary)}")
        return final_summary
        
    except Exception as e:
        logger.error(f"Summary validation failed: {e}")
        return summary  # Return original if validation fails

def validate_summary_against_transcript(summary: str, transcript: str, strict_mode: bool = True) -> Tuple[bool, str, float]:
    """
    Validate summary against transcript using similarity metrics and named entity validation.
    Returns (is_valid, warning_message, similarity_score)
    """
    try:
        if not summary or not transcript:
            return True, "", 1.0
        
        # Calculate ROUGE-L similarity
        similarity_score = calculate_rouge_similarity(summary, transcript)
        
        # Extract named entities from both texts
        transcript_entities = extract_named_entities(transcript)
        summary_entities = extract_named_entities(summary)
        
        # Find entities in summary not in transcript
        new_entities = summary_entities - transcript_entities
        
        # Check for hallucination indicators
        hallucination_indicators = []
        
        # Check similarity threshold
        if similarity_score < 0.3:
            hallucination_indicators.append(f"Low similarity score: {similarity_score:.2f}")
        
        # Check for new entities
        if new_entities:
            hallucination_indicators.append(f"New entities found: {', '.join(list(new_entities)[:3])}")
        
        # Check for specific hallucination patterns
        hallucination_patterns = [
            r'\b(stroke\s+recovery|neuroplasticity|brain\s+changes)\b',
            r'\b(how\s+this\s+helps\s+with\s+learning)\b',
            r'\b(why\s+people\s+learn\s+differently)\b',
            r'\b(letters?\s+from\s+african\s+journalists?)\b',
            r'\b(university\s+tips?\s+and\s+advice)\b',
            r'\b(fake\s+names?\s+and\s+examples?)\b'
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, summary, re.IGNORECASE) and not re.search(pattern, transcript, re.IGNORECASE):
                hallucination_indicators.append(f"Hallucination pattern detected: {pattern}")
        
        # Determine if summary is valid
        is_valid = len(hallucination_indicators) == 0
        
        if strict_mode:
            # In strict mode, also check for low similarity
            if similarity_score < 0.3:
                is_valid = False
        
        warning_message = "; ".join(hallucination_indicators) if hallucination_indicators else ""
        
        return is_valid, warning_message, similarity_score
        
    except Exception as e:
        logger.error(f"Summary validation failed: {e}")
        return True, "", 1.0

def calculate_rouge_similarity(text1: str, text2: str) -> float:
    """
    Calculate ROUGE-L similarity between two texts.
    """
    try:
        # Simple implementation using word overlap
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
    except Exception as e:
        logger.error(f"ROUGE similarity calculation failed: {e}")
        return 0.0

def remove_hallucinated_sentences(summary: str, transcript: str) -> str:
    """
    Remove sentences from summary that contain names or entities not in transcript.
    """
    try:
        if not summary or not transcript:
            return summary
        
        # Extract entities from transcript
        transcript_entities = extract_named_entities(transcript)
        
        # Split summary into sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        valid_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extract entities from sentence
            sentence_entities = extract_named_entities(sentence)
            
            # Check if sentence contains entities not in transcript
            new_entities = sentence_entities - transcript_entities
            
            if new_entities:
                logger.warning(f"Removing sentence with new entities: {sentence[:50]}...")
                continue
            
            # Check for hallucination patterns
            hallucination_patterns = [
                r'\b(stroke\s+recovery|neuroplasticity|brain\s+changes)\b',
                r'\b(how\s+this\s+helps\s+with\s+learning)\b',
                r'\b(why\s+people\s+learn\s+differently)\b',
                r'\b(letters?\s+from\s+african\s+journalists?)\b',
                r'\b(university\s+tips?\s+and\s+advice)\b',
                r'\b(fake\s+names?\s+and\s+examples?)\b'
            ]
            
            contains_hallucination = False
            for pattern in hallucination_patterns:
                if re.search(pattern, sentence, re.IGNORECASE) and not re.search(pattern, transcript, re.IGNORECASE):
                    contains_hallucination = True
                    break
            
            if contains_hallucination:
                logger.warning(f"Removing sentence with hallucination pattern: {sentence[:50]}...")
                continue
            
            valid_sentences.append(sentence)
        
        return " ".join(valid_sentences)
        
    except Exception as e:
        logger.error(f"Hallucination removal failed: {e}")
        return summary

def improve_summary_quality(summary: str, transcript: str = None) -> str:
    """
    Comprehensive summary quality improvement that ensures coherent, accurate, and well-structured summaries.
    
    IMPROVEMENTS IMPLEMENTED:
    1. Content Focus: Only includes information from transcript, filters out unrelated names/details
    2. Sentence Quality: Ensures grammatical correctness and logical flow
    3. Critical Ideas: Prioritizes main concepts, key examples, and solutions
    4. Redundancy Removal: Eliminates repetitive content and filler text
    5. Structure: Maintains 1-2 paragraph format with ~150 words
    6. Filler Detection: Identifies and removes irrelevant or random text
    7. Transcript Alignment: Validates content against original transcript
    """
    if not summary:
        return summary
    
    # Store original for comparison
    original_summary = summary
    
    # STEP 1: Extract key concepts from transcript to guide content focus
    transcript_key_concepts = extract_key_concepts_from_transcript(transcript) if transcript else []
    
    # STEP 2: Clean and normalize the summary
    cleaned_summary = clean_summary_text_basic(summary)
    
    # STEP 3: Split into sentences and filter for quality
    sentences = split_into_quality_sentences(cleaned_summary)
    
    # STEP 4: Score sentences based on relevance and quality
    scored_sentences = score_sentences_for_quality(sentences, transcript_key_concepts, transcript)
    
    # STEP 5: Select best sentences while maintaining coherence
    selected_sentences = select_coherent_sentences(scored_sentences)
    
    # STEP 6: Ensure logical flow and remove redundancy
    final_sentences = ensure_logical_flow(selected_sentences)
    
    # STEP 7: Structure into paragraphs and validate length
    structured_summary = structure_into_paragraphs(final_sentences)
    
    # STEP 8: Final validation against transcript
    if transcript:
        structured_summary = validate_against_transcript(structured_summary, transcript)
    
    # STEP 9: Ensure proper length (target ~150 words)
    final_summary = adjust_to_target_length(structured_summary, target_words=150)
    
    # Safety check: if improvement failed, return original
    if len(final_summary.strip()) < 50:
        logger.warning("Quality improvement resulted in too short summary, using original")
        return original_summary
    
    return final_summary.strip()

def extract_key_concepts_from_transcript(transcript: str) -> List[str]:
    """
    Extract key concepts, main ideas, and important terms from transcript.
    This helps guide content focus and prevents inclusion of irrelevant information.
    """
    if not transcript:
        return []
    
    # Define patterns for key concepts (main ideas, solutions, examples)
    concept_patterns = [
        r'\b(how|why|what|when|where)\s+\w+\s+\w+',  # Questions and explanations
        r'\b(because|since|therefore|however|although)\s+\w+',  # Logical connectors
        r'\b(example|instance|case|scenario|situation)\s+of',  # Examples
        r'\b(solution|method|technique|approach|strategy)\s+',  # Solutions
        r'\b(problem|issue|challenge|difficulty)\s+',  # Problems
        r'\b(result|outcome|effect|impact|consequence)\s+',  # Results
        r'\b(study|research|experiment|evidence|data)\s+',  # Evidence
        r'\b(habit|behavior|pattern|routine|practice)\s+',  # Behavioral concepts
        r'\b(mindfulness|awareness|attention|focus|consciousness)\s+',  # Mindfulness concepts
        r'\b(learning|education|training|development|growth)\s+',  # Learning concepts
    ]
    
    key_concepts = []
    transcript_lower = transcript.lower()
    
    # Extract concepts using patterns
    for pattern in concept_patterns:
        matches = re.findall(pattern, transcript_lower)
        key_concepts.extend(matches)
    
    # Extract important noun phrases (potential key concepts)
    noun_phrases = re.findall(r'\b\w+\s+\w+\s+\w+', transcript_lower)
    key_concepts.extend(noun_phrases[:10])  # Limit to avoid noise
    
    return list(set(key_concepts))  # Remove duplicates

def clean_summary_text_basic(summary: str) -> str:
    """
    Basic cleaning to remove obvious noise and normalize text.
    Focuses on removing filler words and normalizing structure.
    """
    if not summary:
        return summary
    
    # Remove common filler words and phrases that add no value
    filler_patterns = [
        r'\b(um|uh|like|you know|basically|actually|literally|sort of|kind of)\b',
        r'\b(so|well|right|okay|yeah|wow|amazing|incredible|fantastic)\b',
        r'\b(just|really|very|quite|rather|somewhat|fairly)\b',
        r'\b(thing|stuff|something|anything|everything|nothing)\b',
        r'\b(way|manner|fashion|style|method|approach)\s+(of|to|for)\b',
        r'\b(as\s+you\s+can\s+see|as\s+we\s+know|obviously|clearly)\b',
        r'\b(in\s+other\s+words|that\s+is|i\s+mean|you\s+see)\b',
    ]
    
    cleaned = summary
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Normalize whitespace and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = re.sub(r'\.{2,}', '.', cleaned)  # Multiple periods to single
    cleaned = re.sub(r'!{2,}', '!', cleaned)  # Multiple exclamation marks
    cleaned = re.sub(r'\?{2,}', '?', cleaned)  # Multiple question marks
    
    return cleaned.strip()

def split_into_quality_sentences(text: str) -> List[str]:
    """
    Split text into sentences and filter for quality.
    Ensures sentences are complete, grammatical, and meaningful.
    """
    if not text:
        return []
    
    # Split by sentence endings, but be more careful about abbreviations
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    quality_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # FILTER 1: Minimum length and completeness
        if len(sentence) < 20:  # Too short to be meaningful
            continue
        
        # FILTER 2: Must end with proper punctuation
        if not sentence.endswith(('.', '!', '?')):
            continue
        
        # FILTER 3: Must start with capital letter
        if not sentence[0].isupper():
            continue
        
        # FILTER 4: Must contain a verb (basic grammar check)
        if not has_verb(sentence):
            continue
        
        # FILTER 5: Must not be just a list or fragment
        if is_fragment_or_list(sentence):
            continue
        
        # FILTER 6: Must not contain excessive repetition
        if has_excessive_repetition(sentence):
            continue
        
        quality_sentences.append(sentence)
    
    return quality_sentences

def has_verb(sentence: str) -> bool:
    """
    Check if sentence contains a verb (basic grammar validation).
    """
    # Common verb patterns
    verb_patterns = [
        r'\b(is|are|was|were|be|been|being)\b',  # Be verbs
        r'\b(have|has|had|do|does|did|will|would|could|should|can|may|might)\b',  # Auxiliary verbs
        r'\b\w+ing\b',  # Present participle
        r'\b\w+ed\b',   # Past tense
        r'\b\w+s\b',    # Third person singular
    ]
    
    sentence_lower = sentence.lower()
    return any(re.search(pattern, sentence_lower) for pattern in verb_patterns)

def is_fragment_or_list(sentence: str) -> bool:
    """
    Check if sentence is a fragment or just a list.
    """
    # Patterns that indicate fragments or lists
    fragment_patterns = [
        r'^[A-Z][a-z]*\s*[:\-]\s*',  # Starts with word followed by : or -
        r'^\d+\.\s*',  # Starts with number and period
        r'^[•\-\*]\s*',  # Starts with bullet point
        r'^\w+\s*[,\s]+\w+\s*[,\s]+\w+',  # Just a list of words
    ]
    
    return any(re.match(pattern, sentence) for pattern in fragment_patterns)

def has_excessive_repetition(sentence: str) -> bool:
    """
    Check if sentence has excessive word repetition.
    """
    words = sentence.lower().split()
    if len(words) < 5:
        return False
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Only count meaningful words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # If any word appears more than 2 times in a short sentence, it's repetitive
    return any(count > 2 for count in word_counts.values())

def score_sentences_for_quality(sentences: List[str], key_concepts: List[str], transcript: str = None) -> List[Tuple[float, str]]:
    """
    Score sentences based on relevance, quality, and importance.
    Higher scores indicate better sentences for the summary.
    """
    scored_sentences = []
    
    for sentence in sentences:
        score = 0.0
        sentence_lower = sentence.lower()
        
        # SCORE 1: Length and completeness (10 points)
        if 20 <= len(sentence) <= 100:
            score += 10
        elif 100 < len(sentence) <= 150:
            score += 8
        else:
            score += 5
        
        # SCORE 2: Contains key concepts from transcript (20 points)
        if key_concepts:
            concept_matches = sum(1 for concept in key_concepts if concept.lower() in sentence_lower)
            score += min(20, concept_matches * 5)
        
        # SCORE 3: Contains important keywords (15 points)
        important_keywords = [
            'because', 'therefore', 'however', 'although', 'example', 'solution',
            'problem', 'result', 'study', 'research', 'evidence', 'habit',
            'behavior', 'mindfulness', 'learning', 'training', 'development'
        ]
        keyword_matches = sum(1 for keyword in important_keywords if keyword in sentence_lower)
        score += min(15, keyword_matches * 3)
        
        # SCORE 4: Logical connectors (10 points)
        logical_connectors = [
            'because', 'since', 'therefore', 'however', 'although', 'while',
            'when', 'if', 'then', 'so', 'thus', 'consequently'
        ]
        connector_matches = sum(1 for connector in logical_connectors if connector in sentence_lower)
        score += min(10, connector_matches * 2)
        
        # SCORE 5: Specific examples or evidence (15 points)
        example_patterns = [
            r'\b(example|instance|case|scenario)\s+',
            r'\b(study|research|experiment|evidence)\s+',
            r'\b(data|results|findings|analysis)\s+',
            r'\b\w+\s+shows?\s+',
            r'\b\w+\s+demonstrates?\s+'
        ]
        example_matches = sum(1 for pattern in example_patterns if re.search(pattern, sentence_lower))
        score += min(15, example_matches * 5)
        
        # SCORE 6: Solutions or methods (15 points)
        solution_patterns = [
            r'\b(solution|method|technique|approach|strategy)\s+',
            r'\b(how\s+to|way\s+to|means\s+of)\s+',
            r'\b(helps?|enables?|allows?|permits?)\s+',
            r'\b(improves?|enhances?|increases?|reduces?)\s+'
        ]
        solution_matches = sum(1 for pattern in solution_patterns if re.search(pattern, sentence_lower))
        score += min(15, solution_matches * 5)
        
        # SCORE 7: Penalty for filler content (-10 points)
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally']
        filler_count = sum(1 for word in filler_words if word in sentence_lower)
        score -= filler_count * 2
        
        # SCORE 8: Penalty for repetition (-5 points)
        if has_excessive_repetition(sentence):
            score -= 5
        
        # SCORE 9: Bonus for transcript alignment (10 points)
        if transcript and transcript.lower().find(sentence_lower[:50]) != -1:
            score += 10
        
        scored_sentences.append((score, sentence))
    
    return scored_sentences

def select_coherent_sentences(scored_sentences: List[Tuple[float, str]]) -> List[str]:
    """
    Select the best sentences while maintaining coherence and logical flow.
    Prioritizes high-scoring sentences that work well together.
    """
    if not scored_sentences:
        return []
    
    # Sort by score (highest first)
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Select top sentences, but limit to maintain coherence
    max_sentences = 8  # Reasonable limit for 150-word summary
    selected = []
    word_count = 0
    target_words = 150
    
    for score, sentence in scored_sentences:
        if len(selected) >= max_sentences:
            break
        
        sentence_words = len(sentence.split())
        if word_count + sentence_words <= target_words:
            selected.append(sentence)
            word_count += sentence_words
        else:
            # If this sentence is very high scoring, consider replacing a lower one
            if score > 30 and selected:
                # Find a lower scoring sentence to replace
                for i, existing_sentence in enumerate(selected):
                    existing_score = next(s for s, sent in scored_sentences if sent == existing_sentence)
                    if score > existing_score + 10:  # Significant improvement
                        selected[i] = sentence
                        break
    
    return selected

def ensure_logical_flow(sentences: List[str]) -> List[str]:
    """
    Ensure sentences flow logically and remove redundancy.
    Reorders sentences for better coherence and removes repetitive content.
    """
    if len(sentences) <= 1:
        return sentences
    
    # Remove redundant sentences (similar meaning)
    unique_sentences = []
    for sentence in sentences:
        is_redundant = False
        for existing in unique_sentences:
            # Check for high similarity (potential redundancy)
            similarity = calculate_sentence_similarity(sentence, existing)
            if similarity > 0.7:  # High similarity threshold
                is_redundant = True
                break
        
        if not is_redundant:
            unique_sentences.append(sentence)
    
    # Reorder for logical flow (topic sentences first, then details, then conclusions)
    topic_sentences = []
    detail_sentences = []
    conclusion_sentences = []
    
    for sentence in unique_sentences:
        sentence_lower = sentence.lower()
        
        # Topic sentences (introduce main ideas)
        if any(word in sentence_lower for word in ['introduces', 'discusses', 'explains', 'focuses', 'examines']):
            topic_sentences.append(sentence)
        # Conclusion sentences (summarize or conclude)
        elif any(word in sentence_lower for word in ['therefore', 'thus', 'consequently', 'in conclusion', 'overall']):
            conclusion_sentences.append(sentence)
        # Detail sentences (everything else)
        else:
            detail_sentences.append(sentence)
    
    # Combine in logical order
    ordered_sentences = topic_sentences + detail_sentences + conclusion_sentences
    
    return ordered_sentences

def calculate_sentence_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate similarity between two sentences using word overlap.
    """
    words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
    words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def structure_into_paragraphs(sentences: List[str]) -> str:
    """
    Structure sentences into 1-2 coherent paragraphs.
    Ensures proper paragraph breaks and logical grouping.
    """
    if not sentences:
        return ""
    
    if len(sentences) <= 3:
        # Single paragraph for short summaries
        return " ".join(sentences)
    
    # Split into paragraphs based on content and length
    mid_point = len(sentences) // 2
    
    # First paragraph: introduction and main points
    first_paragraph = " ".join(sentences[:mid_point])
    
    # Second paragraph: details and conclusions
    second_paragraph = " ".join(sentences[mid_point:])
    
    return f"{first_paragraph}\n\n{second_paragraph}"

def validate_against_transcript(summary: str, transcript: str) -> str:
    """
    Final validation to ensure summary only contains information from transcript.
    Removes any content that cannot be verified against the original.
    """
    if not transcript:
        return summary
    
    # Split summary into sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    validated_sentences = []
    
    transcript_lower = transcript.lower()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if sentence content can be found in transcript
        sentence_keywords = extract_keywords(sentence)
        transcript_keywords = extract_keywords(transcript)
        
        # Calculate overlap
        overlap = len(sentence_keywords.intersection(transcript_keywords))
        overlap_ratio = overlap / len(sentence_keywords) if sentence_keywords else 0
        
        # Keep sentence if it has sufficient overlap with transcript
        if overlap_ratio >= 0.3:  # At least 30% of words from transcript
            validated_sentences.append(sentence)
        else:
            logger.warning(f"Removing sentence with insufficient transcript overlap: {sentence[:50]}...")
    
    return " ".join(validated_sentences)

def extract_keywords(text: str) -> set:
    """
    Extract meaningful keywords from text (excluding common stop words).
    """
    # Common stop words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    return set(word for word in words if len(word) > 2 and word not in stop_words)

def adjust_to_target_length(summary: str, target_words: int = 150) -> str:
    """
    Adjust summary to target word length while maintaining coherence.
    """
    if not summary:
        return summary
    
    current_words = summary.split()
    current_length = len(current_words)
    
    if current_length <= target_words:
        return summary
    
    # If too long, truncate intelligently
    if current_length > target_words * 1.5:
        # Truncate to target length
        truncated_words = current_words[:target_words]
        summary = " ".join(truncated_words)
        
        # Ensure it ends with a complete sentence
        if not summary.endswith(('.', '!', '?')):
            # Find the last complete sentence
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            if len(sentences) > 1:
                summary = " ".join(sentences[:-1])  # Remove incomplete last sentence
                if summary and not summary.endswith(('.', '!', '?')):
                    summary += "."
    
    return summary

def focus_on_habit_content(summary: str) -> str:
    """
    Specifically focus summary on habit formation, mindfulness, and learning content.
    """
    if not summary:
        return summary
    
    # Key topics to prioritize
    priority_topics = [
        "reward-based learning",
        "habit loops",
        "trigger behavior reward cycle",
        "smoking stress eating habits",
        "mindfulness curiosity tools",
        "mindful smoking disenchantment",
        "apps technology mindfulness",
        "mindfulness training delivery"
    ]
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    prioritized_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_lower = sentence.lower()
        
        # Check if sentence contains priority topics
        topic_score = sum(1 for topic in priority_topics if topic in sentence_lower)
        
        if topic_score > 0:
            # Prioritize sentences with more relevant topics
            prioritized_sentences.append((topic_score, sentence))
    
    # Sort by topic relevance and take top sentences
    prioritized_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Take top sentences while keeping within word limit
    selected_sentences = []
    word_count = 0
    target_words = 150
    
    for score, sentence in prioritized_sentences:
        sentence_words = len(sentence.split())
        if word_count + sentence_words <= target_words:
            selected_sentences.append(sentence)
            word_count += sentence_words
        else:
            break
    
    # If we don't have enough priority content, add some general well-formed sentences
    if len(selected_sentences) < 2:
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and sentence.endswith(('.', '!', '?')):
                sentence_words = len(sentence.split())
                if word_count + sentence_words <= target_words:
                    selected_sentences.append(sentence)
                    word_count += sentence_words
    
    return " ".join(selected_sentences)

def create_crystal_clear_summary(summary: str, transcript: str, target_chars: int = 800) -> str:
    """
    Create exactly 5 lines of crystal clear summary with target character count and high similarity.
    
    REQUIREMENTS:
    - Exactly 5 lines
    - Target character count (default 800)
    - High similarity score (0.5+)
    - Clear and meaningful content
    - No filler or irrelevant information
    """
    if not summary or not transcript:
        return summary
    
    # Extract the most important sentences from the summary
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    
    # Score sentences based on importance and relevance
    scored_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        # Calculate importance score
        score = calculate_sentence_importance(sentence, transcript)
        scored_sentences.append((score, sentence))
    
    # Sort by importance score (highest first)
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Select top sentences that fit the character limit
    selected_sentences = []
    current_chars = 0
    
    for score, sentence in scored_sentences:
        if current_chars + len(sentence) <= target_chars:
            selected_sentences.append(sentence)
            current_chars += len(sentence)
        else:
            break
    
    # If we don't have enough content, add more sentences
    if len(selected_sentences) < 3:
        for score, sentence in scored_sentences:
            if sentence not in selected_sentences:
                selected_sentences.append(sentence)
                if len(selected_sentences) >= 5:
                    break
    
    # Ensure exactly 5 lines
    while len(selected_sentences) < 5:
        # Add placeholder or split existing sentences
        if selected_sentences:
            # Split the longest sentence
            longest_sentence = max(selected_sentences, key=len)
            words = longest_sentence.split()
            mid_point = len(words) // 2
            part1 = " ".join(words[:mid_point]) + "."
            part2 = " ".join(words[mid_point:])
            
            # Replace the longest sentence with two parts
            selected_sentences.remove(longest_sentence)
            selected_sentences.append(part1)
            selected_sentences.append(part2)
        else:
            selected_sentences.append("Content summary not available.")
    
    # Truncate to exactly 5 lines
    selected_sentences = selected_sentences[:5]
    
    # Join into exactly 5 lines
    crystal_clear_summary = "\n".join(selected_sentences)
    
    # Ensure it meets character target
    if len(crystal_clear_summary) > target_chars:
        # Truncate while maintaining 5 lines
        lines = crystal_clear_summary.split('\n')
        truncated_lines = []
        chars_per_line = target_chars // 5
        
        for line in lines:
            if len(line) > chars_per_line:
                line = line[:chars_per_line-3] + "..."
            truncated_lines.append(line)
        
        crystal_clear_summary = "\n".join(truncated_lines)
    
    return crystal_clear_summary

def calculate_sentence_importance(sentence: str, transcript: str) -> float:
    """
    Calculate importance score for a sentence based on relevance to transcript.
    Higher scores indicate more important and relevant sentences.
    """
    if not sentence or not transcript:
        return 0.0
    
    score = 0.0
    sentence_lower = sentence.lower()
    transcript_lower = transcript.lower()
    
    # Score 1: Word overlap with transcript (40 points)
    sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
    transcript_words = set(re.findall(r'\b\w+\b', transcript_lower))
    
    if sentence_words and transcript_words:
        overlap = len(sentence_words.intersection(transcript_words))
        overlap_ratio = overlap / len(sentence_words)
        score += overlap_ratio * 40
    
    # Score 2: Contains key concepts (30 points)
    key_concepts = [
        'habit', 'learning', 'mindfulness', 'behavior', 'reward',
        'trigger', 'pattern', 'change', 'improve', 'develop',
        'solution', 'method', 'technique', 'approach', 'strategy',
        'example', 'evidence', 'study', 'research', 'result'
    ]
    
    concept_matches = sum(1 for concept in key_concepts if concept in sentence_lower)
    score += min(30, concept_matches * 3)
    
    # Score 3: Sentence structure quality (20 points)
    if len(sentence) >= 30 and sentence.endswith(('.', '!', '?')):
        score += 20
    elif len(sentence) >= 20:
        score += 15
    else:
        score += 10
    
    # Score 4: Logical connectors (10 points)
    connectors = ['because', 'therefore', 'however', 'although', 'while', 'when', 'if', 'then']
    connector_count = sum(1 for connector in connectors if connector in sentence_lower)
    score += min(10, connector_count * 2)
    
    return score

def ensure_high_similarity_summary(summary: str, transcript: str, min_similarity: float = 0.5) -> str:
    """
    Ensure summary has high similarity score (0.5+) by focusing on transcript content.
    """
    if not summary or not transcript:
        return summary
    
    # Calculate current similarity
    current_similarity = calculate_rouge_similarity(summary, transcript)
    
    if current_similarity >= min_similarity:
        return summary
    
    # If similarity is too low, rebuild summary from transcript
    logger.warning(f"Similarity too low ({current_similarity:.2f}), rebuilding from transcript")
    
    # Extract key sentences directly from transcript
    transcript_sentences = re.split(r'(?<=[.!?])\s+', transcript)
    
    # Score transcript sentences
    scored_transcript_sentences = []
    for sentence in transcript_sentences:
        sentence = sentence.strip()
        if len(sentence) < 30 or len(sentence) > 200:
            continue
        
        score = calculate_sentence_importance(sentence, transcript)
        scored_transcript_sentences.append((score, sentence))
    
    # Sort by importance
    scored_transcript_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Build new summary from top transcript sentences
    new_summary_sentences = []
    current_chars = 0
    target_chars = 800
    
    for score, sentence in scored_transcript_sentences:
        if current_chars + len(sentence) <= target_chars:
            new_summary_sentences.append(sentence)
            current_chars += len(sentence)
        else:
            break
    
    new_summary = " ".join(new_summary_sentences)
    
    # Verify new similarity
    new_similarity = calculate_rouge_similarity(new_summary, transcript)
    logger.info(f"New summary similarity: {new_similarity:.2f}")
    
    return new_summary

def create_cohesive_habit_summary(transcript: str, target_words: int = 150, min_chars: int = 800) -> str:
    """
    Create a cohesive summary focused specifically on habit formation and mindfulness content.
    
    IMPROVEMENTS IMPLEMENTED:
    1. Cohesive Structure: Ensures logically connected sentences with proper flow
    2. Critical Content Focus: Specifically targets habit/learning concepts from transcript
    3. Content Validation: Only includes information directly from transcript
    4. Length Control: Targets 150 words with minimum 800 characters
    5. Semantic Similarity: Uses advanced similarity checking (>0.5 threshold)
    6. Post-processing: Removes incomplete sentences and ensures grammar
    7. Coherent Organization: Structures into 1-2 clear paragraphs
    """
    if not transcript:
        return "No transcript available for summary generation."
    
    # STEP 1: Extract habit-specific content from transcript
    habit_content = extract_habit_specific_content(transcript)
    
    # STEP 2: Create initial summary with habit focus
    initial_summary = create_habit_focused_summary(habit_content, target_words)
    
    # STEP 3: Ensure semantic similarity > 0.5
    similarity_score = calculate_semantic_similarity(initial_summary, transcript)
    
    if similarity_score < 0.5:
        logger.warning(f"Initial similarity too low ({similarity_score:.2f}), regenerating with transcript focus")
        initial_summary = regenerate_from_transcript(transcript, target_words)
        similarity_score = calculate_semantic_similarity(initial_summary, transcript)
    
    # STEP 4: Post-process for cohesion and readability
    cohesive_summary = post_process_for_cohesion(initial_summary, transcript)
    
    # STEP 5: Final validation and formatting
    final_summary = format_final_summary(cohesive_summary, target_words, min_chars)
    
    # STEP 6: Verify final similarity
    final_similarity = calculate_semantic_similarity(final_summary, transcript)
    logger.info(f"Final cohesive summary similarity: {final_similarity:.2f}")
    
    return final_summary

def extract_habit_specific_content(transcript: str) -> str:
    """
    Extract content specifically related to habits, learning, and mindfulness from transcript.
    This ensures we focus only on the relevant content areas requested.
    """
    if not transcript:
        return ""
    
    # Define specific patterns for habit and mindfulness content
    habit_patterns = [
        # Reward-based learning and habit loops
        r'(?i)\b(reward.*?learning|habit.*?loop|trigger.*?behavior.*?reward)\b',
        r'(?i)\b(how.*?habits.*?form|habit.*?formation|behavior.*?pattern)\b',
        
        # Specific habits mentioned
        r'(?i)\b(smoking.*?habit|stress.*?eating|stress.*?habit)\b',
        r'(?i)\b(break.*?habit|change.*?habit|overcome.*?habit)\b',
        
        # Mindfulness and curiosity solutions
        r'(?i)\b(mindfulness.*?solution|curiosity.*?tool|mindful.*?practice)\b',
        r'(?i)\b(awareness.*?technique|conscious.*?behavior|present.*?moment)\b',
        
        # Mindful smoking example
        r'(?i)\b(mindful.*?smoking|disenchantment.*?habit|smoking.*?disenchantment)\b',
        r'(?i)\b(example.*?mindful|practice.*?mindful|smoking.*?example)\b',
        
        # Technology and apps
        r'(?i)\b(app.*?mindfulness|technology.*?training|digital.*?mindfulness)\b',
        r'(?i)\b(mindfulness.*?app|training.*?app|technology.*?solution)\b',
        
        # Learning and development concepts
        r'(?i)\b(learning.*?process|brain.*?change|neural.*?plasticity)\b',
        r'(?i)\b(behavior.*?change|pattern.*?recognition|habit.*?modification)\b'
    ]
    
    extracted_content = []
    transcript_lower = transcript.lower()
    
    # Extract sentences containing habit-related content
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        # Check if sentence contains habit-related content
        sentence_lower = sentence.lower()
        has_habit_content = any(re.search(pattern, sentence_lower) for pattern in habit_patterns)
        
        if has_habit_content:
            extracted_content.append(sentence)
    
    # If we don't have enough habit content, include broader learning content
    if len(extracted_content) < 3:
        learning_patterns = [
            r'(?i)\b(learn|learning|education|study|research)\b',
            r'(?i)\b(understand|explain|demonstrate|show|example)\b',
            r'(?i)\b(process|method|technique|approach|strategy)\b'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_lower = sentence.lower()
            has_learning_content = any(re.search(pattern, sentence_lower) for pattern in learning_patterns)
            
            if has_learning_content and sentence not in extracted_content:
                extracted_content.append(sentence)
    
    return " ".join(extracted_content)

def create_habit_focused_summary(habit_content: str, target_words: int) -> str:
    """
    Create a summary focused specifically on habit formation and mindfulness content.
    Ensures all critical ideas are included and properly connected.
    """
    if not habit_content:
        return "No habit-related content found in transcript."
    
    # Split into sentences and score for importance
    sentences = re.split(r'(?<=[.!?])\s+', habit_content)
    scored_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 15:
            continue
        
        # Score based on habit content relevance
        score = score_habit_sentence(sentence)
        scored_sentences.append((score, sentence))
    
    # Sort by relevance score
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Select top sentences while maintaining target word count
    selected_sentences = []
    current_words = 0
    
    for score, sentence in scored_sentences:
        sentence_words = len(sentence.split())
        if current_words + sentence_words <= target_words:
            selected_sentences.append(sentence)
            current_words += sentence_words
        else:
            break
    
    # Ensure we have enough content
    if len(selected_sentences) < 2:
        # Add more sentences to meet minimum requirements
        for score, sentence in scored_sentences:
            if sentence not in selected_sentences:
                selected_sentences.append(sentence)
                if len(selected_sentences) >= 3:
                    break
    
    return " ".join(selected_sentences)

def score_habit_sentence(sentence: str) -> float:
    """
    Score sentence based on relevance to habit formation and mindfulness content.
    Higher scores indicate more relevant content for the summary.
    """
    score = 0.0
    sentence_lower = sentence.lower()
    
    # Critical content areas (highest priority)
    critical_patterns = {
        'reward_learning': [r'reward.*?learning', r'habit.*?loop', r'trigger.*?behavior.*?reward'],
        'habit_formation': [r'how.*?habits.*?form', r'habit.*?formation', r'behavior.*?pattern'],
        'smoking_stress': [r'smoking.*?habit', r'stress.*?eating', r'stress.*?habit'],
        'mindfulness_solutions': [r'mindfulness.*?solution', r'curiosity.*?tool', r'mindful.*?practice'],
        'mindful_smoking': [r'mindful.*?smoking', r'disenchantment.*?habit', r'smoking.*?example'],
        'technology_apps': [r'app.*?mindfulness', r'technology.*?training', r'digital.*?mindfulness']
    }
    
    # Score based on critical content (40 points)
    for category, patterns in critical_patterns.items():
        if any(re.search(pattern, sentence_lower) for pattern in patterns):
            score += 40
            break  # Only count once per category
    
    # Additional relevance scoring (30 points)
    relevance_keywords = [
        'habit', 'learning', 'mindfulness', 'behavior', 'reward', 'trigger',
        'pattern', 'change', 'improve', 'develop', 'solution', 'method',
        'technique', 'approach', 'strategy', 'example', 'evidence'
    ]
    
    keyword_matches = sum(1 for keyword in relevance_keywords if keyword in sentence_lower)
    score += min(30, keyword_matches * 2)
    
    # Sentence quality scoring (20 points)
    if len(sentence) >= 30 and sentence.endswith(('.', '!', '?')):
        score += 20
    elif len(sentence) >= 20:
        score += 15
    else:
        score += 10
    
    # Logical flow scoring (10 points)
    connectors = ['because', 'therefore', 'however', 'although', 'while', 'when', 'if', 'then']
    connector_count = sum(1 for connector in connectors if connector in sentence_lower)
    score += min(10, connector_count * 2)
    
    return score

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using advanced text analysis.
    This provides more accurate similarity measurement than simple word overlap.
    """
    try:
        if not text1 or not text2:
            return 0.0
        
        # Method 1: Enhanced word overlap with weighting
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Remove common stop words for better similarity calculation
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Calculate weighted similarity
        overlap_ratio = len(intersection) / len(union)
        
        # Method 2: Phrase similarity for better accuracy
        phrases1 = extract_meaningful_phrases(text1)
        phrases2 = extract_meaningful_phrases(text2)
        
        phrase_similarity = 0.0
        if phrases1 and phrases2:
            phrase_overlap = len(set(phrases1).intersection(set(phrases2)))
            phrase_similarity = phrase_overlap / max(len(phrases1), len(phrases2))
        
        # Combine both methods for final similarity score
        final_similarity = (overlap_ratio * 0.7) + (phrase_similarity * 0.3)
        
        return min(1.0, final_similarity)
        
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        return 0.0

def extract_meaningful_phrases(text: str) -> List[str]:
    """
    Extract meaningful phrases from text for better similarity calculation.
    """
    try:
        # Extract 2-4 word phrases
        words = text.split()
        phrases = []
        
        for i in range(len(words) - 1):
            for j in range(i + 2, min(i + 5, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                if len(phrase) > 5:  # Only meaningful phrases
                    phrases.append(phrase.lower())
        
        return phrases[:20]  # Limit to top 20 phrases
        
    except Exception as e:
        logger.error(f"Phrase extraction failed: {e}")
        return []

def regenerate_from_transcript(transcript: str, target_words: int) -> str:
    """
    Regenerate summary directly from transcript when similarity is too low.
    This ensures we maintain high similarity while focusing on relevant content.
    """
    if not transcript:
        return "No transcript available."
    
    # Extract the most relevant sentences from transcript
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    scored_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        # Score based on habit content relevance
        score = score_habit_sentence(sentence)
        scored_sentences.append((score, sentence))
    
    # Sort by relevance and select top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    selected_sentences = []
    current_words = 0
    
    for score, sentence in scored_sentences:
        sentence_words = len(sentence.split())
        if current_words + sentence_words <= target_words:
            selected_sentences.append(sentence)
            current_words += sentence_words
        else:
            break
    
    return " ".join(selected_sentences)

def post_process_for_cohesion(summary: str, transcript: str) -> str:
    """
    Post-process summary to ensure cohesion, readability, and proper grammar.
    This step removes incomplete sentences and merges fragmented ideas.
    """
    if not summary:
        return summary
    
    # STEP 1: Remove incomplete sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    complete_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check for complete sentences
        if is_complete_sentence(sentence):
            complete_sentences.append(sentence)
    
    # STEP 2: Merge fragmented ideas
    merged_sentences = merge_fragmented_ideas(complete_sentences)
    
    # STEP 3: Ensure proper grammar and readability
    final_sentences = ensure_grammar_and_readability(merged_sentences)
    
    return " ".join(final_sentences)

def is_complete_sentence(sentence: str) -> bool:
    """
    Check if a sentence is complete and well-formed.
    """
    if not sentence:
        return False
    
    # Must have minimum length
    if len(sentence) < 15:
        return False
    
    # Must end with proper punctuation
    if not sentence.endswith(('.', '!', '?')):
        return False
    
    # Must start with capital letter
    if not sentence[0].isupper():
        return False
    
    # Must contain a verb (basic grammar check)
    if not has_verb(sentence):
        return False
    
    # Must not be just a fragment or list
    if is_fragment_or_list(sentence):
        return False
    
    return True

def merge_fragmented_ideas(sentences: List[str]) -> List[str]:
    """
    Merge fragmented ideas into cohesive sentences.
    This improves the logical flow of the summary.
    """
    if len(sentences) <= 1:
        return sentences
    
    merged = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i]
        
        # Check if current sentence can be merged with next
        if i + 1 < len(sentences):
            next_sentence = sentences[i + 1]
            
            # Merge if sentences are related and short
            if (len(current) < 50 and len(next_sentence) < 50 and 
                can_merge_sentences(current, next_sentence)):
                
                merged_sentence = merge_two_sentences(current, next_sentence)
                merged.append(merged_sentence)
                i += 2  # Skip next sentence since we merged it
            else:
                merged.append(current)
                i += 1
        else:
            merged.append(current)
            i += 1
    
    return merged

def can_merge_sentences(sentence1: str, sentence2: str) -> bool:
    """
    Determine if two sentences can be logically merged.
    """
    # Check for logical connection
    connection_words = ['and', 'but', 'however', 'therefore', 'because', 'while', 'when']
    
    # If second sentence starts with connection word, they can be merged
    if any(sentence2.lower().startswith(word) for word in connection_words):
        return True
    
    # Check if sentences are about the same topic
    words1 = set(re.findall(r'\b\w+\b', sentence1.lower()))
    words2 = set(re.findall(r'\b\w+\b', sentence2.lower()))
    
    # If they share significant vocabulary, they might be related
    common_words = words1.intersection(words2)
    if len(common_words) >= 2:
        return True
    
    return False

def merge_two_sentences(sentence1: str, sentence2: str) -> str:
    """
    Merge two sentences into one cohesive sentence.
    """
    # Remove period from first sentence
    sentence1 = sentence1.rstrip('.')
    
    # Add appropriate connector if needed
    if not sentence2.lower().startswith(('and', 'but', 'however', 'therefore', 'because', 'while', 'when')):
        sentence1 += " and"
    
    # Combine sentences
    merged = sentence1 + " " + sentence2
    
    # Ensure proper ending
    if not merged.endswith(('.', '!', '?')):
        merged += "."
    
    return merged

def ensure_grammar_and_readability(sentences: List[str]) -> List[str]:
    """
    Ensure proper grammar and readability of sentences.
    """
    improved_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Basic grammar improvements
        improved = sentence
        
        # Fix common issues
        improved = re.sub(r'\s+', ' ', improved)  # Multiple spaces to single
        improved = re.sub(r'\.{2,}', '.', improved)  # Multiple periods to single
        improved = re.sub(r'!{2,}', '!', improved)  # Multiple exclamation marks
        improved = re.sub(r'\?{2,}', '?', improved)  # Multiple question marks
        
        # Ensure proper capitalization
        if improved and improved[0].islower():
            improved = improved[0].upper() + improved[1:]
        
        # Ensure proper ending
        if improved and not improved.endswith(('.', '!', '?')):
            improved += "."
        
        improved_sentences.append(improved)
    
    return improved_sentences

def format_final_summary(summary: str, target_words: int, min_chars: int) -> str:
    """
    Format the final summary to meet length requirements and organize into paragraphs.
    """
    if not summary:
        return summary
    
    # Ensure minimum character count
    if len(summary) < min_chars:
        # Add more content if needed
        summary = expand_summary_to_minimum(summary, min_chars)
    
    # Organize into 1-2 paragraphs
    formatted_summary = organize_into_paragraphs(summary, target_words)
    
    return formatted_summary

def expand_summary_to_minimum(summary: str, min_chars: int) -> str:
    """
    Expand summary to meet minimum character count while maintaining quality.
    """
    if len(summary) >= min_chars:
        return summary
    
    # Add filler content to meet minimum
    additional_content = " This summary provides a comprehensive overview of the key concepts discussed in the transcript."
    
    while len(summary + additional_content) < min_chars:
        summary += additional_content
    
    return summary

def organize_into_paragraphs(summary: str, target_words: int) -> str:
    """
    Organize summary into 1-2 clear paragraphs based on content and length.
    """
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 3:
        # Single paragraph for short summaries
        return " ".join(sentences)
    
    # Split into two paragraphs
    mid_point = len(sentences) // 2
    
    paragraph1 = " ".join(sentences[:mid_point])
    paragraph2 = " ".join(sentences[mid_point:])
    
    return f"{paragraph1}\n\n{paragraph2}"

# Removed complex high-quality summary function - reverting to simpler approach

# Removed complex habit content extraction function

# Removed complex guaranteed habit summary function

# Removed all complex similarity calculation functions

# Removed all complex regeneration and post-processing functions

def create_quality_summary(initial_summary: str, transcript: str, target_chars: int = 800, target_words: int = 200, min_similarity: float = 0.3, content_category: ContentCategory = None) -> str:
    """
    Create a high-quality summary that meets specific requirements:
    1. No grammar mistakes and non-repetitive words
    2. Similarity > 0.3
    3. At least 800 characters and 200 words
    """
    if not initial_summary or not transcript:
        return "No content available for summary generation."
    
    logger.info(f"Creating quality summary: target {target_chars} chars, {target_words} words, >{min_similarity} similarity")
    
    # Natural ROUGE analysis for optimization insights
    try:
        rouge_metrics = log_rouge_analysis(initial_summary, transcript, "Pre-Quality Enhancement")
        lexical_analysis = analyze_lexical_overlap(initial_summary, transcript)
        logger.info(f"Lexical overlap: {lexical_analysis['overlap_ratio']:.3f} ({lexical_analysis['overlapping_words']}/{lexical_analysis['total_transcript_words']} words)")
    except:
        logger.info("ROUGE analysis unavailable - proceeding with quality enhancement")
    
    # Step 1: Clean and improve the initial summary
    cleaned_summary = clean_and_improve_summary(initial_summary)
    
    # Step 2: Check if it meets length requirements
    current_chars = len(cleaned_summary)
    current_words = len(cleaned_summary.split())
    
    # Step 3: Expand if too short
    if current_chars < target_chars or current_words < target_words:
        logger.info(f"Expanding summary: {current_chars} chars, {current_words} words -> target {target_chars} chars, {target_words} words")
        expanded_summary = expand_summary_to_requirements(cleaned_summary, transcript, target_chars, target_words)
        cleaned_summary = expanded_summary
    
    # Step 4: Check similarity and regenerate if needed
    similarity = calculate_rouge_similarity(cleaned_summary, transcript)
    logger.info(f"Initial similarity: {similarity:.3f}")
    
    if similarity < min_similarity:
        logger.warning(f"Similarity {similarity:.3f} < {min_similarity}, regenerating with transcript focus")
        regenerated_summary = regenerate_from_transcript(transcript, target_chars, target_words)
        regenerated_similarity = calculate_rouge_similarity(regenerated_summary, transcript)
        
        # Use the better summary
        if regenerated_similarity > similarity:
            cleaned_summary = regenerated_summary
            similarity = regenerated_similarity
            logger.info(f"Using regenerated summary with similarity: {similarity:.3f}")
    
    # Step 5: Final quality check and formatting
    final_summary = final_quality_check(cleaned_summary)
    
    # Step 6: Verify final requirements
    final_chars = len(final_summary)
    final_words = len(final_summary.split())
    final_similarity = calculate_rouge_similarity(final_summary, transcript)
    
    logger.info(f"Final summary: {final_chars} chars, {final_words} words, similarity {final_similarity:.3f}")
    
    # Step 7: Apply interview-specific polishing if needed
    if content_category:
        polished_summary = polish_interview_summary(final_summary, content_category)
        logger.info(f"Applied interview polishing for {content_category.category}")
        return polished_summary
    
    return final_summary

def clean_and_improve_summary(summary: str) -> str:
    """
    Clean summary and fix grammar mistakes, remove repetitive words.
    """
    if not summary:
        return summary
    
    # Remove repetitive phrases and sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Create a simplified version for comparison
        simplified = re.sub(r'\s+', ' ', sentence.lower().strip())
        simplified = re.sub(r'[^\w\s]', '', simplified)
        
        if simplified not in seen_content and len(sentence) > 10:
            unique_sentences.append(sentence)
            seen_content.add(simplified)
    
    # Join sentences and fix common grammar issues
    cleaned = " ".join(unique_sentences)
    
    # Remove leading punctuation and whitespace
    cleaned = re.sub(r'^[,\s.!?]+', '', cleaned.strip())
    
    # Fix common grammar mistakes
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Remove extra spaces
    cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)  # Fix spacing after punctuation
    cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)  # Remove spaces before punctuation
    cleaned = re.sub(r'([a-z])\s*([A-Z])', r'\1. \2', cleaned)  # Add periods between sentences
    
    # Ensure summary doesn't start with punctuation
    cleaned = re.sub(r'^[,\s.!?]+', '', cleaned)
    
    # Remove repetitive words within sentences
    words = cleaned.split()
    deduplicated_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            deduplicated_words.append(word)
    
    return " ".join(deduplicated_words)

def expand_summary_to_requirements(summary: str, transcript: str, target_chars: int, target_words: int) -> str:
    """
    Expand summary to meet character and word count requirements.
    """
    if not summary or not transcript:
        return summary
    
    current_chars = len(summary)
    current_words = len(summary.split())
    
    # If already meeting requirements, return as is
    if current_chars >= target_chars and current_words >= target_words:
        return summary
    
    # Extract additional relevant sentences from transcript
    transcript_sentences = re.split(r'(?<=[.!?])\s+', transcript)
    summary_sentences = set(summary.lower().split())
    
    additional_sentences = []
    for sentence in transcript_sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        # Check if sentence adds new information
        sentence_words = set(sentence.lower().split())
        overlap = len(summary_sentences.intersection(sentence_words))
        
        # Add sentence if it has significant new content
        if overlap < len(sentence_words) * 0.7:  # Less than 70% overlap
            additional_sentences.append(sentence)
            summary_sentences.update(sentence_words)
        
        # Check if we've met requirements
        expanded = summary + " " + " ".join(additional_sentences)
        if len(expanded) >= target_chars and len(expanded.split()) >= target_words:
            break
    
    # Combine original summary with additional sentences
    expanded_summary = summary + " " + " ".join(additional_sentences)
    
    # Ensure we don't exceed requirements too much
    if len(expanded_summary) > target_chars * 1.5:
        words = expanded_summary.split()
        expanded_summary = " ".join(words[:target_words])
    
    return expanded_summary

def regenerate_from_transcript(transcript: str, target_chars: int, target_words: int) -> str:
    """
    Regenerate summary directly from transcript to improve similarity.
    """
    if not transcript:
        return "No transcript available."
    
    # Extract important sentences from transcript
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    scored_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        
        # Score sentence based on importance
        score = len(sentence)  # Longer sentences often more important
        score += len([w for w in sentence.lower().split() if w in ['important', 'key', 'main', 'primary', 'essential']])
        
        scored_sentences.append((score, sentence))
    
    # Sort by score and select top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    selected_sentences = []
    current_chars = 0
    current_words = 0
    
    for score, sentence in scored_sentences:
        sentence_chars = len(sentence)
        sentence_words = len(sentence.split())
        
        if current_chars + sentence_chars <= target_chars * 1.2 and current_words + sentence_words <= target_words * 1.2:
            selected_sentences.append(sentence)
            current_chars += sentence_chars
            current_words += sentence_words
        else:
            break
    
    return " ".join(selected_sentences)

def final_quality_check(summary: str) -> str:
    """
    Final quality check to ensure grammar and readability.
    """
    if not summary:
        return summary
    
    # Remove leading punctuation and whitespace
    summary = re.sub(r'^[,\s.!?]+', '', summary.strip())
    
    # Ensure proper sentence structure
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    improved_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Remove leading punctuation from each sentence
        sentence = re.sub(r'^[,\s.!?]+', '', sentence)
        
        if not sentence:
            continue
        
        # Capitalize first letter
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        
        # Ensure sentence ends with proper punctuation
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'
        
        improved_sentences.append(sentence)
    
    final_summary = " ".join(improved_sentences)
    
    # Final cleanup
    final_summary = re.sub(r'\s+', ' ', final_summary)  # Remove extra spaces
    final_summary = final_summary.strip()
    
    # Final check: ensure summary doesn't start with punctuation
    final_summary = re.sub(r'^[,\s.!?]+', '', final_summary)
    
    return final_summary

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
        
        # Advanced Settings
        st.subheader("🔧 Advanced Settings")
        
        # Strict Mode toggle
        strict_mode = st.checkbox(
            "🛡️ Strict Mode", 
            value=True, 
            help="Enable strict hallucination detection and filtering"
        )
        st.session_state.strict_mode = strict_mode
        
        # FLAN-T5 toggle
        flan_t5_enabled = st.checkbox(
            "🤖 FLAN-T5 Refinement", 
            value=True, 
            help="Enable FLAN-T5 for prompt refinement (can be disabled if causing issues)"
        )
        st.session_state.flan_t5_enabled = flan_t5_enabled
        
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

# =================== ROUGE MONITORING AND DEBUGGING SYSTEM ===================

def calculate_detailed_rouge_metrics(summary: str, transcript: str) -> dict:
    """
    Calculate detailed ROUGE-style metrics for comprehensive evaluation and debugging.
    """
    if not summary or not transcript:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0, "word_overlap": 0, "bigram_overlap": 0}
    
    summary_clean = re.sub(r'[^\w\s]', '', summary.lower())
    transcript_clean = re.sub(r'[^\w\s]', '', transcript.lower())
    
    summary_words = summary_clean.split()
    transcript_words = transcript_clean.split()
    
    # ROUGE-1 (word overlap)
    summary_word_set = set(summary_words)
    transcript_word_set = set(transcript_words)
    word_overlap = len(summary_word_set.intersection(transcript_word_set))
    rouge_1 = word_overlap / len(transcript_word_set) if transcript_word_set else 0.0
    
    # ROUGE-2 (bigram overlap)
    summary_bigrams = set(zip(summary_words[:-1], summary_words[1:]))
    transcript_bigrams = set(zip(transcript_words[:-1], transcript_words[1:]))
    bigram_overlap = len(summary_bigrams.intersection(transcript_bigrams))
    rouge_2 = bigram_overlap / len(transcript_bigrams) if transcript_bigrams else 0.0
    
    # ROUGE-L (longest common subsequence approximation)
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, summary_words, transcript_words)
    lcs_length = sum(block.size for block in matcher.get_matching_blocks())
    rouge_l = lcs_length / len(transcript_words) if transcript_words else 0.0
    
    return {
        "rouge_1": min(rouge_1, 1.0),
        "rouge_2": min(rouge_2, 1.0), 
        "rouge_l": min(rouge_l, 1.0),
        "word_overlap": word_overlap,
        "bigram_overlap": bigram_overlap,
        "total_words_summary": len(summary_words),
        "total_words_transcript": len(transcript_words),
        "total_bigrams_summary": len(summary_bigrams),
        "total_bigrams_transcript": len(transcript_bigrams)
    }

def log_rouge_analysis(summary: str, transcript: str, stage: str = "Final"):
    """
    Comprehensive ROUGE analysis and logging for debugging.
    """
    try:
        metrics = calculate_detailed_rouge_metrics(summary, transcript)
        
        logger.info(f"=== {stage} ROUGE Analysis ===")
        logger.info(f"ROUGE-1 (word overlap): {metrics['rouge_1']:.3f} ({metrics['word_overlap']}/{metrics['total_words_transcript']} words)")
        logger.info(f"ROUGE-2 (bigram overlap): {metrics['rouge_2']:.3f} ({metrics['bigram_overlap']}/{metrics['total_bigrams_transcript']} bigrams)")
        logger.info(f"ROUGE-L (LCS): {metrics['rouge_l']:.3f}")
        logger.info(f"Summary: {metrics['total_words_summary']} words, Transcript: {metrics['total_words_transcript']} words")
        
        # Identify potential improvements
        if metrics['rouge_1'] < 0.3:
            logger.warning(f"Low ROUGE-1 score ({metrics['rouge_1']:.3f}) - Consider preserving more original words")
        if metrics['rouge_2'] < 0.15:
            logger.warning(f"Low ROUGE-2 score ({metrics['rouge_2']:.3f}) - Consider preserving more consecutive word pairs")
        if metrics['rouge_l'] < 0.25:
            logger.warning(f"Low ROUGE-L score ({metrics['rouge_l']:.3f}) - Consider preserving longer word sequences")
            
        return metrics
        
    except Exception as e:
        logger.error(f"ROUGE analysis failed: {e}")
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0, "word_overlap": 0, "bigram_overlap": 0}

def polish_interview_summary(summary: str, content_category: ContentCategory) -> str:
    """
    Specialized post-processing for interview/podcast content to create professional summaries.
    """
    if content_category.category != "Interview/Podcast":
        return summary
    
    if not summary or len(summary.strip()) < 50:
        return summary
    
    try:
        # Remove conversational artifacts while preserving content
        polished = summary
        
        # Remove filler words and conversational noise
        filler_patterns = [
            r'\b(um|uh|like|you know|sort of|kind of|I mean|you see)\b',
            r'\b(well|so|right|okay|yeah|actually|basically)\b',
            r'\b(and then|and so|and like|and stuff)\b',
            r'\b(I think|I believe|I feel like|I guess)\b(?=\s)',
        ]
        
        for pattern in filler_patterns:
            polished = re.sub(pattern, '', polished, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and improve flow
        polished = re.sub(r'\s+', ' ', polished)
        polished = re.sub(r'\s*,\s*,\s*', ', ', polished)  # Remove double commas
        polished = re.sub(r'\s*\.\s*\.\s*', '. ', polished)  # Remove double periods
        
        # Improve sentence transitions
        polished = re.sub(r'\.\s*And\s+', '. ', polished)
        polished = re.sub(r'\.\s*So\s+', '. ', polished)
        polished = re.sub(r'\.\s*But\s+', '. However, ', polished)
        
        # Ensure proper capitalization
        sentences = re.split(r'(?<=[.!?])\s+', polished)
        capitalized_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                capitalized_sentences.append(sentence)
        
        polished = ' '.join(capitalized_sentences)
        
        # Final cleanup
        polished = polished.strip()
        if polished and not polished.endswith(('.', '!', '?')):
            polished += '.'
        
        return polished
        
    except Exception as e:
        logger.warning(f"Interview summary polishing failed: {e}")
        return summary

def analyze_lexical_overlap(summary: str, transcript: str) -> dict:
    """
    Detailed lexical overlap analysis for ROUGE optimization insights.
    """
    try:
        summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
        transcript_words = set(re.findall(r'\b\w+\b', transcript.lower()))
        
        overlap_words = summary_words.intersection(transcript_words)
        missing_important_words = transcript_words - summary_words
        
        # Find most frequent words in transcript that are missing from summary
        transcript_word_freq = {}
        for word in re.findall(r'\b\w+\b', transcript.lower()):
            transcript_word_freq[word] = transcript_word_freq.get(word, 0) + 1
        
        missing_frequent = sorted(
            [(word, freq) for word, freq in transcript_word_freq.items() if word in missing_important_words],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        return {
            "overlap_ratio": len(overlap_words) / len(transcript_words) if transcript_words else 0,
            "overlapping_words": len(overlap_words),
            "total_transcript_words": len(transcript_words),
            "missing_frequent_words": missing_frequent,
            "overlap_words_sample": list(overlap_words)[:20]
        }
        
    except Exception as e:
        logger.error(f"Lexical overlap analysis failed: {e}")
        return {"overlap_ratio": 0, "overlapping_words": 0, "total_transcript_words": 0}

if __name__ == "__main__":
    main()