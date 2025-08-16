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
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")

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
    sentences = sent_tokenize(text)
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
        if not st.session_state.history:
            return None
            
        # Prepare data for the chart
        df = pd.DataFrame(st.session_state.history)
        
        # Convert timestamp to datetime and extract hour
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Count analyses by hour
        hour_counts = df['hour'].value_counts().sort_index()
        
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
                title="Hour of Day",
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[-0.5, 23.5],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff'),
                titlefont=dict(color='#ffffff')
            ),
            yaxis=dict(
                title="Number of Analyses",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#ffffff'),
                titlefont=dict(color='#ffffff')
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
        st.error(f"Error creating analysis distribution chart: {str(e)}")
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
        duration_sec = transcript[-1]['start'] + transcript[-1]['duration']
        duration_min = duration_sec / 60
        
        # Determine appropriate interval
        interval = get_sentiment_interval(duration_min)
        
        # Group transcript by time intervals
        sentiments = []
        time_labels = []
        current_text = ""
        current_start = 0
        
        for segment in transcript:
            segment_min = segment['start'] / 60
            
            # If we've reached a new interval, analyze the accumulated text
            if segment_min - current_start >= interval:
                if current_text:
                    blob = TextBlob(current_text)
                    sentiments.append(blob.sentiment.polarity)
                    time_labels.append(f"{int(current_start)}-{int(current_start)+interval}min")
                
                # Reset for next interval
                current_text = segment['text']
                current_start = interval * (segment_min // interval)
            else:
                current_text += " " + segment['text']
        
        # Analyze the last segment
        if current_text:
            blob = TextBlob(current_text)
            sentiments.append(blob.sentiment.polarity)
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
                    title="Sentiment",
                    titleside="right",
                    titlefont=dict(color='#ffffff'),
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
            title="Video Timeline (minutes)",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff'),
            titlefont=dict(color='#ffffff')
        ),
        yaxis=dict(
            title="Sentiment Score",
            range=[-1, 1],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            tickfont=dict(color='#ffffff'),
            titlefont=dict(color='#ffffff')
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

def process_video(url):
    """Process YouTube video and generate summary"""
    try:
        cleanup_audio()
        process_start_time = time.time()
        
        # Get video ID
        video_id = url.split("watch?v=")[-1].split('&')[0]
        
        # Get transcript
        with st.status("Fetching transcript...", expanded=True) as status:
            try:
                try:
                    ytt_api = YouTubeTranscriptApi()
                    transcript = ytt_api.fetch(video_id)
                    full_text = " ".join([line['text'] for line in transcript])
                    if not full_text.strip():
                        st.error("No transcript available for this video")
                        return False
                    st.session_state.transcript = full_text
                    status.update(label="Transcript fetched successfully!", state="complete")
                except Exception as e:
                    from youtube_transcript_api._errors import NoTranscriptFound
                    if isinstance(e, NoTranscriptFound):
                        st.error("No transcript available for this video.")
                    else:
                        st.error(f"Error fetching transcript: {str(e)}")
                    return False
            except Exception as e:
                st.error(f"Error fetching transcript: {str(e)}")
                return False
        
        # Initialize model
        with st.status("Initializing summarization model...", expanded=True) as status:
            try:
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                status.update(label="Model initialized successfully!", state="complete")
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")
                return False
        
        # Generate summary
        with st.status("Generating summary...", expanded=True) as status:
            try:
                # Split into chunks that fit within the model's context window
                max_chunk_size = 1024
                chunks = [full_text[i:i+max_chunk_size] for i in range(0, len(full_text), max_chunk_size)]
                
                summarized_text = []
                for chunk in chunks:
                    summary = summarizer(chunk)
                    summarized_text.append(summary[0]['summary_text'])
                
                st.session_state.summary = " ".join(summarized_text)
                status.update(label="Summary generated successfully!", state="complete")
            except Exception as e:
                st.error(f"Error during summarization: {str(e)}")
                return False
        
        # Generate audio
        with st.status("Generating audio summary...", expanded=True) as status:
            audio_file = text_to_speech(st.session_state.summary)
            if audio_file:
                st.session_state.audio_file = audio_file
                status.update(label="Audio summary generated!", state="complete")
        
        # Sentiment analysis
        with st.status("Analyzing sentiment...", expanded=True) as status:
            analyze_sentiment(url)
            status.update(label="Sentiment analysis complete!", state="complete")
            
        # Calculate and store execution time
        st.session_state.execution_time = time.time() - process_start_time
        st.session_state.analysis_complete = True
        
        # Add to history
        add_to_history(
            video_id,
            f"Video {video_id[:8]}...",
            st.session_state.summary,
            st.session_state.avg_polarity
        )
        
        return True
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return False

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
        
        highlighted_summary = highlight_key_sentences(st.session_state.summary)
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
            <h3 style='margin-bottom: 1.5rem;'>Sentiment Analysis</h3>
            """, unsafe_allow_html=True)
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
    if st.session_state.history:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Distribution")
        
        # Create visualization of analysis types
        fig = create_analysis_distribution_chart()
        st.plotly_chart(fig, use_container_width=True)
        
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
        st.markdown("""
            <h3 style='margin-bottom: 1.5rem;'>Sentiment Analysis</h3>""", 
		unsafe_allow_html=True)
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
            'voice': voice,
            'audio_quality': quality,
            'playback_speed': playback_speed,
            'volume': volume,
            'auto_play': auto_play,
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
    """Create the main analyzer view"""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='margin-bottom: 1rem;'>🎥 YouTube Video Summarizer</h1>
            <p style='opacity: 0.8;'>Transform your video content into concise, actionable insights</p>
        </div>
    """, unsafe_allow_html=True)
    
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
            if process_video(url):
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
        <p style="margin: 0;">Made with ❤️ | Powered by BART and TextBlob</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function"""
    # Install required packages
    # install_packages()
    
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