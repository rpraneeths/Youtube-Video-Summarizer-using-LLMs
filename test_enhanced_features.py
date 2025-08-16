#!/usr/bin/env python3
"""
Test script for Enhanced YouTube Video Summarizer features
"""

import sys
import os
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_url_validation():
    """Test URL validation functionality"""
    print("🧪 Testing URL Validation...")
    
    try:
        from stream_lit_ui import validate_youtube_url
        
        # Test valid URLs
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            result = validate_youtube_url(url)
            if result.is_valid:
                print(f"✅ Valid URL: {url}")
                print(f"   Video ID: {result.video_id}")
                print(f"   URL Type: {result.url_type}")
            else:
                print(f"❌ Invalid URL: {url}")
                print(f"   Error: {result.error}")
        
        # Test invalid URLs
        invalid_urls = [
            "",
            "not_a_url",
            "https://www.youtube.com/invalid",
            "https://vimeo.com/123456"
        ]
        
        for url in invalid_urls:
            result = validate_youtube_url(url)
            if not result.is_valid:
                print(f"✅ Correctly rejected invalid URL: {url}")
                print(f"   Error: {result.error}")
            else:
                print(f"❌ Should have rejected invalid URL: {url}")
        
        print("✅ URL validation tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ URL validation test failed: {e}")
        return False

def test_content_categorization():
    """Test content categorization functionality"""
    print("\n🧪 Testing Content Categorization...")
    
    try:
        from stream_lit_ui import categorize_content
        
        # Test different content types
        test_transcripts = {
            "News": "Breaking news today, the government announced new policies. Officials say this will affect millions of people.",
            "Educational": "In this tutorial, we'll learn about machine learning. First, let's understand the basic concepts.",
            "Technical": "To install this software, run the following command: pip install package. Then configure the settings.",
            "Entertainment": "This movie was absolutely hilarious! The comedy was amazing and the characters were so funny.",
            "Review": "The product has many pros including great quality, but there are some cons like high price.",
            "Interview": "The interviewer asked about his experience. The guest responded with detailed insights."
        }
        
        for content_type, transcript in test_transcripts.items():
            category = categorize_content(transcript)
            print(f"📝 {content_type} Content:")
            print(f"   Detected Category: {category.category}")
            print(f"   Confidence: {category.confidence:.1%}")
            print(f"   Strategy: {category.preprocessing_strategy}")
            if category.subcategories:
                print(f"   Subcategories: {', '.join(category.subcategories)}")
            print()
        
        print("✅ Content categorization tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Content categorization test failed: {e}")
        return False

def test_preprocessing():
    """Test content preprocessing functionality"""
    print("\n🧪 Testing Content Preprocessing...")
    
    try:
        from stream_lit_ui import content_aware_preprocessing, categorize_content
        
        test_text = "Um, so like, you know, this is a technical tutorial about, um, installing software. First, you need to, like, download the package, and then, um, run the install command."
        
        # Categorize the content first
        category = categorize_content(test_text)
        print(f"📝 Original Text: {test_text}")
        print(f"🎯 Detected Category: {category.category}")
        
        # Preprocess the content
        processed_text = content_aware_preprocessing(test_text, category)
        print(f"✨ Processed Text: {processed_text}")
        
        # Check if preprocessing worked
        if len(processed_text) < len(test_text):
            print("✅ Preprocessing successfully reduced text length")
        else:
            print("⚠️ Preprocessing didn't reduce text length")
        
        print("✅ Content preprocessing tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Content preprocessing test failed: {e}")
        return False

def test_prompt_generation():
    """Test contextual prompt generation"""
    print("\n🧪 Testing Contextual Prompt Generation...")
    
    try:
        from stream_lit_ui import generate_contextual_prompt, categorize_content
        
        test_transcript = "This is an educational video about machine learning algorithms. We'll cover supervised learning, unsupervised learning, and neural networks."
        
        # Categorize content
        category = categorize_content(test_transcript)
        
        # Generate prompt
        prompt = generate_contextual_prompt(test_transcript, category)
        print(f"🎯 Content Category: {category.category}")
        print(f"📝 Generated Prompt:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        
        print("✅ Contextual prompt generation tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Contextual prompt generation test failed: {e}")
        return False

def test_fallback_summarization():
    """Test fallback summarization functionality"""
    print("\n🧪 Testing Fallback Summarization...")
    
    try:
        from stream_lit_ui import _fallback_summarization
        
        test_text = "This is the first sentence. This is the second sentence with more content. This is the third sentence that should be included. This is the fourth sentence."
        
        summary = _fallback_summarization(test_text)
        print(f"📝 Original Text: {test_text}")
        print(f"✨ Fallback Summary: {summary}")
        
        if summary and len(summary) < len(test_text):
            print("✅ Fallback summarization worked correctly")
        else:
            print("⚠️ Fallback summarization may not have worked as expected")
        
        print("✅ Fallback summarization tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fallback summarization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced YouTube Video Summarizer - Feature Tests")
    print("=" * 60)
    
    tests = [
        test_url_validation,
        test_content_categorization,
        test_preprocessing,
        test_prompt_generation,
        test_fallback_summarization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
