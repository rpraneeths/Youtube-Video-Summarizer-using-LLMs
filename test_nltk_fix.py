#!/usr/bin/env python3
"""
Test script to verify NLTK fallback functionality
"""

def test_nltk_fallback():
    """Test the NLTK fallback functionality"""
    print("Testing NLTK fallback functionality...")
    
    # Test 1: Try to import NLTK components
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        print("✅ NLTK imported successfully")
        nltk_available = True
    except ImportError:
        print("⚠️ NLTK not available, using fallback")
        nltk_available = False
        
        def sent_tokenize(text):
            """Fallback sentence tokenization using simple period splitting"""
            return [s.strip() for s in text.split('.') if s.strip()]
    
    # Test 2: Test sentence tokenization
    test_text = "This is a test sentence. This is another sentence. And a third one."
    
    try:
        sentences = sent_tokenize(test_text)
        print(f"✅ Sentence tokenization successful: {len(sentences)} sentences found")
        for i, sentence in enumerate(sentences):
            print(f"   {i+1}. {sentence}")
    except Exception as e:
        print(f"❌ Sentence tokenization failed: {e}")
    
    # Test 3: Test with empty or invalid text
    try:
        empty_sentences = sent_tokenize("")
        print(f"✅ Empty text handling: {len(empty_sentences)} sentences")
        
        invalid_sentences = sent_tokenize("No periods here")
        print(f"✅ No periods handling: {len(invalid_sentences)} sentences")
        
    except Exception as e:
        print(f"❌ Edge case handling failed: {e}")
    
    print("\n🎯 NLTK fallback test completed!")
    return nltk_available

if __name__ == "__main__":
    test_nltk_fallback()
