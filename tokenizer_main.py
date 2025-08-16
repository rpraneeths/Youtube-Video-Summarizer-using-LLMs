from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

def summarize_text(text):
    # Load pre-trained Pegasus model and tokenizer
    model_name = "google/pegasus-xsum"  # You can change this model name to any other pre-trained Pegasus model
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary using the model
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=5, length_penalty=2.0, early_stopping=True)

    # Decode the generated summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage:
if __name__ == "__main__":
    text = """
    The Pegasus model, developed by Google Research, is designed for abstractive text summarization.
    Unlike extractive methods that directly select sentences from the original document, abstractive summarization aims to generate a condensed version of the input text by rephrasing and paraphrasing.
    This makes the summaries more coherent and fluent. Pegasus has achieved state-of-the-art results on several summarization tasks and datasets.
    """

    summary = summarize_text(text)
    print("Original Text:\n", text)
    print("\nSummarized Text:\n", summary)