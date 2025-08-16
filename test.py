from youtube_transcript_api import YouTubeTranscriptApi
# video_id = 'lZTBIO3OtYc'
# video_id = 'A7vqRwVKwD8'
video_id = 'VQkiGeW3T5Q'  # Replace with the actual YouTube video ID
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id)
    # print(transcript_list)
    # The transcript_list will be a list of dictionaries,
    # where each dictionary contains 'text', 'start', and 'duration'
    
    # To get the plain text transcript:
    plain_text_transcript = " ".join([item.text for item in transcript_list])
    print(plain_text_transcript)
        
except Exception as e:
    print(f"Error retrieving transcript: {e}")