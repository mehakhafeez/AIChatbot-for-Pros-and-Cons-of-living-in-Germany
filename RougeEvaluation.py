"""
This script fetches YouTube video transcripts, generates summaries using OpenAI GPT,
and evaluates them using ROUGE scores against reference summaries.
"""

import os
from getpass import getpass
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm
import openai
import evaluate
from nltk.tokenize import sent_tokenize

# Load environment variables
_ = load_dotenv(find_dotenv())

# Set API key for OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or getpass("Enter your OpenAI API key: ")
openai.api_key = OPENAI_API_KEY

# Fetch YouTube video transcripts
def get_video_transcript(vid_id):
    """
    Fetch transcript for a given YouTube video ID.
    """
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(vid_id)
        transcript_text = " ".join([item['text'] for item in transcript_data if 'text' in item])
        return transcript_text
    except Exception as e:
        print(f"Error retrieving transcript for video {vid_id}: {e}")
        return None

# List of YouTube video IDs
video_ids = ['2u4ItZerRac', 'I2zF1I60hPg', '8xqSF-uHCUs', 'LtmS-c1pChY', 'sJNxT-I7L6s']
transcripts = {}

# Fetch transcripts for each video
for vid_id in video_ids:
    trans = get_video_transcript(vid_id)
    if trans:
        transcripts[vid_id] = trans
    else:
        print(f"No transcript available for video {vid_id}")

# Reference summaries for evaluation
reference_summaries = {
    '2u4ItZerRac': "Living in Germany offers a strong economy and high quality of life. However, there are challenges like language barriers and integration issues.",
    'I2zF1I60hPg': "Germany has a robust healthcare system and rich cultural experiences, but the cost of living is high.",
    '8xqSF-uHCUs': "Germany provides efficient public services and strong social security, but finding housing can be difficult.",
    'LtmS-c1pChY': "Germany's education system and work-life balance are excellent, but the tax system is complex.",
    'sJNxT-I7L6s': "Germany's vibrant cities and safety are attractive, but cultural differences can be a challenge."
}

# Generate summaries using GPT-3.5-turbo
def generate_summaries_gpt(transcripts):
    """
    Generate summaries for a list of transcripts using GPT-3.5-turbo.
    """
    summaries = []
    for transcript in tqdm(transcripts, desc="Generating summaries"):
        prompt = f"Summarize the following text concisely:\n\n{transcript}\n\nSummary:"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.5,
            )
            summary = response.choices[0].message['content'].strip()
            summaries.append(summary)
        except Exception as e:
            print(f"Error generating summary for transcript: {e}")
            summaries.append(None)
    return summaries

# Generate summaries for the transcripts
generated_summaries = generate_summaries_gpt(list(transcripts.values()))

# Initialize ROUGE evaluator
rouge_score = evaluate.load("rouge")

# Define the function to compute ROUGE scores
def compute_rouge_score(generated, reference):
    """
    Compute ROUGE scores for generated and reference summaries.
    """
    # Tokenize sentences for better scoring
    generated_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in generated if s]
    reference_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in reference]
    return rouge_score.compute(predictions=generated_with_newlines, references=reference_with_newlines, use_stemmer=True)

# Evaluate the generated summaries against reference summaries
generated_summaries_dict = {video_id: summary for video_id, summary in zip(transcripts.keys(), generated_summaries)}
new_rouge_scores = compute_rouge_score(list(generated_summaries_dict.values()), list(reference_summaries.values()))

# Print results
print("\nGenerated Summaries:")
for video_id, summary in generated_summaries_dict.items():
    print(f"Video ID: {video_id}\nSummary: {summary}\n")

print("\nROUGE Scores:")
print(new_rouge_scores)