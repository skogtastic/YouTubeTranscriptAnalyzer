import re
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from openai import OpenAI

client = OpenAI()

def get_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(item["text"] for item in transcript)

def summarize_and_structure(transcript):
    prompt = f"""
    Analyze this YouTube transcript.

    Return:
    1. A concise summary (max 200 words)
    2. A structured outline of the video sections

    Transcript:
    {transcript[:15000]}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content

    return content

def process_video(url):
    video_id = get_video_id(url)
    yt = YouTube(url)

    title = yt.title
    transcript = get_transcript(video_id)

    analysis = summarize_and_structure(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": analysis,
        "structure": analysis
    }
