
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url):
    parsed = urlparse(url)

    if parsed.hostname in ["youtu.be"]:
        return parsed.path[1:]

    if parsed.hostname and "youtube.com" in parsed.hostname:
        return parse_qs(parsed.query).get("v", [None])[0]

    return None

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def simple_summarize(text, num_sentences=5):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) <= num_sentences:
        return text

    words = clean_text(text).split()
    freq = Counter(words)

    scored = []
    for sentence in sentences:
        score = sum(freq.get(w, 0) for w in clean_text(sentence).split())
        scored.append((score, sentence))

    scored.sort(reverse=True)

    return " ".join([s for _, s in scored[:num_sentences]])

def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([x["text"] for x in transcript])

def process_youtube_url(url):
    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript = fetch_transcript(video_id)

    return {
        "summary": simple_summarize(transcript),
        "word_count": len(transcript.split()),
        "raw_transcript": transcript
    }
