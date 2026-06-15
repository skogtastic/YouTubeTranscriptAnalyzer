
import re
from collections import Counter

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def simple_summarize(text: str, num_sentences: int = 5) -> str:
    if not text:
        return "No transcript provided."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= num_sentences:
        return text
    cleaned = clean_text(text)
    words = cleaned.split()
    word_freq = Counter(words)
    ranked = []
    for sentence in sentences:
        score = sum(word_freq.get(w, 0) for w in clean_text(sentence).split())
        ranked.append((score, sentence))
    ranked.sort(reverse=True)
    return " ".join([s for _, s in ranked[:num_sentences]])

def process_video(transcript_text: str) -> dict:
    return {
        "summary": simple_summarize(transcript_text),
        "word_count": len(transcript_text.split()) if transcript_text else 0,
        "raw_transcript": transcript_text
    }
