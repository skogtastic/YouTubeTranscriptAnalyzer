"""
YouTube Transcript Analyzer — Streamlit App (Free / Gemini version)
=====================================================================
Fetches transcripts for a batch of YouTube URLs and displays the raw
transcript text. Uses Google Gemini (free tier) for summarization.

Setup:
    pip install streamlit youtube-transcript-api google-generativeai python-dotenv

Get a free Gemini API key (no credit card) at: https://aistudio.google.com

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push to GitHub, connect at share.streamlit.io, add GEMINI_API_KEY
    under Settings → Secrets.
"""

import re
import os
import json
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()

MAX_TRANSCRIPT_CHARS = 8_000
MAX_CROSS_CHARS      = 2_000

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VideoResult:
    url: str
    video_id: Optional[str]   = None
    transcript: Optional[str] = None
    analysis: Optional[dict]  = None
    error: Optional[str]      = None

    @property
    def success(self) -> bool:
        return self.transcript is not None and self.error is None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        m = re.search(pattern, url.strip())
        if m:
            return m.group(1)
    return None


def parse_urls(raw: str) -> list:
    return [u.strip() for u in re.split(r"[\n,]+", raw) if u.strip()]


def fetch_transcript(video_id: str) -> str:
    ytt = YouTubeTranscriptApi()

    # Try English first
    try:
        fetched = ytt.fetch(video_id, languages=["en", "en-US", "en-GB", "en-CA", "en-AU"])
        return " ".join(s.text for s in fetched)
    except Exception:
        pass

    # Fall back to any available language, translate to English
    try:
        for transcript in ytt.list(video_id):
            try:
                if transcript.is_translatable:
                    fetched = transcript.translate("en").fetch()
                else:
                    fetched = transcript.fetch()
                return " ".join(s.text for s in fetched)
            except Exception:
                continue
    except Exception:
        pass

    raise RuntimeError(
        "Could not retrieve a transcript. The video may have no captions, "
        "be private, age-restricted, or blocked in this region."
    )


def call_gemini(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text


def analyze_transcript(model, transcript: str) -> dict:
    prompt = f"""Analyze the YouTube video transcript below and respond with ONLY a valid JSON
object — no markdown fences, no explanation.

Schema:
{{
  "summary": "2-3 sentence overview of the video",
  "key_points": ["point 1", "point 2", "point 3", "point 4"],
  "themes": ["theme 1", "theme 2", "theme 3"],
  "sentiment": "positive | neutral | negative | mixed",
  "speaker_tone": "brief description of tone and delivery style"
}}

Transcript:
{transcript[:MAX_TRANSCRIPT_CHARS]}
"""
    raw = call_gemini(model, prompt)
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"summary": raw, "key_points": [], "themes": [], "sentiment": "unknown", "speaker_tone": ""}


def cross_video_analysis(model, results: list) -> dict:
    snippets = "\n\n---\n\n".join(
        f"VIDEO {i+1} ({r.url}):\n{r.transcript[:MAX_CROSS_CHARS]}"
        for i, r in enumerate(results)
        if r.success
    )
    prompt = f"""You are analyzing transcripts from {len(results)} YouTube videos.
Identify cross-video patterns and respond with ONLY valid JSON — no markdown fences.

Schema:
{{
  "overarching_themes": ["theme shared across videos"],
  "common_narrative": "the story or message that keeps appearing",
  "notable_patterns": "unusual patterns, repeated claims, or coordinated messaging",
  "content_diversity": "are these videos saying similar or different things?",
  "recommendations": "what should someone know after reviewing all of these?"
}}

Transcripts:
{snippets}
"""
    raw = call_gemini(model, prompt)
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"common_narrative": raw, "overarching_themes": [], "notable_patterns": "",
                "content_diversity": "", "recommendations": ""}


def export_report(results: list, cross: Optional[dict]) -> str:
    lines = ["YouTube Transcript Report", "=" * 50, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"VIDEO {i}: {r.url}")
        lines.append("-" * 40)
        if r.error:
            lines.append(f"ERROR: {r.error}")
        elif r.transcript:
            if r.analysis:
                lines.append(f"Summary: {r.analysis.get('summary', '')}")
                lines.append(f"Sentiment: {r.analysis.get('sentiment', '')}")
                lines.append(f"Speaker Tone: {r.analysis.get('speaker_tone', '')}")
                lines.append("\nKey Points:")
                for kp in r.analysis.get("key_points", []):
                    lines.append(f"  • {kp}")
                lines.append("Themes: " + ", ".join(r.analysis.get("themes", [])))
                lines.append("")
            lines.append("FULL TRANSCRIPT:")
            lines.append(r.transcript)
        lines.append("")

    if cross:
        lines += [
            "CROSS-VIDEO ANALYSIS", "=" * 50,
            f"Common Narrative: {cross.get('common_narrative', '')}",
            "",
            "Overarching Themes: " + ", ".join(cross.get("overarching_themes", [])),
            "",
            f"Notable Patterns: {cross.get('notable_patterns', '')}",
            "",
            f"Content Diversity: {cross.get('content_diversity', '')}",
            "",
            f"Recommendations: {cross.get('recommendations', '')}",
        ]
    return "\n".join(lines)


# ─── UI helpers ───────────────────────────────────────────────────────────────

def render_card(result: VideoResult, index: int):
    label = f"Video {index}: `{result.url}`"

    if result.error:
        with st.expander(f"❌ {label}", expanded=False):
            st.error(result.error)
        return

    with st.expander(f"✅ {label}", expanded=True):

        # ── Analysis (if available) ───────────────────────────────────────────
        if result.analysis:
            a = result.analysis
            st.markdown(f"**Summary**\n\n{a.get('summary', '')}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Sentiment:** {a.get('sentiment', '—')}")
            with col2:
                st.markdown(f"**Tone:** {a.get('speaker_tone', '—')}")
            if a.get("key_points"):
                st.markdown("**Key Points**")
                for kp in a["key_points"]:
                    st.markdown(f"- {kp}")
            if a.get("themes"):
                st.markdown("**Themes:** " + " · ".join(f"`{t}`" for t in a["themes"]))
            st.divider()

        # ── Raw transcript ────────────────────────────────────────────────────
        st.markdown("**📄 Full Raw Transcript**")
        st.text_area(
            label="raw",
            value=result.transcript or "",
            height=300,
            label_visibility="collapsed",
        )
        st.caption(f"{len(result.transcript or ''):,} characters · "
                   f"~{len((result.transcript or '').split()):,} words")


def render_cross(cross: dict, n: int):
    st.divider()
    st.subheader(f"🔍 Cross-Video Analysis ({n} videos)")
    if cross.get("overarching_themes"):
        st.markdown("**Overarching Themes:** " +
                    "  ".join(f"`{t}`" for t in cross["overarching_themes"]))
    if cross.get("common_narrative"):
        st.info(cross["common_narrative"])
    col1, col2 = st.columns(2)
    with col1:
        if cross.get("notable_patterns"):
            st.markdown("**Notable Patterns**")
            st.markdown(cross["notable_patterns"])
    with col2:
        if cross.get("content_diversity"):
            st.markdown("**Content Diversity**")
            st.markdown(cross["content_diversity"])
    if cross.get("recommendations"):
        st.success(cross["recommendations"])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="YouTube Transcript Analyzer",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 YouTube Transcript Analyzer")
    st.caption("Free — uses YouTube captions + Google Gemini (free tier) for analysis.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        api_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Free key from https://aistudio.google.com — no credit card needed",
        )

        analyze = st.toggle(
            "Summarize with Gemini",
            value=True,
            help="Turn off to extract raw transcripts only — no API key needed at all",
        )

        st.markdown("---")
        st.markdown(
            "**Free tier limits (Gemini Flash)**\n"
            "- 15 requests / minute\n"
            "- 1,500 requests / day\n\n"
            "Get your key → [aistudio.google.com](https://aistudio.google.com)"
        )
        st.markdown("---")
        st.markdown(
            "**Tips**\n"
            "- Videos need captions (auto-generated counts)\n"
            "- Private / age-restricted videos will fail\n"
            "- Transcripts only mode needs no API key at all"
        )

    # ── URL input ─────────────────────────────────────────────────────────────
    raw_input = st.text_area(
        "YouTube URLs — one per line (or comma-separated)",
        height=160,
        placeholder=(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
            "https://youtu.be/anotherVideoId\n"
            "..."
        ),
    )

    urls         = parse_urls(raw_input)
    valid_urls   = [u for u in urls if extract_video_id(u)]
    invalid_urls = [u for u in urls if not extract_video_id(u)]

    if urls:
        col1, col2 = st.columns(2)
        col1.metric("Valid URLs", len(valid_urls))
        col2.metric("Unrecognised", len(invalid_urls))
        if invalid_urls:
            with st.expander("⚠️ Unrecognised URLs (will be skipped)"):
                for u in invalid_urls:
                    st.code(u)

    needs_key   = analyze and not api_key
    run_disabled = not valid_urls or needs_key
    run_button   = st.button(
        f"{'Analyze' if analyze else 'Extract'} "
        f"{len(valid_urls)} video{'s' if len(valid_urls) != 1 else ''}",
        type="primary",
        disabled=run_disabled,
    )

    if needs_key:
        st.warning("Add a Gemini API key in the sidebar, or turn off 'Summarize with Gemini' to extract raw transcripts only.")

    # ── Processing ────────────────────────────────────────────────────────────
    if run_button and valid_urls:

        # Set up Gemini only if summarization is on
        gemini_model = None
        if analyze and api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        results      = []
        progress_bar = st.progress(0, text="Starting…")
        status_text  = st.empty()

        for idx, url in enumerate(valid_urls):
            video_id = extract_video_id(url)
            result   = VideoResult(url=url, video_id=video_id)

            # Step 1 — fetch transcript
            status_text.markdown(
                f"**[{idx+1}/{len(valid_urls)}]** Fetching transcript for `{url}` …"
            )
            try:
                result.transcript = fetch_transcript(video_id)
            except Exception as e:
                result.error = str(e)

            # Step 2 — analyze with Gemini (optional)
            if result.transcript and gemini_model:
                status_text.markdown(
                    f"**[{idx+1}/{len(valid_urls)}]** Analyzing `{url}` with Gemini…"
                )
                try:
                    result.analysis = analyze_transcript(gemini_model, result.transcript)
                except Exception as e:
                    # Don't fail the whole result — just skip analysis
                    st.warning(f"Gemini analysis failed for video {idx+1}: {e}")

            results.append(result)
            progress_bar.progress(
                int((idx + 1) / len(valid_urls) * 100),
                text=f"Processed {idx+1} of {len(valid_urls)} videos",
            )

        progress_bar.empty()
        status_text.empty()

        # ── Results ───────────────────────────────────────────────────────────
        successful = [r for r in results if r.success]
        failed     = [r for r in results if not r.success]
        st.markdown(f"**Done.** {len(successful)} succeeded · {len(failed)} failed")

        for i, result in enumerate(results, 1):
            render_card(result, i)

        # ── Cross-video analysis ──────────────────────────────────────────────
        cross = None
        if gemini_model and len(successful) > 1:
            with st.spinner("Running cross-video analysis…"):
                try:
                    cross = cross_video_analysis(gemini_model, successful)
                    render_cross(cross, len(successful))
                except Exception as e:
                    st.warning(f"Cross-video analysis failed: {e}")

        # ── Download ──────────────────────────────────────────────────────────
        st.divider()
        report = export_report(results, cross)
        st.download_button(
            label="⬇️ Download full report (.txt)",
            data=report,
            file_name="transcript_report.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
