"""
YouTube Transcript Analyzer — Streamlit App
============================================
Fetches transcripts for a batch of YouTube URLs, summarizes each one,
then produces a cross-video thematic analysis using the Anthropic API.

Setup:
    pip install streamlit anthropic youtube-transcript-api python-dotenv

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push to GitHub, connect at share.streamlit.io, add ANTHROPIC_API_KEY
    under Settings → Secrets.
"""

import re
import os
import time
import json
from dataclasses import dataclass
from typing import Optional

import anthropic
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()

# Best-value model for transcript analysis — fast and cost-efficient.
# See: https://platform.claude.com/docs/en/about-claude/models/overview
MODEL = "claude-haiku-4-5-20251001"

MAX_TRANSCRIPT_CHARS = 8_000   # characters sent to Claude per video
MAX_CROSS_CHARS      = 2_000   # characters per video for cross-video analysis
RETRY_ATTEMPTS       = 3
RETRY_DELAY_SECS     = 2

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
    """Parse a YouTube URL and return the 11-character video ID, or None."""
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",  # bare ID
    ]
    for pattern in patterns:
        m = re.search(pattern, url.strip())
        if m:
            return m.group(1)
    return None


def parse_urls(raw: str) -> list:
    """Split newline- or comma-separated URLs and strip blanks."""
    return [u.strip() for u in re.split(r"[\n,]+", raw) if u.strip()]


def fetch_transcript(video_id: str) -> str:
    """
    Fetch the transcript for a video ID using youtube-transcript-api v1.2+.

    v1.2.0 removed list_transcripts / get_transcript / get_transcripts entirely.
    The new API is:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)               # fetches default/best language
        text = " ".join(s.text for s in fetched)

    To prefer English and fall back gracefully we use ytt.list() then ytt.fetch()
    with an explicit language preference list.
    """
    ytt = YouTubeTranscriptApi()

    # Try fetching English directly first — fastest path
    try:
        fetched = ytt.fetch(video_id, languages=["en", "en-US", "en-GB", "en-CA", "en-AU"])
        return " ".join(s.text for s in fetched)
    except Exception:
        pass

    # Fall back: list available transcripts, pick any and translate to English
    try:
        transcript_list = ytt.list(video_id)
        # transcript_list is iterable; grab the first available
        for transcript in transcript_list:
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


def call_claude(client: anthropic.Anthropic, prompt: str, retries: int = RETRY_ATTEMPTS) -> str:
    """
    Call the Claude API with simple retry logic for transient errors.
    Returns the response text.
    """
    for attempt in range(1, retries + 1):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.RateLimitError:
            if attempt < retries:
                time.sleep(RETRY_DELAY_SECS * attempt)
            else:
                raise
        except anthropic.APIStatusError as e:
            if attempt < retries and e.status_code >= 500:
                time.sleep(RETRY_DELAY_SECS * attempt)
            else:
                raise


def analyze_transcript(client: anthropic.Anthropic, transcript: str) -> dict:
    """
    Ask Claude to summarize and extract structured insights from a transcript.
    Returns a dict with keys: summary, key_points, themes, sentiment, speaker_tone.
    Falls back to raw text if JSON parsing fails.
    """
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
    raw = call_claude(client, prompt)
    # Strip stray code fences in case the model adds them
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Graceful fallback: wrap raw response so the UI still has something to show
        return {"summary": raw, "key_points": [], "themes": [], "sentiment": "unknown", "speaker_tone": ""}


def cross_video_analysis(client: anthropic.Anthropic, results: list) -> dict:
    """
    Ask Claude to identify patterns, narratives, and themes across all videos.
    Returns a dict with keys: overarching_themes, common_narrative,
    notable_patterns, content_diversity, recommendations.
    """
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
    raw = call_claude(client, prompt)
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"common_narrative": raw, "overarching_themes": [], "notable_patterns": "",
                "content_diversity": "", "recommendations": ""}


def export_results_as_text(results: list, cross: Optional[dict]) -> str:
    """Serialize all results to a plain-text report for download."""
    lines = ["YouTube Transcript Analysis Report", "=" * 50, ""]

    for i, r in enumerate(results, 1):
        lines.append(f"VIDEO {i}: {r.url}")
        lines.append("-" * 40)
        if r.error:
            lines.append(f"ERROR: {r.error}")
        elif r.analysis:
            lines.append(f"Summary: {r.analysis.get('summary', '')}")
            lines.append(f"Sentiment: {r.analysis.get('sentiment', '')}")
            lines.append(f"Speaker Tone: {r.analysis.get('speaker_tone', '')}")
            lines.append("\nKey Points:")
            for kp in r.analysis.get("key_points", []):
                lines.append(f"  • {kp}")
            lines.append("\nThemes: " + ", ".join(r.analysis.get("themes", [])))
            lines.append("\nRaw Transcript (first 1000 chars):")
            lines.append((r.transcript or "")[:1000] + "…")
        lines.append("")

    if cross:
        lines += [
            "CROSS-VIDEO ANALYSIS",
            "=" * 50,
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

def render_analysis_card(result: VideoResult, index: int):
    """Render a single video's result as an expandable Streamlit section."""
    label = f"Video {index}: `{result.url}`"

    if result.error:
        with st.expander(f"❌ {label}", expanded=False):
            st.error(result.error)
        return

    if not result.analysis:
        with st.expander(f"⏳ {label}", expanded=False):
            st.info("Processing…")
        return

    a = result.analysis
    with st.expander(f"✅ {label}", expanded=True):
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

        with st.expander("📄 Raw transcript", expanded=False):
            st.text_area(
                label="transcript",
                value=result.transcript or "",
                height=200,
                label_visibility="collapsed",
            )


def render_cross_analysis(cross: dict, n_videos: int):
    """Render the cross-video analysis panel."""
    st.divider()
    st.subheader(f"🔍 Cross-Video Analysis ({n_videos} videos)")

    if cross.get("overarching_themes"):
        st.markdown("**Overarching Themes**")
        st.markdown("  ".join(f"`{t}`" for t in cross["overarching_themes"]))

    if cross.get("common_narrative"):
        st.markdown("**Common Narrative**")
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
        st.markdown("**Recommendations**")
        st.success(cross["recommendations"])


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="YouTube Transcript Analyzer",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 YouTube Transcript Analyzer")
    st.caption("Paste YouTube URLs to extract transcripts, summarize each video, and identify cross-video patterns.")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        api_key = st.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Your key from console.anthropic.com — or set ANTHROPIC_API_KEY in .env",
        )

        st.markdown("---")
        st.markdown(
            "**Model:** `claude-haiku-4-5`\n\n"
            "Fast and cost-efficient for bulk transcript analysis. "
            "Swap `MODEL` in the source for Sonnet if you need deeper reasoning."
        )
        st.markdown("---")
        st.markdown(
            "**Tips**\n"
            "- Videos must have captions (auto-generated counts)\n"
            "- Private / age-restricted videos will fail\n"
            "- Long videos may be truncated before analysis"
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

    urls = parse_urls(raw_input)
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

    run_disabled = not valid_urls or not api_key
    run_button   = st.button(
        f"Analyze {len(valid_urls)} video{'s' if len(valid_urls) != 1 else ''}",
        type="primary",
        disabled=run_disabled,
    )

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to continue.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    if run_button and valid_urls and api_key:
        client  = anthropic.Anthropic(api_key=api_key)
        results = []

        progress_bar = st.progress(0, text="Starting…")
        status_text  = st.empty()

        for idx, url in enumerate(valid_urls):
            video_id = extract_video_id(url)
            result   = VideoResult(url=url, video_id=video_id)

            # Step 1 — fetch transcript
            status_text.markdown(f"**[{idx+1}/{len(valid_urls)}]** Fetching transcript for `{url}` …")
            try:
                result.transcript = fetch_transcript(video_id)
            except Exception as e:
                result.error = str(e)

            # Step 2 — analyze with Claude
            if result.transcript:
                status_text.markdown(f"**[{idx+1}/{len(valid_urls)}]** Analyzing `{url}` with Claude…")
                try:
                    result.analysis = analyze_transcript(client, result.transcript)
                except anthropic.AuthenticationError:
                    st.error("Invalid API key. Please check your key in the sidebar.")
                    st.stop()
                except Exception as e:
                    result.error = f"Claude analysis failed: {e}"

            results.append(result)
            progress_bar.progress(
                int((idx + 1) / len(valid_urls) * 100),
                text=f"Processed {idx+1} of {len(valid_urls)} videos",
            )

        progress_bar.empty()
        status_text.empty()

        # ── Per-video results ─────────────────────────────────────────────────
        successful = [r for r in results if r.success]
        failed     = [r for r in results if not r.success]

        st.markdown(
            f"**Done.** {len(successful)} succeeded · {len(failed)} failed"
        )

        for i, result in enumerate(results, 1):
            render_analysis_card(result, i)

        # ── Cross-video analysis ──────────────────────────────────────────────
        cross = None
        if len(successful) > 1:
            with st.spinner("Running cross-video analysis…"):
                try:
                    cross = cross_video_analysis(client, successful)
                    render_cross_analysis(cross, len(successful))
                except Exception as e:
                    st.warning(f"Cross-video analysis failed: {e}")

        # ── Download button ───────────────────────────────────────────────────
        report = export_results_as_text(results, cross)
        st.download_button(
            label="⬇️ Download full report (.txt)",
            data=report,
            file_name="transcript_analysis.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
