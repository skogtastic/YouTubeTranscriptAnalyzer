
import streamlit as st
from backend import process_youtube_url

st.title("YouTube Transcript Analyzer")

url = st.text_input("Paste a YouTube URL")

if st.button("Fetch & Analyze"):
    try:
        result = process_youtube_url(url)

        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Word Count")
        st.write(result["word_count"])

        with st.expander("Transcript"):
            st.write(result["raw_transcript"])

    except Exception as e:
        st.error(str(e))
