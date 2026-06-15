
import streamlit as st
from backend import process_video

st.title("YouTube Transcript Analyzer (No API)")

transcript = st.text_area("Paste transcript", height=300)

if st.button("Analyze"):
    result = process_video(transcript)
    st.subheader("Summary")
    st.write(result["summary"])
    st.subheader("Word Count")
    st.write(result["word_count"])
