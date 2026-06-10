import streamlit as st
from backend import process_video

st.set_page_config(page_title="YouTube Transcript & Summary Tool")

st.title("YouTube Transcript & Summary Tool")

urls = st.text_area(
    "Paste one YouTube URL per line",
    height=200
)

if st.button("Generate Transcript"):
    url_list = [u.strip() for u in urls.split("\n") if u.strip()]

    for url in url_list:
        with st.spinner(f"Processing {url}"):
            try:
                result = process_video(url)

                st.header(result["title"])

                st.subheader("Summary")
                st.write(result["summary"])

                st.subheader("Video Structure")
                st.write(result["structure"])

                st.subheader("Transcript")
                st.text_area("", result["transcript"], height=300)

            except Exception as e:
                st.error(str(e))
