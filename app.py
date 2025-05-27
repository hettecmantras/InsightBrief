import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import yt_dlp
import time
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

# Configure Google API
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page config
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")

# Initialize agent
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodal_Agent = initialize_agent()

# ---- FUNCTION TO DOWNLOAD YOUTUBE VIDEO ----
def download_youtube_video(url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = f"{temp_dir}/video.%(ext)s"

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'noplaylist': True,
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for file in os.listdir(temp_dir):
            if file.endswith('.mp4'):
                return os.path.join(temp_dir, file)

        raise Exception("Video download failed or format not found.")

    except Exception as e:
        raise Exception(f"Failed to download YouTube video: {e}")

# ---- VIDEO INPUT (UPLOAD OR YOUTUBE) ----
st.subheader("Step 1: Upload a Video or Paste a YouTube URL")
uploaded_file = st.file_uploader("Upload a local video file", type=["mp4", "mov", "avi"])
youtube_url = st.text_input("Or enter a YouTube video URL")

video_path = None

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("Video uploaded successfully!")
    st.video(video_path)

elif youtube_url:
    try:
        with st.spinner("Downloading YouTube video..."):
            video_path = download_youtube_video(youtube_url)
            st.success("YouTube video downloaded successfully!")
            st.video(video_path)
    except Exception as e:
        st.error(str(e))

# ---- USER PROMPT ----
user_query = st.text_area(
    "What insights are you seeking from the video?",
    placeholder="Ask anything about the video content...",
    help="Provide specific questions or insights you want from the video."
)

# ---- ANALYZE VIDEO ----
if st.button("🔍 Analyze Video", key="analyze_video_button"):
    if not video_path:
        st.warning("Please upload a video file or provide a YouTube URL.")
    elif not user_query:
        st.warning("Please enter your query.")
    else:
        try:
            with st.spinner("Uploading and analyzing video..."):
                processed_video = upload_file(video_path)
                while processed_video.state.name == "PROCESSING":
                    time.sleep(1)
                    processed_video = get_file(processed_video.name)

                prompt = (
                    f"""
                    Analyze the uploaded video for content and context.
                    Respond to the following query using video insights and web search if needed:
                    {user_query}

                    Provide a detailed, actionable, and user-friendly answer.
                    """
                )

                response = multimodal_Agent.run(prompt, videos=[processed_video])

            st.subheader("Analysis Result")
            st.markdown(response.content)

            st.download_button(
                label="💾 Save Summary",
                data=response.content,
                file_name="video_summary.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if video_path and os.path.exists(video_path):
                Path(video_path).unlink(missing_ok=True)

# ---- STYLING ----
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
