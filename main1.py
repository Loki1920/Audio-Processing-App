import streamlit as st
from pytubefix import YouTube
from pytubefix.cli import on_progress
import whisper
import yake
import tempfile
import os
from openai import OpenAI
import psutil
import signal

# OpenAI API key setup
api_key = os.getenv("api_key")
client = OpenAI(api_key=api_key)

# Whisper model
model = whisper.load_model("base")

# Function to process YouTube URL and perform transcription, summarization, etc.
def process_youtube(url):
    try:
        yt = YouTube(url, on_progress_callback=on_progress, use_po_token=True)
        ys = yt.streams.get_audio_only()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_file = os.path.join(temp_dir, "sample_audio.mp3")

            # Download the audio file
            ys.download(output_path=temp_dir, filename="sample_audio.mp3")

            # Check if the file exists
            if not os.path.exists(audio_file):
                st.write({"error": "File not found: " + audio_file})
                return

            # Transcribe the audio using Whisper
            result = model.transcribe(audio_file)
            final_text = result['text']

            # Extract summary from given text
            system = [{"role": "system", "content": "You are Summary AI."}]
            user = [{"role": "user", "content": f"Summarize this briefly:\n\n{final_text}"}]

            chat_completion = client.chat.completions.create(
                messages=system + user,
                model="gpt-3.5-turbo",
                max_tokens=500, top_p=0.9,
            )
            summary_text = chat_completion.choices[0].message.content

            # Give the title based on text
            user = [{"role": "user", "content": f"Give a title based on given text:\n\n{final_text}"}]
            chat_completion = client.chat.completions.create(
                messages=system + user,
                model="gpt-3.5-turbo",
                max_tokens=500, top_p=0.9,
            )
            title_text = chat_completion.choices[0].message.content

            # Extract keywords using YAKE
            language = "en"
            max_ngram_size = 3
            deduplication_threshold = 0.9
            deduplication_algo = 'seqm'
            window_size = 1
            num_of_keywords = 20

            kw_extractor = yake.KeywordExtractor(
                lan=language,
                n=max_ngram_size,
                dedupLim=deduplication_threshold,
                dedupFunc=deduplication_algo,
                windowsSize=window_size,
                top=num_of_keywords,
                features=None
            )
            keywords = kw_extractor.extract_keywords(final_text)

            # Perform sentiment analysis with OpenAI GPT-3.5
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in sentiment analysis."
                    },
                    {
                        "role": "user",
                        "content": f"Text: {final_text}"
                    }
                ],
                temperature=1
            )
            sentiment = response.choices[0].message.content.strip()

            # Return results
            return {
                "transcription": final_text,
                "summary": summary_text,
                "title": title_text,
                "keywords": keywords,
                "sentiment": sentiment
            }

    except Exception as e:
        st.write({"error": str(e)})


# Streamlit app
st.title("Audio Processing")

# Input for YouTube URL
url_input = st.text_input("Enter YouTube Video URL", "")

# Process the video when the button is clicked
if st.button("Process"):
    if url_input:
        # Process the YouTube video and get the results
        result = process_youtube(url_input)

        if result:
            # Display the results
            st.write("### Transcription:")
            st.write(result['transcription'])

            st.write("### Summary:")
            st.write(result['summary'])

            st.write("### Title:")
            st.write(result['title'])

            st.write("### Extracted Keywords:")
            for keyword, score in result['keywords']:
                st.write(f"{keyword} - {score:.4f}")

            st.write("### Sentiment Analysis:")
            st.write(result['sentiment'])
        else:
            st.write("Error processing the video.")
    else:
        st.write("Please enter a valid YouTube URL.")

# Function to kill any process running on a specific port (e.g., Flask instances)
def kill_process_on_port(port):
    for proc in psutil.process_iter():
        for conn in proc.connections(kind='inet'):
            if conn.laddr.port == port:
                proc.send_signal(signal.SIGTERM)  # or signal.SIGKILL
                st.write(f"Process running on port {port} has been killed.")


# Button to kill all servers running on port 5000
if st.button("Kill All Servers"):
    kill_process_on_port(5000)
