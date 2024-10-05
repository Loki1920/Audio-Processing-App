import streamlit as st
import requests
from flask import Flask, request, jsonify
from pytubefix import YouTube
from pytubefix.cli import on_progress
import whisper
import yake
import tempfile
import os
from openai import OpenAI
from threading import Thread, Event

# Initialize Flask app
app = Flask(__name__)

# OpenAI API key setup
api_key = os.getenv("api_key")

client = OpenAI(api_key=api_key)

# Whisper model
model = whisper.load_model("base")

# Create an event to signal the Flask app to stop
stop_event = Event()

# Flask route to process YouTube URL
@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    data = request.json
    url = data.get('url')

    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        ys = yt.streams.get_audio_only()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_file = os.path.join(temp_dir, "sample_audio.mp3")

            # Download the audio file
            ys.download(output_path=temp_dir, filename="sample_audio.mp3")

            # Check if the file exists
            if not os.path.exists(audio_file):
                return jsonify({"error": "File not found: " + audio_file})

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
            return jsonify({
                "transcription": final_text,
                "summary": summary_text,
                "title": title_text,
                "keywords": keywords,
                "sentiment": sentiment
            })

    except Exception as e:
        return jsonify({"error": str(e)})

# Function to run the Flask app
def run_flask():
    app.run(debug=False, use_reloader=False)  # Set use_reloader to False to prevent the app from starting twice

# Start the Flask app in a background thread
flask_thread = Thread(target=run_flask)
flask_thread.start()

# Streamlit app
st.title("Audio Processing")

url_input = st.text_input("Enter YouTube Video URL", "")

if st.button("Process"):
    if url_input:
        # Call the Flask API with the YouTube URL
        api_url = "http://127.0.0.1:5000/process_youtube"
        response = requests.post(api_url, json={"url": url_input})

        if response.status_code == 200:
            result = response.json()

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
            st.write("Error:", response.json().get("error", "Unknown error occurred"))
    else:
        st.write("Please enter a valid YouTube URL.")

