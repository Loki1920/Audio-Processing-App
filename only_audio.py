import streamlit as st
import whisper
import yake
import tempfile
import os
from openai import OpenAI

# OpenAI API key setup
api_key = os.getenv("api_key")

client = OpenAI(api_key=api_key)

# Whisper model
model = whisper.load_model("base")

st.title("Audio Processing App")


st.markdown("### **Upload Audio File**")
uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])


# Streamlit file uploader for audio file
#uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])

if st.button("Process"):
    if uploaded_file is not None:
        try:
            # Create a temporary directory to store the uploaded audio
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_file_path = os.path.join(temp_dir, uploaded_file.name)

                # Save the uploaded file to the temporary directory
                with open(audio_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Transcribe the audio using Whisper
                result = model.transcribe(audio_file_path)
                final_text = result['text']

                # Extract summary using OpenAI GPT-3.5
                system = [{"role": "system", "content": "You are Summary AI."}]
                user = [{"role": "user", "content": f"Summarize this briefly:\n\n{final_text}"}]

                chat_completion = client.chat.completions.create(
                    messages=system + user,
                    model="gpt-3.5-turbo",
                    max_tokens=500,
                    top_p=0.9,
                )
                summary_text = chat_completion.choices[0].message.content

                # Generate a title based on the text
                user = [{"role": "user", "content": f"Give a title based on the given text:\n\n{final_text}"}]
                chat_completion = client.chat.completions.create(
                    messages=system + user,
                    model="gpt-3.5-turbo",
                    max_tokens=500,
                    top_p=0.9,
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

                # Perform sentiment analysis using OpenAI GPT-3.5
                sentiment_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in sentiment analysis."},
                        {"role": "user", "content": f"Text: {final_text}"}
                    ],
                    temperature=1
                )
                sentiment = sentiment_response.choices[0].message.content.strip()

                # Display the results
                st.write("### Transcription:")
                st.write(final_text)

                st.write("### Summary:")
                st.write(summary_text)

                st.write("### Title:")
                st.write(title_text)

                st.write("### Extracted Keywords:")
                for keyword, score in keywords:
                    st.write(f"{keyword} - {score:.4f}")

                st.write("### Sentiment Analysis:")
                st.write(sentiment)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please upload a valid audio file.")
