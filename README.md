# Audio Processing App

## Overview

This project is a Audio Processing App that allows users to input a YouTube video URL and processes the audio for various analyses. The app utilizes Flask as a backend API for audio processing and Streamlit for a user-friendly interface. Key features include transcription, summarization, keyword extraction, title generation, and sentiment analysis of the audio content.

## Features

- **Transcription**: Converts audio from the YouTube video to text using Whisper.
- **Summary Generation**: Creates a concise summary of the transcribed text.
- **Keyword Extraction**: Extracts relevant keywords from the transcribed text.
- **Title Generation**: Generates an appropriate title based on the transcribed content.
- **Sentiment Analysis**: Analyzes the sentiment of the transcribed text using OpenAI's API.

## Technologies Used

- Python
- Flask
- Streamlit
- Whisper (for transcription)
- YAKE (for keyword extraction)
- OpenAI API (for summarization and sentiment analysis)
- pytubefix (for downloading YouTube audio)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/Loki1920/youtube-video-processing-app.git
cd youtube-video-processing-app
