# Spanish Accent Recognition Using MFCCs

This project uses machine learning to classify English accents from different Spanish-speaking countries based on audio recordings. It involves metadata scraping, audio processing, and machine learning for accent classification.

---

## Project Structure

### 1. Metadata Collection
- **File**: `fromwebsite.py`
- **Purpose**: Scrapes metadata from the GMU Speech Accent Archive for speakers of specified languages.
- **Output**: `bio_metadata.csv` containing metadata for specified languages.

### 2. Audio Retrieval and Conversion
- **File**: `getaudio.py`
- **Purpose**: Downloads MP3 audio files and converts them to WAV format.
- **Output**: A directory of WAV files for further processing.

### 3. Accent Classification
- **File**: `español.py`
- **Purpose**: Extracts MFCC features, trains a machine learning model, and provides a GUI for classifying new audio files.
- **Output**: GUI for user interaction and classification results.

---

## Prerequisites

- **Python Version**: Requires Python 3.8 or later.
- **External Tool**: `ffmpeg` is required for audio file conversion. Install it using:
  - On Linux: `sudo apt-get install ffmpeg`
  - On Windows/Mac: [Download FFmpeg](https://ffmpeg.org/download.html)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/koletix/AccentRecognition.git
2. Install required libraries:
    ```bash
    pip install -r requirements.txt



## How to Run

### Step 1: Collect Metadata
Run `fromwebsite.py` to scrape metadata and create a CSV file.
```bash
python fromwebsite.py bio_metadata.csv [language1] [language2] ...

### Step 2: Download and Convert Audio
Run `getaudio.py` to scrape metadata and create a CSV file.
```bash
python getaudio.py bio_metadata.csv

### Step 3: Train and Classify Accents
Run `español.py` to train the model and classify new audio files via a GUI.
```bash
python español.py
