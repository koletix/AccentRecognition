# Spanish Accent Recognition Using MFCCs

This project uses machine learning to classify English accents from different Spanish-speaking countries. It includes scripts for data collection, audio processing, and model training.

## Project Structure

### 1. Metadata Collection
- **File**: `fromwebsite.py`
- **Purpose**: Scrapes metadata from the GMU Speech Accent Archive for speakers of specified languages.

### 2. Audio Retrieval and Conversion
- **File**: `getaudio.py`
- **Purpose**: Downloads audio files in MP3 format and converts them to WAV format for processing.

### 3. Accent Classification
- **File**: `español.py`
- **Purpose**: Extracts MFCC features, trains a machine learning model, and provides a GUI for classifying new audio files.

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
