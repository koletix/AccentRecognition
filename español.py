import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tkinter import Tk, filedialog, messagebox
from collections import Counter

# Global scaler to normalize feature values
scaler = StandardScaler()

# Function to extract enhanced audio features
def extract_features(audio_file):
    """
    Extract audio features using MFCC and delta-MFCC.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        np.array: Combined audio features.
    """
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    combined = np.vstack((mfcc, delta_mfcc))  # Combine MFCC and delta-MFCC
    combined = np.mean(combined, axis=1)  # Take the mean of features across time
    return combined

# Function to load data and filter out classes with few examples
def load_data(directory):
    """
    Load audio data and assign labels based on filenames.
    Filters classes with fewer than 3 examples.

    Args:
        directory (str): Path to the directory containing audio files.

    Returns:
        tuple: (features, labels)
    """
    features = []
    labels = []
    class_counts = Counter()

    # First pass: Count the number of examples per class
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            label = get_label_from_filename(filename)
            if label is not None:
                class_counts[label] += 1

    # Second pass: Load data only for classes with enough examples
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio_file = os.path.join(directory, filename)
            feature = extract_features(audio_file)
            label = get_label_from_filename(filename)

            # Only include classes with at least 3 examples
            if label is not None and class_counts[label] >= 3:
                features.append(feature)
                labels.append(label)

    return np.array(features), np.array(labels)

# Helper function to assign labels based on filenames
def get_label_from_filename(filename):
    """
    Assign labels based on keywords in the filename.

    Args:
        filename (str): The name of the audio file.

    Returns:
        int: Label corresponding to the accent.
    """
    if 'venezuela' in filename.lower():
        return 1
    elif 'uruguay' in filename.lower():
        return 2
    elif 'usa' in filename.lower():
        return 3
    elif 'spain' in filename.lower():
        return 4
    elif 'puerto' in filename.lower():
        return 5
    elif 'peru' in filename.lower():
        return 6
    elif 'nicaragua' in filename.lower():
        return 7
    elif 'panama' in filename.lower():
        return 8
    elif 'paraguay' in filename.lower():
        return 9
    elif 'mexico' in filename.lower():
        return 10
    elif 'honduras' in filename.lower():
        return 11
    elif 'salvador' in filename.lower():
        return 12
    elif 'guatemala' in filename.lower():
        return 13
    elif 'dominican' in filename.lower():
        return 14
    elif 'ecuador' in filename.lower():
        return 15
    elif 'cuba' in filename.lower():
        return 16
    elif 'colombia' in filename.lower():
        return 17
    elif 'costa' in filename.lower():
        return 18
    elif 'chile' in filename.lower():
        return 19
    elif 'bolivia' in filename.lower():
        return 20
    elif 'argentina' in filename.lower():
        return 21
    else:
        return None

# Directory containing the audio files for training
directory = r"C:\Users\kolet\Downloads\AccentRecognition\Spanish"

# Load and normalize features
features, labels = load_data(directory)
features = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sync labels with detected classes
detected_classes = model.classes_
all_labels = [
    "Venezuela", "Uruguay", "USA", "Spain", "Puerto Rico", "Peru", 
    "Nicaragua", "Panama", "Paraguay", "Mexico", "Honduras", "El Salvador", 
    "Guatemala", "Dominican Republic", "Ecuador", "Cuba", "Colombia", 
    "Costa Rica", "Chile", "Bolivia", "Argentina"
]
filtered_labels = [all_labels[i - 1] for i in detected_classes]

# GUI for selecting and analyzing a file
def gui_classify_audio():
    """
    Opens a GUI for the user to select an audio file, then classifies it.
    """
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    messagebox.showinfo("Audio Classifier", "Please select a WAV audio file to analyze.")
    
    # Open file dialog for the user to select a file
    audio_file = filedialog.askopenfilename(
        title="Select a WAV file",
        filetypes=[("WAV Files", "*.wav")]
    )
    
    if not audio_file:
        messagebox.showerror("Error", "No file selected. Exiting.")
        return

    # Classify the selected file
    try:
        feature = extract_features(audio_file)
        feature = scaler.transform([feature])
        probabilities = model.predict_proba(feature)[0]
        
        result_message = ""
        for i, label in enumerate(filtered_labels):
            result_message += f"{label}: {probabilities[i] * 100:.2f}%\n"
        max_index = np.argmax(probabilities)
        result_message += f"\nPredicted Accent: {filtered_labels[max_index]}"
        
        messagebox.showinfo("Classification Result", result_message)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Run the GUI
if __name__ == "__main__":
    gui_classify_audio()



