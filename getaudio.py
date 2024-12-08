import pandas as pd
import urllib.request
import os
import sys
from pydub import AudioSegment


class GetAudio:

    def __init__(self, csv_filepath, destination_folder='audio/', debug=True):
        '''
        Initializes GetAudio class object
        :param csv_filepath (str): Path to the CSV file
        :param destination_folder (str): Folder where audio files will be saved
        :param debug (bool): Outputs status indicators to console when True
        '''
        self.csv_filepath = csv_filepath
        self.audio_df = pd.read_csv(csv_filepath)
        self.destination_folder = destination_folder
        self.base_url = 'http://accent.gmu.edu/soundtracks/'  # Base URL for audio files
        self.debug = debug

    def check_path(self):
        '''
        Checks if self.destination_folder exists. If not, creates it.
        '''
        if not os.path.exists(self.destination_folder):
            print(f'{self.destination_folder} does not exist, creating...')
            os.makedirs(self.destination_folder)

    def get_audio(self):
        '''
        Retrieves all audio files using 'language_num' column of self.audio_df
        If audio file already exists, move on to the next
        :return (int): Number of audio files downloaded
        '''
        self.check_path()

        counter = 0

        for _, row in self.audio_df.iterrows():
            language_num = row['language_num']  # 'mandarin1', 'english2', etc.
            audio_url = f"{self.base_url}{language_num}.mp3"
            file_name = f"{language_num}.mp3"
            file_path = os.path.join(self.destination_folder, file_name)

            # Skip if the file already exists
            if os.path.exists(file_path):
                print(f"File already exists: {file_path}")
                continue

            try:
                # Download the MP3 file
                print(f"Downloading audio from {audio_url}...")
                urllib.request.urlretrieve(audio_url, file_path)
                print(f"Downloaded: {file_path}")

                # Convert to WAV using pydub
                sound = AudioSegment.from_file(file_path, format="mp3")
                wav_file_path = file_path.replace('.mp3', '.wav')
                sound.export(wav_file_path, format="wav")
                print(f"Converted to WAV: {wav_file_path}")
                counter += 1
            except Exception as e:
                print(f"Error downloading {audio_url}: {e}")

        return counter


if __name__ == '__main__':
    '''
    Example console command
    python getaudio.py bio_metadata.csv
    '''
    if len(sys.argv) < 2:
        print("Usage: python getaudio.py <path_to_csv>")
    else:
        csv_file = sys.argv[1]
        ga = GetAudio(csv_filepath=csv_file)
        downloaded_count = ga.get_audio()
        print(f"Total files downloaded: {downloaded_count}")
