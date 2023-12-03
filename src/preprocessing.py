import os
import librosa
import stempeg
import numpy as np
from datetime import datetime

# Define base paths
musdb_path = os.path.join('data', 'raw')
base_output_path = os.path.join('data', 'processed')

# Processing parameters
sample_rate = 22050
segment_length = 5
hop_length = 512

def extract_and_preprocess(track, output_dir, duration=None):
    """
    Extract stems from a given track and preprocess them.

    This function processes each stem in a music track by extracting it,
    splitting it into segments of a predefined length, and then extracting
    MFCC features from each segment. The resulting features are saved as
    NumPy files.

    Parameters:
    track (str): Path to the music track file.
    output_dir (str): Output directory where feature data are going to be saved.
    duration (float, optional): Duration in seconds to process from each stem. 
                                If None, process the entire stem.

    Returns:
    None

    Note:
    - The stems are extracted using the `stempeg` library.
    - The track is assumed to be in a format compatible with `stempeg`.
    - MFCC feature extraction is done using the `librosa` library.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract stems
    stems, _ = stempeg.read_stems(track, sample_rate=sample_rate)

    # Process each stem
    for i in range(stems.shape[0]):
        stem = stems[i]
        print(f"{datetime.now()} - STEM {i}: {track}")

        # Calculate the number of segments to process
        total_duration = len(stem) / sample_rate
        process_duration = duration if duration is not None else total_duration
        num_segments = int(np.ceil(process_duration * sample_rate / (sample_rate * segment_length)))
        print(f"{datetime.now()} - \tNumber of segments: {num_segments}")
        for j in range(num_segments):
            start = j * sample_rate * segment_length
            end = min(start + sample_rate * segment_length, len(stem))
            segment = stem[start:end]

            # Extract features, in this case MFCC
            print(f"{datetime.now()} - \tProcessing feature: {track}")
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, hop_length=hop_length)
            feature_file = os.path.join(output_dir, f'{os.path.basename(track)}_stem{i}_segment{j}.npy')
            np.save(feature_file, mfcc)
            print(f"{datetime.now()} - \tProcessed feature: {track}")
            
def main(duration=None):
    """
    Process each audio track in the musdb_path directory.

    Parameters:
    duration (float, optional): Duration in seconds to process from each track. 
                                If None, process the entire track.

    Returns:
    None
    """
    for root, dirs, files in os.walk(musdb_path):
        for file in files:
            track_path = os.path.join(root, file)

            # Determine output directory
            relative_path = os.path.relpath(root, musdb_path)
            output_dir = os.path.join(base_output_path, relative_path)

            # Process each track with the specified duration
            extract_and_preprocess(track_path, output_dir, duration)
            
            # Uncomment for debugging
            # print(track_path, output_dir)

if __name__ == '__main__':
    main()