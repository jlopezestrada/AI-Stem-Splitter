import os
import numpy as np
import librosa
import stempeg

# Define base paths
musdb_path = os.path.join('data', 'raw')
base_output_path = os.path.join('data', 'processed')

# Processing parameters
sample_rate = 22050
segment_length = 3
hop_length = 512

def extract_and_preprocess(track, output_dir):
    """
    Extract stems from a given track and preprocess them.

    This function processes each stem in a music track by extracting it,
    splitting it into segments of a predefined length, and then extracting
    MFCC features from each segment. The resulting features are saved as
    NumPy files.

    Parameters:
    track (str): Path to the music track file.

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

        # Split stem into segments
        num_segments = int(np.ceil(len(stem) / (sample_rate * segment_length)))
        for j in range(num_segments):
            start = j * sample_rate * segment_length
            end = start + sample_rate * segment_length
            segment = stem[start:end]

            # Extract features, in this case MFCC
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, hop_length=hop_length)
            feature_file = os.path.join(output_dir, f'{os.path.basename(track)}_stem{i}_segment{j}.npy')
            np.save(feature_file, mfcc)

def main():
    """
    Process each audio track in the musdb_path directory.

    Parameters:
    None

    Returns:
    None
    """
    for root, dir, files in os.walk(musdb_path):
        for file in files:
            track_path = os.path.join(root, file)

            # Determine output directory
            relative_path = os.path.relpath(root, musdb_path)
            output_dir = os.path.join(base_output_path, relative_path)

            # Process each track
            extract_and_preprocess(track_path, output_dir)
            
            # print(track_path, output_dir)

if __name__ == '__main__':
    main()