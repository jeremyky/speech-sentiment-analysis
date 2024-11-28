import os
import json
import soundfile as sf
import numpy as np

def create_test_directory():
    """Create test data directory structure"""
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "audio"), exist_ok=True)
    
    # Test cases with audio files
    test_cases = [
        {
            "text": "I am really happy about this amazing result!",
            "ground_truth_emotions": ["joy"],
            "audio_file": "happy_test.wav"
        },
        {
            "text": "This makes me so angry and frustrated.",
            "ground_truth_emotions": ["anger", "disgust"],
            "audio_file": "angry_test.wav"
        },
        {
            "text": "I'm feeling quite sad and disappointed today.",
            "ground_truth_emotions": ["sadness"],
            "audio_file": "sad_test.wav"
        },
        {
            "text": "Wow, this is such a surprising and exciting development!",
            "ground_truth_emotions": ["surprise", "joy"],
            "audio_file": "surprise_test.wav"
        }
    ]
    
    # Save test cases metadata
    with open(os.path.join(test_dir, "test_cases.json"), "w") as f:
        json.dump(test_cases, f, indent=2)
    
    print("Test directory created at test_data/")
    print("Please record audio files for each test case using the GUI")
    print("Save them in test_data/audio/ with the specified filenames")

if __name__ == "__main__":
    create_test_directory() 