import os
import json

def create_test_dataset():
    """Create a simple test dataset structure"""
    test_dir = "test_data"
    os.makedirs(os.path.join(test_dir, "audio"), exist_ok=True)
    
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
        }
    ]
    
    # Save test cases metadata
    with open(os.path.join(test_dir, "test_cases.json"), "w") as f:
        json.dump(test_cases, f, indent=2)

if __name__ == "__main__":
    create_test_dataset() 