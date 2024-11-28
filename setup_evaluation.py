import os
import subprocess
import sys

def setup_environment():
    """Set up the evaluation environment"""
    print("Setting up evaluation environment...")
    
    # Create datasets directory
    os.makedirs("datasets", exist_ok=True)
    
    # Clone CMU-MultimodalSDK
    sdk_path = "datasets/CMU-MultimodalSDK"
    if not os.path.exists(sdk_path):
        print("Cloning CMU-MultimodalSDK...")
        subprocess.run([
            "git", "clone",
            "https://github.com/A2Zadeh/CMU-MultimodalSDK.git",
            sdk_path
        ])
        
        # Install SDK
        print("Installing CMU-MultimodalSDK...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", sdk_path
        ])
    
    # Install additional requirements
    print("Installing additional requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "mmsdk>=1.0.0",
        "soundfile>=0.12.1"
    ])
    
    print("\nSetup complete! You can now run:")
    print("PYTHONPATH=. python evaluate.py")

if __name__ == "__main__":
    setup_environment() 