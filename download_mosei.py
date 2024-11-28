import gdown
import os

def download_mosei_data():
    """Download MOSEI dataset from Google Drive"""
    print("Downloading MOSEI dataset...")
    
    # Create directory if it doesn't exist
    save_dir = "MultiBench/datasets/affect/processed"
    os.makedirs(save_dir, exist_ok=True)
    
    # Google Drive file ID for MOSEI data
    file_id = "1O-FoQb_uC0JqXF3CYlK4Lc4G_qBbdvZm"  # This is the ID for mosei_senti_data.pkl
    output_path = os.path.join(save_dir, "mosei_senti_data.pkl")
    
    # Download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Download the file
        gdown.download(url, output_path, quiet=False)
        print(f"\nMOSEI data downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Error downloading MOSEI data: {e}")
        
if __name__ == "__main__":
    download_mosei_data() 