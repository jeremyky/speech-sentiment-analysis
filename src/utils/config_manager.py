import yaml
import os
from pathlib import Path

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Get project root directory
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.root_dir / 'config'
        
        self.load_configs()
        self._initialized = True

    def load_configs(self):
        """Load all configuration files"""
        try:
            with open(self.config_dir / 'config.yaml', 'r') as f:
                self.main_config = yaml.safe_load(f)
                
            with open(self.config_dir / 'emotions.yaml', 'r') as f:
                self.emotion_config = yaml.safe_load(f)
                
        except FileNotFoundError as e:
            print(f"Configuration file not found: {e}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading configurations: {e}")
            raise

    def get_emotion_config(self):
        """Get current emotion configuration based on selected mode"""
        mode = self.main_config['model']['emotion_mode']
        return self.emotion_config['emotion_modes'][mode]

    def get_model_path(self):
        """Get appropriate model path based on emotion mode"""
        mode = self.main_config['model']['emotion_mode']
        return self.main_config['model']['sentiment_models'][mode]

    def get_audio_config(self):
        """Get audio configuration"""
        return self.main_config['audio']

    def get_display_config(self):
        """Get display configuration"""
        return self.main_config['display']
        
    def switch_emotion_mode(self, mode):
        """Switch between emotion detection modes"""
        if mode not in self.emotion_config['emotion_modes']:
            raise ValueError(f"Invalid emotion mode: {mode}")
            
        self.main_config['model']['emotion_mode'] = mode
        # Save updated config
        with open(self.config_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.main_config, f)
            
    def get_emotion_threshold(self, emotion_label):
        """Get threshold for a specific emotion"""
        mode = self.main_config['model']['emotion_mode']
        emotions = self.emotion_config['emotion_modes'][mode]['emotions']
        for emotion in emotions:
            if emotion['label'] == emotion_label:
                return emotion['threshold']
        return self.emotion_config['emotion_modes'][mode]['model_config']['threshold'] 