"""
Create YOLOv8 data.yaml configuration from dataset structure
"""

import yaml
from pathlib import Path

def create_data_yaml():
    """Create data.yaml for YOLOv8 training."""
    
    # Define paths
    base_path = Path(__file__).parent.parent.absolute()
    
    config = {
        'path': str(base_path / 'data' / 'dataset'),  # dataset root directory
        'train': 'train',
        'val': 'valid',
        'test': 'test',
        
        # Track error classes
        'names': {
            0: 'track_crack',
            1: 'missing_bolt',
            2: 'ballast_issue',
            3: 'broken_sleeper',
            4: 'gauge_error',
            5: 'fishplate_issue',
            6: 'crushed_stone'
        },
        
        # Number of classes
        'nc': 7
    }
    
    output_path = 'data.yaml'
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[OK] Created {output_path}")
    print(f"[DATASET] Dataset root: {config['path']}")
    print(f"[CLASSES] Number of classes: {config['nc']}")

if __name__ == "__main__":
    create_data_yaml()

