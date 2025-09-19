"""
DINOv3 Utilities for MuSc Project
Helper functions for loading and using DINOv3 models
"""

import torch
import torch.nn as nn
import os
import urllib.request
from typing import Dict, Optional


class DINOv3ModelManager:
    """Manages DINOv3 model loading and pretrained weights"""
    
    DINOV3_URLS = {
        'dinov3_vits14': 'https://dl.fbaipublicfiles.com/dinov3/dinov3_vits14/dinov3_vits14_pretrain.pth',
        'dinov3_vitb14': 'https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb14/dinov3_vitb14_pretrain.pth', 
        'dinov3_vitl14': 'https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl14/dinov3_vitl14_pretrain.pth',
        'dinov3_vitg14': 'https://dl.fbaipublicfiles.com/dinov3/dinov3_vitg14/dinov3_vitg14_pretrain.pth',
    }
    
    MODEL_CONFIGS = {
        'dinov3_vits14': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'dinov3_vitb14': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'dinov3_vitl14': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
        'dinov3_vitg14': {'embed_dim': 1536, 'depth': 40, 'num_heads': 24},
    }
    
    def __init__(self, cache_dir: str = './pretrained_models'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_pretrained_weights(self, model_name: str, force_download: bool = False) -> str:
        """Download pretrained weights for DINOv3 model"""
        if model_name not in self.DINOV3_URLS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.DINOV3_URLS.keys())}")
        
        url = self.DINOV3_URLS[model_name]
        filename = f"{model_name}_pretrain.pth"
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath) or force_download:
            print(f"Downloading {model_name} pretrained weights...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded to: {filepath}")
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
                print("Please download manually from Meta's official repository")
                return None
        else:
            print(f"Using cached weights: {filepath}")
        
        return filepath
    
    def load_pretrained_model(self, model_name: str, device: torch.device = None) -> nn.Module:
        """Load a pretrained DINOv3 model"""
        from models.backbone import dinov3_vision_transformer as dinov3_vits
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get model configuration
        config = self.MODEL_CONFIGS.get(model_name, {})
        
        # Create model
        if 'vits' in model_name:
            model = dinov3_vits.dinov3_vit_small(patch_size=14, **config)
        elif 'vitb' in model_name:
            model = dinov3_vits.dinov3_vit_base(patch_size=14, **config)
        elif 'vitl' in model_name:
            model = dinov3_vits.dinov3_vit_large(patch_size=14, **config)
        elif 'vitg' in model_name:
            model = dinov3_vits.dinov3_vit_giant(patch_size=14, **config)
        else:
            raise ValueError(f"Unknown model variant: {model_name}")
        
        # Download and load weights
        weights_path = self.download_pretrained_weights(model_name)
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights for {model_name}")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Using model with random initialization")
        
        model.to(device)
        model.eval()
        
        # Freeze parameters for feature extraction
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def get_optimal_settings(self, model_name: str) -> Dict:
        """Get optimal settings for DINOv3 model"""
        base_settings = {
            'img_resize': 518,
            'batch_size': 4,
            'feature_layers': [5, 11, 17, 23],
            'r_list': [1, 3, 5]
        }
        
        # Adjust based on model size
        if 'vits' in model_name:
            base_settings.update({'batch_size': 8})
        elif 'vitl' in model_name:
            base_settings.update({'batch_size': 2})
        elif 'vitg' in model_name:
            base_settings.update({'batch_size': 1, 'img_resize': 420})
        
        return base_settings


def setup_dinov3_config(model_name: str = 'dinov3_vitb14', dataset: str = 'mvtec_ad') -> Dict:
    """Setup configuration for DINOv3 models"""
    manager = DINOv3ModelManager()
    optimal_settings = manager.get_optimal_settings(model_name)
    
    config = {
        'datasets': {
            'dataset_name': dataset,
            'data_path': f'./data/{dataset}/',
            'class_name': 'ALL',
            'img_resize': optimal_settings['img_resize'],
            'divide_num': 1
        },
        'models': {
            'backbone_name': model_name,
            'pretrained': 'dinov3',
            'batch_size': optimal_settings['batch_size'],
            'feature_layers': optimal_settings['feature_layers'],
            'r_list': optimal_settings['r_list']
        },
        'device': 0,
        'testing': {
            'output_dir': f'output_{model_name}',
            'vis': False,
            'vis_type': 'single_norm',
            'save_excel': False
        }
    }
    
    return config


def validate_dinov3_installation():
    """Validate DINOv3 installation and dependencies"""
    try:
        import models.backbone.dinov3_vision_transformer as dinov3_vits
        print("✓ DINOv3 vision transformer imported successfully")
        
        from models.modules._DINOv3_LNAMD import create_dinov3_enhanced_lnamd
        print("✓ Enhanced LNAMD for DINOv3 imported successfully")
        
        # Test model creation
        device = torch.device('cpu')
        model = dinov3_vits.dinov3_vit_small(patch_size=14)
        print("✓ DINOv3 model creation successful")
        
        # Test enhanced LNAMD
        lnamd = create_dinov3_enhanced_lnamd(device=device, model_name='dinov3_vits14')
        print("✓ Enhanced LNAMD creation successful")
        
        print("\n✅ DINOv3 installation validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    # Validate installation
    validate_dinov3_installation()
    
    # Example usage
    manager = DINOv3ModelManager()
    config = setup_dinov3_config('dinov3_vitb14', 'mvtec_ad')
    print("\nExample DINOv3 configuration:")
    print(config)