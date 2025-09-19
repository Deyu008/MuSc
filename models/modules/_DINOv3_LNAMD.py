"""
Enhanced LNAMD module for DINOv3
Leverages DINOv3's improved features for better anomaly detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import math

from models.modules._LNAMD import LNAMD, PatchMaker, Preprocessing, MeanMapper


class DINOv3EnhancedLNAMD(LNAMD):
    """Enhanced LNAMD that leverages DINOv3's specific features"""
    
    def __init__(self, device, r=2, feature_dim=768, feature_layer=[0, 1, 2, 3], use_attention_weights=True):
        super().__init__(device, r, feature_dim, feature_layer)
        self.use_attention_weights = use_attention_weights
        
        # Enhanced multi-scale aggregation
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=7, padding=3),
        ])
        
        # Attention-based feature weighting
        if use_attention_weights:
            self.attention_weights = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True,
                dropout=0.1
            )
            
        # Temperature-scaled similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        self.to(device)
    
    def _enhanced_embed(self, features):
        """Enhanced embedding with DINOv3-specific features"""
        if not self.use_attention_weights:
            return super()._embed(features)
        
        features_layers = []
        for l in range(len(features)):
            feature = features[l]  # (B, L, C)
            B, L, C = feature.shape
            
            # Apply multi-scale convolutions for enhanced feature extraction
            feature_transposed = feature.transpose(1, 2)  # (B, C, L)
            multi_scale_features = []
            
            for conv in self.multi_scale_conv:
                ms_feat = conv(feature_transposed)
                multi_scale_features.append(ms_feat)
            
            # Combine multi-scale features
            enhanced_feature = torch.cat(multi_scale_features, dim=1)  # (B, 3*C, L)
            enhanced_feature = F.adaptive_avg_pool1d(enhanced_feature, C).transpose(1, 2)  # (B, L, C)
            
            # Apply self-attention for better context understanding
            attended_feature, _ = self.attention_weights(enhanced_feature, enhanced_feature, enhanced_feature)
            
            # Residual connection
            final_feature = feature + attended_feature
            
            features_layers.append(final_feature)
        
        # Apply original embedding logic with enhanced features
        return self._original_embed(features_layers)
    
    def _original_embed(self, features):
        """Original embedding logic from parent class"""
        # This replicates the original _embed method but uses enhanced features
        features_agg = []
        
        for l in range(len(features)):
            feature = features[l]
            B, L, C = feature.shape
            # Apply aggregation as in original implementation
            feature_reshaped = feature.view(B*L, C)
            
            # Enhanced similarity computation with temperature scaling
            if hasattr(self, 'temperature'):
                feature_reshaped = feature_reshaped / self.temperature
            
            features_agg.append(feature_reshaped.view(B, L, C))
        
        return torch.stack(features_agg, dim=2)  # (B, L, n_layers, C)

    def compute_enhanced_similarity(self, query_features, support_features):
        """Enhanced similarity computation leveraging DINOv3 features"""
        # Temperature-scaled cosine similarity
        query_norm = F.normalize(query_features, dim=-1)
        support_norm = F.normalize(support_features, dim=-1)
        
        similarity = torch.matmul(query_norm, support_norm.transpose(-2, -1))
        
        # Apply temperature scaling
        if hasattr(self, 'temperature'):
            similarity = similarity / self.temperature
        
        return similarity


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor optimized for DINOv3"""
    
    def __init__(self, feature_dim=768, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.feature_dim = feature_dim
        
        # Multi-scale pooling layers
        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((scale, scale)) for scale in scales
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(
            len(scales) * feature_dim, 
            feature_dim, 
            kernel_size=1
        )
        
    def forward(self, feature_maps):
        """
        Args:
            feature_maps: List of feature maps from different layers
        Returns:
            Enhanced multi-scale features
        """
        multi_scale_features = []
        
        for feature_map in feature_maps:
            B, L, C = feature_map.shape
            H = W = int(math.sqrt(L))
            
            # Reshape to spatial format
            spatial_feature = feature_map.view(B, H, W, C).permute(0, 3, 1, 2)
            
            # Apply multi-scale pooling
            scale_features = []
            for pooling in self.pooling_layers:
                pooled = pooling(spatial_feature)
                # Resize back to original spatial size
                resized = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
                scale_features.append(resized)
            
            # Concatenate and fuse
            concatenated = torch.cat(scale_features, dim=1)
            fused = self.fusion_conv(concatenated)
            
            # Reshape back to sequence format
            enhanced_feature = fused.permute(0, 2, 3, 1).view(B, L, C)
            multi_scale_features.append(enhanced_feature)
        
        return multi_scale_features


def create_dinov3_enhanced_lnamd(device, r=2, feature_dim=768, feature_layer=[0, 1, 2, 3], model_name='dinov3_vitb14'):
    """Create DINOv3-optimized LNAMD based on model variant"""
    
    # Adjust parameters based on DINOv3 model variant
    if 'vits' in model_name:
        feature_dim = 384
    elif 'vitb' in model_name:
        feature_dim = 768
    elif 'vitl' in model_name:
        feature_dim = 1024
    elif 'vitg' in model_name:
        feature_dim = 1536
    
    return DINOv3EnhancedLNAMD(
        device=device,
        r=r,
        feature_dim=feature_dim,
        feature_layer=feature_layer,
        use_attention_weights=True
    )