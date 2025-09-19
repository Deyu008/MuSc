# DINOv3 Integration for MuSc

This document describes the DINOv3 integration in the MuSc project, providing enhanced anomaly detection capabilities through improved vision transformer architectures.

## Overview

DINOv3 represents a significant advancement over DINOv2, offering several key improvements:

- **Enhanced Multi-Head Attention**: Temperature scaling for better attention distribution
- **Multi-Scale Patch Embedding**: Multiple convolutional scales for better feature extraction
- **Register Tokens**: Additional learnable tokens for improved attention and reduced artifacts
- **Learnable Positional Embeddings**: Adaptive to different input sizes
- **Improved Semantic Understanding**: Better representations for anomaly detection tasks

## Available Models

| Model | Parameters | Embedding Dim | Layers | Heads | Recommended Use |
|-------|------------|---------------|--------|-------|-----------------|
| `dinov3_vits14` | 22M | 384 | 12 | 6 | Fast inference, limited resources |
| `dinov3_vitb14` | 86M | 768 | 12 | 12 | **Recommended** - Best balance |
| `dinov3_vitl14` | 304M | 1024 | 24 | 16 | High accuracy, more resources |
| `dinov3_vitg14` | 1.1B | 1536 | 40 | 24 | Maximum performance, high-end GPUs |

## Quick Start

### 1. Basic Usage

```bash
# Run with DINOv3 Base model
python examples/musc_main.py --config configs/musc_dinov3.yaml

# Or specify model directly
python examples/musc_main.py \
    --config configs/musc.yaml \
    --backbone_name dinov3_vitb14 \
    --output_dir output_dinov3
```

### 2. Configuration

Create a DINOv3 configuration file:

```yaml
datasets:
  dataset_name: 'mvtec_ad'
  data_path: ./data/mvtec_anomaly_detection/
  class_name: 'ALL'
  img_resize: 518
  divide_num: 1

models:
  backbone_name: 'dinov3_vitb14'  # Choose your DINOv3 model
  pretrained: 'dinov3'
  batch_size: 4
  feature_layers: [5, 11, 17, 23]  # Optimized for DINOv3
  r_list: [1, 3, 5]

device: 0
testing:
  output_dir: 'output_dinov3'
  vis: False
  vis_type: single_norm
  save_excel: False
```

### 3. Migration from DINOv2

Use the migration script to upgrade existing configurations:

```bash
# Migrate single configuration
python scripts/migrate_to_dinov3.py --config configs/musc.yaml --output configs/musc_dinov3.yaml

# Migrate all configurations in a directory
python scripts/migrate_to_dinov3.py --config-dir configs/ --output-dir configs/dinov3/ --report
```

## Advanced Features

### Enhanced LNAMD

DINOv3 models automatically use the enhanced LNAMD module, which includes:

- **Attention-based feature weighting**: Better context understanding
- **Multi-scale aggregation**: Features at different granularities
- **Temperature-scaled similarity**: Improved similarity computation

### Multi-Scale Feature Extraction

DINOv3's multi-scale patch embedding provides better feature extraction:

```python
# The system automatically uses enhanced features for DINOv3 models
# No additional configuration required
```

### Optimal Settings by Model

The system automatically adjusts settings based on the chosen model:

```python
from utils.dinov3_utils import DINOv3ModelManager

manager = DINOv3ModelManager()
optimal_settings = manager.get_optimal_settings('dinov3_vitb14')
print(optimal_settings)
# Output: {'img_resize': 518, 'batch_size': 4, 'feature_layers': [5, 11, 17, 23], 'r_list': [1, 3, 5]}
```

## Pretrained Weights

### Automatic Download

The system can automatically download pretrained weights:

```python
from utils.dinov3_utils import DINOv3ModelManager

manager = DINOv3ModelManager()
model = manager.load_pretrained_model('dinov3_vitb14')
```

### Manual Setup

If automatic download fails, download weights manually:

1. Visit Meta's official DINOv3 repository
2. Download the desired model weights
3. Place them in `./pretrained_models/` directory
4. Use the naming convention: `{model_name}_pretrain.pth`

## Performance Recommendations

### Memory Usage

| Model | GPU Memory (Batch=1) | GPU Memory (Batch=4) | Recommended GPU |
|-------|---------------------|---------------------|-----------------|
| DINOv3-S | ~2GB | ~6GB | GTX 1060+ |
| DINOv3-B | ~4GB | ~12GB | RTX 3080+ |
| DINOv3-L | ~8GB | ~24GB | RTX 4090+ |
| DINOv3-G | ~16GB | ~48GB | A100+ |

### Batch Size Guidelines

```python
# Recommended batch sizes by model
batch_sizes = {
    'dinov3_vits14': 8,   # Can handle larger batches
    'dinov3_vitb14': 4,   # Balanced performance
    'dinov3_vitl14': 2,   # Reduce for memory
    'dinov3_vitg14': 1,   # Single image processing
}
```

### Feature Layer Selection

```python
# Optimized feature layers by model depth
feature_layers = {
    'dinov3_vits14': [3, 6, 9, 11],    # 12 layers total
    'dinov3_vitb14': [3, 6, 9, 11],    # 12 layers total  
    'dinov3_vitl14': [6, 12, 18, 23],  # 24 layers total
    'dinov3_vitg14': [10, 20, 30, 39], # 40 layers total
}
```

## Validation and Testing

### Installation Validation

```bash
python utils/dinov3_utils.py
```

This will validate:
- DINOv3 module imports
- Model creation
- Enhanced LNAMD functionality

### Performance Comparison

To compare DINOv2 vs DINOv3 performance:

```bash
# Run with DINOv2
python examples/musc_main.py --backbone_name dinov2_vitb14 --output_dir output_dinov2

# Run with DINOv3
python examples/musc_main.py --backbone_name dinov3_vitb14 --output_dir output_dinov3
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model variant
   - Reduce image resolution

2. **Slow Performance**
   - Ensure CUDA is available
   - Use appropriate batch size
   - Consider model size vs. hardware

3. **Import Errors**
   - Check Python path includes project root
   - Verify all dependencies are installed
   - Run validation script

### Debug Mode

Enable verbose logging:

```python
import logging
logging.getLogger("dinov3").setLevel(logging.DEBUG)
```

## Citation

If you use DINOv3 in your research, please cite:

```bibtex
@article{oquab2023dinov3,
  title={DINOv3: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Théo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## Support

For issues specific to DINOv3 integration:
1. Check this README
2. Run the validation script
3. Review migration logs
4. Check the original MuSc documentation