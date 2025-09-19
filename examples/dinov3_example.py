#!/usr/bin/env python3
"""
DINOv3 Example Script for MuSc
Demonstrates usage of DINOv3 backbone with enhanced features
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from models.musc import MuSc
from utils.dinov3_utils import DINOv3ModelManager, setup_dinov3_config, validate_dinov3_installation
from scripts.migrate_to_dinov3 import DINOv2ToDINOv3Migrator


def run_dinov3_example(model_name='dinov3_vitb14', dataset='mvtec_ad', data_path=None, category='bottle'):
    """Run DINOv3 example with specified parameters"""
    
    print(f"🚀 Running DINOv3 Example")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Category: {category}")
    print("-" * 50)
    
    # Validate installation
    print("1. Validating DINOv3 installation...")
    if not validate_dinov3_installation():
        print("❌ Installation validation failed. Please check dependencies.")
        return False
    
    # Setup configuration
    print("2. Setting up configuration...")
    config = setup_dinov3_config(model_name, dataset)
    
    if data_path:
        config['datasets']['data_path'] = data_path
    
    # Set specific category if provided
    if category != 'ALL':
        config['datasets']['class_name'] = category
    
    print(f"Configuration:")
    print(f"  - Backbone: {config['models']['backbone_name']}")
    print(f"  - Image size: {config['datasets']['img_resize']}")
    print(f"  - Batch size: {config['models']['batch_size']}")
    print(f"  - Feature layers: {config['models']['feature_layers']}")
    print(f"  - Output dir: {config['testing']['output_dir']}")
    
    # Create and run model
    print("3. Initializing MuSc with DINOv3...")
    try:
        musc_model = MuSc(config, seed=42)
        
        print("4. Running anomaly detection...")
        musc_model.main()
        
        print("✅ DINOv3 example completed successfully!")
        print(f"Results saved to: {config['testing']['output_dir']}")
        return True
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return False


def compare_dinov2_vs_dinov3(dataset='mvtec_ad', data_path=None, category='bottle'):
    """Compare performance between DINOv2 and DINOv3"""
    
    print("🔄 Comparing DINOv2 vs DINOv3 Performance")
    print("-" * 50)
    
    models_to_compare = [
        ('dinov2_vitb14', 'DINOv2 Base'),
        ('dinov3_vitb14', 'DINOv3 Base')
    ]
    
    results = {}
    
    for model_name, model_desc in models_to_compare:
        print(f"\n🔍 Testing {model_desc} ({model_name})")
        
        # Setup configuration
        if 'dinov3' in model_name:
            config = setup_dinov3_config(model_name, dataset)
        else:
            # Basic DINOv2 config
            config = {
                'datasets': {
                    'dataset_name': dataset,
                    'data_path': data_path or f'./data/{dataset}/',
                    'class_name': category,
                    'img_resize': 518,
                    'divide_num': 1
                },
                'models': {
                    'backbone_name': model_name,
                    'pretrained': 'openai' if 'dinov2' not in model_name else 'dinov2',
                    'batch_size': 4,
                    'feature_layers': [5, 11, 17, 23],
                    'r_list': [1, 3, 5]
                },
                'device': 0,
                'testing': {
                    'output_dir': f'output_{model_name}_comparison',
                    'vis': False,
                    'vis_type': 'single_norm',
                    'save_excel': False
                }
            }
        
        try:
            musc_model = MuSc(config, seed=42)
            image_metrics, pixel_metrics = musc_model.make_category_data(category)
            
            results[model_name] = {
                'image_auroc': image_metrics[0],
                'image_f1': image_metrics[1],
                'image_ap': image_metrics[2],
                'pixel_auroc': pixel_metrics[0],
                'pixel_f1': pixel_metrics[1],
                'pixel_ap': pixel_metrics[2],
                'pixel_aupro': pixel_metrics[3]
            }
            
            print(f"✅ {model_desc} completed")
            
        except Exception as e:
            print(f"❌ {model_desc} failed: {e}")
            results[model_name] = None
    
    # Print comparison results
    print("\n📊 Performance Comparison Results")
    print("=" * 60)
    
    if results['dinov2_vitb14'] and results['dinov3_vitb14']:
        dinov2_results = results['dinov2_vitb14']
        dinov3_results = results['dinov3_vitb14']
        
        print(f"{'Metric':<20} {'DINOv2':<12} {'DINOv3':<12} {'Improvement':<12}")
        print("-" * 60)
        
        metrics = [
            ('Image AUROC', 'image_auroc'),
            ('Image F1', 'image_f1'),
            ('Image AP', 'image_ap'),
            ('Pixel AUROC', 'pixel_auroc'),
            ('Pixel F1', 'pixel_f1'),
            ('Pixel AP', 'pixel_ap'),
            ('Pixel AUPRO', 'pixel_aupro')
        ]
        
        for metric_name, metric_key in metrics:
            dinov2_val = dinov2_results[metric_key] * 100
            dinov3_val = dinov3_results[metric_key] * 100
            improvement = dinov3_val - dinov2_val
            
            print(f"{metric_name:<20} {dinov2_val:<12.2f} {dinov3_val:<12.2f} {improvement:+.2f}")
    
    return results


def migrate_config_example():
    """Demonstrate configuration migration from DINOv2 to DINOv3"""
    
    print("🔄 Configuration Migration Example")
    print("-" * 40)
    
    # Create example DINOv2 config
    dinov2_config = {
        'datasets': {
            'dataset_name': 'mvtec_ad',
            'data_path': './data/mvtec_anomaly_detection/',
            'class_name': 'ALL',
            'img_resize': 518,
            'divide_num': 1
        },
        'models': {
            'backbone_name': 'dinov2_vitb14',
            'pretrained': 'openai',
            'batch_size': 8,
            'feature_layers': [5, 11, 17, 23],
            'r_list': [1, 3, 5]
        },
        'device': 0,
        'testing': {
            'output_dir': 'output_dinov2',
            'vis': False,
            'vis_type': 'single_norm',
            'save_excel': False
        }
    }
    
    # Save example config
    example_config_path = '/tmp/example_dinov2_config.yaml'
    with open(example_config_path, 'w') as f:
        yaml.dump(dinov2_config, f, default_flow_style=False)
    
    print(f"📝 Created example DINOv2 config: {example_config_path}")
    
    # Migrate configuration
    migrator = DINOv2ToDINOv3Migrator()
    migrated_path = migrator.migrate_config_file(example_config_path, '/tmp/example_dinov3_config.yaml')
    
    print(f"📝 Migrated to DINOv3 config: {migrated_path}")
    
    # Show migration report
    report = migrator.generate_migration_report()
    print("\n📋 Migration Report:")
    print(report)
    
    # Clean up
    os.remove(example_config_path)
    os.remove(migrated_path)


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Example Script for MuSc")
    parser.add_argument("--action", choices=['example', 'compare', 'migrate'], 
                       default='example', help="Action to perform")
    parser.add_argument("--model", default='dinov3_vitb14', 
                       choices=['dinov3_vits14', 'dinov3_vitb14', 'dinov3_vitl14', 'dinov3_vitg14'],
                       help="DINOv3 model to use")
    parser.add_argument("--dataset", default='mvtec_ad', 
                       choices=['mvtec_ad', 'visa', 'btad'],
                       help="Dataset to use")
    parser.add_argument("--data-path", type=str, help="Path to dataset")
    parser.add_argument("--category", default='bottle', help="Category to test (or 'ALL')")
    
    args = parser.parse_args()
    
    if args.action == 'example':
        print("🎯 Running DINOv3 Example")
        success = run_dinov3_example(args.model, args.dataset, args.data_path, args.category)
        return 0 if success else 1
        
    elif args.action == 'compare':
        print("⚖️ Running DINOv2 vs DINOv3 Comparison")
        compare_dinov2_vs_dinov3(args.dataset, args.data_path, args.category)
        return 0
        
    elif args.action == 'migrate':
        print("🔄 Running Migration Example")
        migrate_config_example()
        return 0


if __name__ == "__main__":
    exit(main())