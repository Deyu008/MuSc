"""
Migration Script: DINOv2 to DINOv3
Helps users upgrade their configurations and workflows from DINOv2 to DINOv3
"""

import os
import yaml
import argparse
from typing import Dict, List


class DINOv2ToDINOv3Migrator:
    """Handles migration from DINOv2 to DINOv3"""
    
    DINOV2_TO_DINOV3_MAPPING = {
        'dinov2_vits14': 'dinov3_vits14',
        'dinov2_vitb14': 'dinov3_vitb14', 
        'dinov2_vitl14': 'dinov3_vitl14',
    }
    
    def __init__(self):
        self.migration_log = []
    
    def migrate_config_file(self, config_path: str, output_path: str = None) -> str:
        """Migrate a configuration file from DINOv2 to DINOv3"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Migrate configuration
        migrated_config = self._migrate_config_dict(config)
        
        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(config_path)[0]
            output_path = f"{base_name}_dinov3.yaml"
        
        # Save migrated configuration
        with open(output_path, 'w') as f:
            yaml.dump(migrated_config, f, default_flow_style=False, indent=2)
        
        self.migration_log.append(f"Migrated {config_path} -> {output_path}")
        return output_path
    
    def _migrate_config_dict(self, config: Dict) -> Dict:
        """Migrate configuration dictionary"""
        migrated = config.copy()
        
        # Migrate backbone name
        if 'models' in migrated and 'backbone_name' in migrated['models']:
            old_backbone = migrated['models']['backbone_name']
            if old_backbone in self.DINOV2_TO_DINOV3_MAPPING:
                new_backbone = self.DINOV2_TO_DINOV3_MAPPING[old_backbone]
                migrated['models']['backbone_name'] = new_backbone
                self.migration_log.append(f"Updated backbone: {old_backbone} -> {new_backbone}")
        
        # Update pretrained field
        if 'models' in migrated:
            migrated['models']['pretrained'] = 'dinov3'
            self.migration_log.append("Updated pretrained to 'dinov3'")
        
        # Optimize batch size for DINOv3
        if 'models' in migrated and 'backbone_name' in migrated['models']:
            backbone = migrated['models']['backbone_name']
            if 'dinov3' in backbone:
                old_batch_size = migrated['models'].get('batch_size', 4)
                new_batch_size = self._get_optimal_batch_size(backbone, old_batch_size)
                migrated['models']['batch_size'] = new_batch_size
                if new_batch_size != old_batch_size:
                    self.migration_log.append(f"Optimized batch size: {old_batch_size} -> {new_batch_size}")
        
        # Update output directory
        if 'testing' in migrated and 'output_dir' in migrated['testing']:
            old_output = migrated['testing']['output_dir']
            if 'dinov2' in old_output:
                migrated['testing']['output_dir'] = old_output.replace('dinov2', 'dinov3')
            elif 'output' == old_output:
                migrated['testing']['output_dir'] = 'output_dinov3'
            self.migration_log.append(f"Updated output directory")
        
        # Optimize feature layers for DINOv3
        if 'models' in migrated:
            backbone = migrated['models'].get('backbone_name', '')
            if 'dinov3' in backbone:
                old_layers = migrated['models'].get('feature_layers', [5, 11, 17, 23])
                new_layers = self._get_optimal_feature_layers(backbone)
                migrated['models']['feature_layers'] = new_layers
                if new_layers != old_layers:
                    self.migration_log.append(f"Optimized feature layers: {old_layers} -> {new_layers}")
        
        return migrated
    
    def _get_optimal_batch_size(self, backbone: str, current_batch_size: int) -> int:
        """Get optimal batch size for DINOv3 model"""
        if 'vits' in backbone:
            return min(current_batch_size, 8)
        elif 'vitb' in backbone:
            return min(current_batch_size, 4)
        elif 'vitl' in backbone:
            return min(current_batch_size, 2)
        elif 'vitg' in backbone:
            return 1
        return current_batch_size
    
    def _get_optimal_feature_layers(self, backbone: str) -> List[int]:
        """Get optimal feature layers for DINOv3 model"""
        if 'vits' in backbone:
            return [3, 6, 9, 11]  # 12 layers total
        elif 'vitb' in backbone:
            return [3, 6, 9, 11]  # 12 layers total
        elif 'vitl' in backbone:
            return [6, 12, 18, 23]  # 24 layers total
        elif 'vitg' in backbone:
            return [10, 20, 30, 39]  # 40 layers total
        return [5, 11, 17, 23]  # default
    
    def generate_migration_report(self) -> str:
        """Generate a migration report"""
        report = "DINOv2 to DINOv3 Migration Report\n"
        report += "=" * 40 + "\n\n"
        
        if not self.migration_log:
            report += "No migrations performed.\n"
        else:
            report += "Migrations performed:\n"
            for i, log_entry in enumerate(self.migration_log, 1):
                report += f"{i}. {log_entry}\n"
        
        report += "\nDINOv3 Benefits:\n"
        report += "- Enhanced multi-head attention with temperature scaling\n"
        report += "- Multi-scale patch embedding for better feature extraction\n"
        report += "- Register tokens for improved attention\n"
        report += "- Learnable positional embeddings\n"
        report += "- Better semantic understanding for anomaly detection\n"
        
        return report
    
    def migrate_multiple_configs(self, config_dir: str, output_dir: str = None) -> List[str]:
        """Migrate multiple configuration files"""
        if output_dir is None:
            output_dir = os.path.join(config_dir, "dinov3_configs")
        
        os.makedirs(output_dir, exist_ok=True)
        
        migrated_files = []
        for filename in os.listdir(config_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                config_path = os.path.join(config_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                try:
                    migrated_path = self.migrate_config_file(config_path, output_path)
                    migrated_files.append(migrated_path)
                except Exception as e:
                    self.migration_log.append(f"Failed to migrate {config_path}: {e}")
        
        return migrated_files


def main():
    parser = argparse.ArgumentParser(description="Migrate DINOv2 configurations to DINOv3")
    parser.add_argument("--config", type=str, help="Path to configuration file to migrate")
    parser.add_argument("--config-dir", type=str, help="Directory containing configuration files to migrate")
    parser.add_argument("--output", type=str, help="Output path for migrated configuration")
    parser.add_argument("--output-dir", type=str, help="Output directory for migrated configurations")
    parser.add_argument("--report", action="store_true", help="Generate migration report")
    
    args = parser.parse_args()
    
    migrator = DINOv2ToDINOv3Migrator()
    
    if args.config:
        # Migrate single configuration file
        try:
            output_path = migrator.migrate_config_file(args.config, args.output)
            print(f"✅ Migration successful: {output_path}")
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return 1
    
    elif args.config_dir:
        # Migrate multiple configuration files
        try:
            migrated_files = migrator.migrate_multiple_configs(args.config_dir, args.output_dir)
            print(f"✅ Migrated {len(migrated_files)} configuration files")
            for file_path in migrated_files:
                print(f"  - {file_path}")
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return 1
    
    else:
        print("Please specify either --config or --config-dir")
        return 1
    
    if args.report:
        report = migrator.generate_migration_report()
        print("\n" + report)
        
        # Save report to file
        report_path = "dinov3_migration_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📄 Migration report saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())