#!/bin/bash

# DINOv3 Quick Start Script for MuSc
# This script provides easy access to DINOv3 features

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL="dinov3_vitb14"
DATASET="mvtec_ad"
CATEGORY="bottle"
ACTION="example"
DATA_PATH=""

# Help function
show_help() {
    echo -e "${BLUE}DINOv3 Quick Start Script for MuSc${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL         DINOv3 model to use (default: dinov3_vitb14)"
    echo "                           Available: dinov3_vits14, dinov3_vitb14, dinov3_vitl14, dinov3_vitg14"
    echo "  -d, --dataset DATASET     Dataset to use (default: mvtec_ad)"
    echo "                           Available: mvtec_ad, visa, btad"
    echo "  -c, --category CATEGORY   Category to test (default: bottle)"
    echo "  -p, --data-path PATH      Path to dataset directory"
    echo "  -a, --action ACTION       Action to perform (default: example)"
    echo "                           Available: example, compare, migrate, validate"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run basic example"
    echo "  $0 -m dinov3_vitl14 -c toothbrush   # Use large model on toothbrush category"
    echo "  $0 -a compare                        # Compare DINOv2 vs DINOv3"
    echo "  $0 -a validate                       # Validate installation"
    echo "  $0 -a migrate                        # Show migration example"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -p|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate model choice
case $MODEL in
    dinov3_vits14|dinov3_vitb14|dinov3_vitl14|dinov3_vitg14)
        ;;
    *)
        echo -e "${RED}Invalid model: $MODEL${NC}"
        echo "Available models: dinov3_vits14, dinov3_vitb14, dinov3_vitl14, dinov3_vitg14"
        exit 1
        ;;
esac

# Validate dataset choice
case $DATASET in
    mvtec_ad|visa|btad)
        ;;
    *)
        echo -e "${RED}Invalid dataset: $DATASET${NC}"
        echo "Available datasets: mvtec_ad, visa, btad"
        exit 1
        ;;
esac

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════╗"
echo "║              DINOv3 for MuSc                 ║"
echo "║         Enhanced Anomaly Detection           ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Category: $CATEGORY"
echo "  Action: $ACTION"
if [ -n "$DATA_PATH" ]; then
    echo "  Data Path: $DATA_PATH"
fi
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "models/musc.py" ]; then
    echo -e "${RED}Error: Please run this script from the MuSc project root directory${NC}"
    exit 1
fi

# Execute the appropriate action
case $ACTION in
    example)
        echo -e "${GREEN}Running DINOv3 example...${NC}"
        CMD="python examples/dinov3_example.py --action example --model $MODEL --dataset $DATASET --category $CATEGORY"
        if [ -n "$DATA_PATH" ]; then
            CMD="$CMD --data-path $DATA_PATH"
        fi
        echo "Command: $CMD"
        echo ""
        $CMD
        ;;
    
    compare)
        echo -e "${GREEN}Comparing DINOv2 vs DINOv3...${NC}"
        CMD="python examples/dinov3_example.py --action compare --dataset $DATASET --category $CATEGORY"
        if [ -n "$DATA_PATH" ]; then
            CMD="$CMD --data-path $DATA_PATH"
        fi
        echo "Command: $CMD"
        echo ""
        $CMD
        ;;
    
    migrate)
        echo -e "${GREEN}Running migration example...${NC}"
        python examples/dinov3_example.py --action migrate
        ;;
    
    validate)
        echo -e "${GREEN}Validating DINOv3 installation...${NC}"
        python utils/dinov3_utils.py
        ;;
    
    *)
        echo -e "${RED}Invalid action: $ACTION${NC}"
        echo "Available actions: example, compare, migrate, validate"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Done!${NC}"

# Show additional information based on action
case $ACTION in
    example)
        echo ""
        echo -e "${YELLOW}Next steps:${NC}"
        echo "  • Check results in the output directory"
        echo "  • Try different models: dinov3_vits14, dinov3_vitl14, dinov3_vitg14"
        echo "  • Compare with DINOv2: $0 -a compare"
        ;;
    
    compare)
        echo ""
        echo -e "${YELLOW}Performance comparison completed!${NC}"
        echo "Check the results above to see DINOv3 improvements over DINOv2"
        ;;
    
    migrate)
        echo ""
        echo -e "${YELLOW}Migration example completed!${NC}"
        echo "Use scripts/migrate_to_dinov3.py to migrate your own configurations"
        ;;
    
    validate)
        echo ""
        echo -e "${YELLOW}Validation completed!${NC}"
        echo "If successful, you can now use DINOv3 models"
        ;;
esac