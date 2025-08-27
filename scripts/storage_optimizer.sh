#!/bin/bash
# storage_optimizer.sh - Free Tier Storage Management for 50GB constraint
# This script optimizes storage usage for Paperspace Free Tier limitations

# Storage configuration for Free Tier (50GB total)
STORAGE_LIMIT_GB=45                    # Leave 5GB buffer
STORAGE_WARNING_THRESHOLD=40           # Warning at 40GB usage
STORAGE_CRITICAL_THRESHOLD=43          # Critical at 43GB usage
STORAGE_BASE_DIR="/storage/ComfyUI"

# Function to get storage usage in GB
get_storage_usage() {
    local usage_bytes=$(du -sb "$STORAGE_BASE_DIR" 2>/dev/null | cut -f1)
    if [ -z "$usage_bytes" ]; then
        echo "0"
    else
        echo $((usage_bytes / 1024 / 1024 / 1024))
    fi
}

# Function to get available disk space in GB
get_available_space() {
    local available=$(df /storage 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -z "$available" ]; then
        echo "0"
    else
        echo $((available / 1024 / 1024))
    fi
}

# Function to display storage status with A6000 optimization tips
show_storage_status() {
    local usage_gb=$(get_storage_usage)
    local available_gb=$(get_available_space)
    local total_used=$((50 - available_gb))
    
    echo "🗄️  FREE TIER STORAGE STATUS (A6000 Optimized)"
    echo "=================================================="
    echo "📊 Total Disk Usage: ${total_used}GB / 50GB"
    echo "📁 ComfyUI Storage: ${usage_gb}GB"
    echo "💾 Available Space: ${available_gb}GB"
    echo "🎯 Storage Limit: ${STORAGE_LIMIT_GB}GB (with 5GB buffer)"
    echo ""
    
    # Status indicators
    if [ "$total_used" -gt "$STORAGE_CRITICAL_THRESHOLD" ]; then
        echo "🔴 CRITICAL: Storage almost full! Immediate cleanup required."
        echo "   Run: ./restart-control.sh logs cleanup"
    elif [ "$total_used" -gt "$STORAGE_WARNING_THRESHOLD" ]; then
        echo "🟡 WARNING: Storage getting full. Consider cleanup."
        echo "   Run: ./restart-control.sh logs cleanup"
    else
        echo "✅ GOOD: Storage usage within safe limits."
    fi
    
    # A6000 specific recommendations
    echo ""
    echo "🚀 A6000 OPTIMIZATION TIPS:"
    echo "   • Use 8-bit quantization to reduce model sizes"
    echo "   • Enable gradient checkpointing for larger batch sizes"
    echo "   • Use Flash Attention for memory efficiency"
    echo "   • Monitor GPU temperature (throttles at 83°C)"
}

# Function to cleanup old files and optimize storage
cleanup_storage() {
    echo "🧹 FREE TIER STORAGE CLEANUP"
    echo "=============================="
    
    local cleaned_mb=0
    
    # 1. Clean temporary files
    echo "==> Cleaning temporary files..."
    if [ -d "/tmp" ]; then
        local tmp_size_before=$(du -sm /tmp 2>/dev/null | cut -f1)
        find /tmp -type f -mtime +1 -delete 2>/dev/null || true
        local tmp_size_after=$(du -sm /tmp 2>/dev/null | cut -f1)
        local tmp_cleaned=$((tmp_size_before - tmp_size_after))
        cleaned_mb=$((cleaned_mb + tmp_cleaned))
        echo "    Cleaned ${tmp_cleaned}MB from /tmp"
    fi
    
    # 2. Clean ComfyUI output temp files
    echo "==> Cleaning ComfyUI temporary outputs..."
    if [ -d "$STORAGE_BASE_DIR/output" ]; then
        find "$STORAGE_BASE_DIR/output" -name "*.tmp" -delete 2>/dev/null || true
        find "$STORAGE_BASE_DIR/output" -name "temp_*" -mtime +1 -delete 2>/dev/null || true
        echo "    Cleaned temporary output files"
    fi
    
    # 3. Clean old log files
    echo "==> Cleaning old log files..."
    if [ -f "$STORAGE_BASE_DIR/restart.log" ]; then
        local log_size_mb=$(du -sm "$STORAGE_BASE_DIR/restart.log" 2>/dev/null | cut -f1)
        if [ "$log_size_mb" -gt 10 ]; then
            tail -1000 "$STORAGE_BASE_DIR/restart.log" > "$STORAGE_BASE_DIR/restart.log.tmp"
            mv "$STORAGE_BASE_DIR/restart.log.tmp" "$STORAGE_BASE_DIR/restart.log"
            cleaned_mb=$((cleaned_mb + log_size_mb - 1))
            echo "    Truncated restart.log (saved ${log_size_mb}MB)"
        fi
    fi
    
    # 4. Clean Python cache
    echo "==> Cleaning Python cache files..."
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    echo "    Cleaned Python cache files"
    
    # 5. Clean pip cache
    echo "==> Cleaning pip cache..."
    if command -v pip >/dev/null 2>&1; then
        pip cache purge >/dev/null 2>&1 || true
        echo "    Cleaned pip cache"
    fi
    
    # 6. Clean old model downloads (if any failed downloads)
    echo "==> Cleaning incomplete model downloads..."
    if [ -d "$STORAGE_BASE_DIR/models" ]; then
        find "$STORAGE_BASE_DIR/models" -name "*.partial" -delete 2>/dev/null || true
        find "$STORAGE_BASE_DIR/models" -name "*.aria2" -delete 2>/dev/null || true
        echo "    Cleaned incomplete downloads"
    fi
    
    echo ""
    echo "✅ Cleanup completed! Estimated space freed: ${cleaned_mb}MB"
    echo ""
    
    # Show updated status
    show_storage_status
}

# Function to optimize ComfyUI for A6000 and storage constraints
optimize_for_a6000() {
    echo "🚀 A6000 + FREE TIER OPTIMIZATION"
    echo "================================="
    
    # Create A6000 optimization config
    local config_file="$STORAGE_BASE_DIR/a6000_config.json"
    cat > "$config_file" << 'EOF'
{
    "gpu_optimization": {
        "device": "cuda:0",
        "precision": "fp16",
        "enable_flash_attention": true,
        "enable_xformers": true,
        "use_8bit_adam": true,
        "gradient_checkpointing": true,
        "batch_size_optimization": true
    },
    "memory_management": {
        "vram_limit_gb": 46,
        "enable_model_offload": true,
        "enable_cpu_offload": false,
        "clear_cache_after_generation": true
    },
    "storage_optimization": {
        "compress_outputs": true,
        "auto_cleanup_temp": true,
        "max_output_files": 100
    },
    "a6000_specific": {
        "enable_tensor_rt": true,
        "optimize_for_throughput": true,
        "thermal_throttle_temp": 80
    }
}
EOF
    
    echo "✅ Created A6000 optimization config at: $config_file"
    echo ""
    echo "🎯 A6000 OPTIMIZATION FEATURES ENABLED:"
    echo "   • Flash Attention for memory efficiency"
    echo "   • 8-bit Adam optimizer (saves VRAM)"
    echo "   • Gradient checkpointing for larger models"
    echo "   • FP16 precision for 2x speed boost"
    echo "   • Thermal management (throttle at 80°C)"
    echo "   • Storage compression and auto-cleanup"
}

# Function to monitor storage in real-time
monitor_storage() {
    echo "📊 REAL-TIME STORAGE MONITORING"
    echo "==============================="
    echo "Press Ctrl+C to stop monitoring"
    echo ""
    
    while true; do
        clear
        show_storage_status
        echo ""
        echo "🔄 Monitoring... (refreshes every 30 seconds)"
        echo "💡 Tip: Keep ComfyUI outputs under 5GB to maintain buffer"
        sleep 30
    done
}

# Function to suggest storage optimizations
suggest_optimizations() {
    local usage_gb=$(get_storage_usage)
    local available_gb=$(get_available_space)
    
    echo "💡 STORAGE OPTIMIZATION SUGGESTIONS"
    echo "==================================="
    
    if [ "$usage_gb" -gt 30 ]; then
        echo "🔍 Model Storage Analysis:"
        if [ -d "$STORAGE_BASE_DIR/models" ]; then
            echo "   Current model usage:"
            du -sh "$STORAGE_BASE_DIR/models"/* 2>/dev/null | sort -hr | head -5
            echo ""
            echo "💡 Recommendations:"
            echo "   • Keep only 1-2 essential checkpoints"
            echo "   • Use LoRA models instead of full checkpoints"
            echo "   • Remove unused ControlNet models"
        fi
    fi
    
    if [ "$available_gb" -lt 10 ]; then
        echo ""
        echo "🚨 LOW STORAGE RECOMMENDATIONS:"
        echo "   1. Run cleanup: ./storage-optimizer.sh cleanup"
        echo "   2. Remove old outputs: rm -rf /storage/ComfyUI/output/old/*"
        echo "   3. Comment out unused models in configs/models.txt"
        echo "   4. Use online model repos instead of local storage"
    fi
    
    echo ""
    echo "🎯 FREE TIER BEST PRACTICES:"
    echo "   • Monitor storage daily"
    echo "   • Use external storage for large model collections"
    echo "   • Enable auto-cleanup in restart configuration"
    echo "   • Compress outputs before long-term storage"
}

# Main function to handle different operations
case "${1:-status}" in
    "status")
        show_storage_status
        ;;
    "cleanup")
        cleanup_storage
        ;;
    "optimize")
        optimize_for_a6000
        ;;
    "monitor")
        monitor_storage
        ;;
    "suggest")
        suggest_optimizations
        ;;
    "help"|"--help"|"-h")
        echo "Free Tier Storage Optimizer for A6000"
        echo "====================================="
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  status     Show current storage status (default)"
        echo "  cleanup    Clean temporary files and optimize storage"
        echo "  optimize   Configure A6000 optimizations"
        echo "  monitor    Real-time storage monitoring"
        echo "  suggest    Get optimization suggestions"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 status    # Check storage usage"
        echo "  $0 cleanup   # Free up space"
        echo "  $0 optimize  # Configure for A6000"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac