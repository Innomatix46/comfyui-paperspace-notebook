# Universal Model Downloader Guide

The Universal Model Downloader is an enhanced system for managing ComfyUI models with support for multiple formats, smart storage management, and advanced features.

## Features

### 🎯 **Multi-Format Support**
- **SafeTensors**: Safe, memory-mapped format (recommended for production)
- **GGUF**: Quantized format for memory efficiency and CPU inference
- **Legacy Formats**: Binary, PyTorch, Checkpoint (with conversion)

### 💾 **Smart Storage Management**
- **50GB Free Tier Optimization**: Intelligent model selection for Paperspace
- **Space Monitoring**: Real-time storage usage tracking
- **Recommendations**: Smart suggestions based on available space
- **Priority System**: Download essential models first

### ⚡ **Advanced Download Features**
- **Parallel Downloads**: Multiple concurrent downloads with progress tracking
- **Resume Capability**: Continue interrupted downloads
- **Integrity Verification**: SHA256 and file format validation
- **Bandwidth Throttling**: Control download speed
- **Progress Tracking**: Real-time progress with ETA

### 🔄 **Format Conversion**
- **SafeTensors ↔ GGUF**: Convert between formats as needed
- **Quantization Options**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16
- **Batch Conversion**: Process multiple models at once
- **Optimization**: Size vs speed optimization

### 🔗 **Integration**
- **Automatic Symlinks**: Seamless ComfyUI integration
- **Legacy Compatibility**: Works with existing download_models.sh
- **Interactive CLI**: User-friendly command-line interface
- **Search & Filter**: Find models by category, format, use case

## Quick Start

### 1. Setup (One-time)
```bash
# Install dependencies and setup
bash scripts/setup_model_downloader.sh
```

### 2. Interactive Mode (Recommended)
```bash
python3 scripts/universal_model_downloader.py --interactive
```

### 3. Quick Commands
```bash
# Show storage information
python3 scripts/universal_model_downloader.py --storage-info

# Get smart recommendations for your available space
python3 scripts/universal_model_downloader.py --list --recommended

# Search for specific models
python3 scripts/universal_model_downloader.py --search "chroma"

# Download a specific model
python3 scripts/universal_model_downloader.py --download "SDXL Base"
```

## Interactive Mode Guide

The interactive CLI provides a menu-driven interface:

```
🎯 Universal Model Downloader for ComfyUI
==================================================

📋 Available Commands:
1. 🔍 Search models
2. 📝 List all models  
3. 💾 Check storage info
4. 🎯 Get recommendations
5. 📥 Download model
6. 📊 Download queue status
7. 🔗 Create symlinks
8. 📈 Download history
9. ❌ Exit
```

### Command Details

#### 1. Search Models
- Search by name, description, or tags
- Filter by category (checkpoints, loras, etc.)
- Filter by format (safetensors, gguf)
- Shows download status for each model

#### 2. List Models
- Browse all available models
- Filter by category and format
- Show recommended models only
- Displays size, status, and description

#### 3. Storage Information
- Total, used, and free space
- Model storage usage
- Available space for new models
- Low space warnings

#### 4. Smart Recommendations
- Analyzes available space
- Suggests essential models first
- Considers storage priorities
- One-click download for all recommendations

#### 5. Download Manager
- Select from available models
- Space validation before download
- Queue management
- Progress tracking with resume capability

#### 6. Queue Status
- Monitor active downloads
- View pending downloads
- Progress tracking with speed and ETA
- Pause/resume capabilities

#### 7. Symlink Creation
- Automatic ComfyUI integration
- Links storage to ComfyUI/models/
- Handles all model categories
- Safe replacement of existing links

#### 8. Download History
- Track all download attempts
- Success/failure statistics
- Total data downloaded
- Recent activity log

## Model Categories

### Essential Models (Recommended for 50GB Free Tier)

| Category | Model | Format | Size | Use Case |
|----------|-------|--------|------|----------|
| Checkpoints | SDXL Base 1.0 | SafeTensors | 6.9 GB | General purpose |
| VAE | SDXL VAE | SafeTensors | 335 MB | Essential for SDXL |
| Embeddings | EasyNegative V2 | SafeTensors | 25 KB | Quality improvement |
| Upscalers | Real-ESRGAN 4x | SafeTensors | 67 MB | Image enhancement |

**Total: ~7.3 GB** - Leaves plenty of space for outputs and additional models.

### Professional Setup (35-40GB)

Add these for professional workflows:
- **Chroma v48**: Latest high-quality model (24 GB)
- **T5-XXL FP8**: Memory-efficient text encoder (4.9 GB)
- **RealFine LoRA**: Realistic enhancement (294 MB)
- **ControlNet Canny**: Structure guidance (2.5 GB)

### Memory-Optimized Setup (18GB)

For memory-constrained environments:
- **SDXL Base GGUF Q4_0**: Quantized checkpoint (3.5 GB)
- **FLUX Schnell GGUF Q4_0**: Fast generation (12 GB)
- **T5-XXL GGUF Q4_0**: Quantized text encoder (2.5 GB)

## Format Comparison

### SafeTensors (Recommended)
**Pros:**
- Memory-mapped loading
- Safe format (no arbitrary code execution)
- Fast loading on GPU
- Wide compatibility

**Cons:**
- Larger file sizes
- No built-in quantization

**Best for:** Production, GPU inference, maximum compatibility

### GGUF (Memory Efficient)
**Pros:**
- Built-in quantization (2-4x smaller files)
- CPU inference support
- Mobile/edge deployment ready
- Excellent for memory-constrained environments

**Cons:**
- May require specific loaders
- Potential quality loss with aggressive quantization
- Newer format with evolving support

**Best for:** Memory-constrained systems, CPU inference, mobile deployment

## Storage Optimization Strategies

### 50GB Free Tier Strategy
1. **Start Essential**: Download only recommended models (~7GB)
2. **Monitor Usage**: Use `--storage-info` regularly
3. **Clean Outputs**: Regularly clean generated images
4. **Use GGUF**: Consider quantized versions for secondary models
5. **Selective Downloads**: Choose models based on specific needs

### Storage Commands
```bash
# Check current usage
python3 scripts/universal_model_downloader.py --storage-info

# Clean up old downloads
find /storage/ComfyUI -name "*.tmp" -delete
find /storage/ComfyUI -name "*.partial" -delete

# List largest files
du -h /storage/ComfyUI/models/* | sort -hr | head -10
```

## Format Conversion

The included model converter supports format conversion:

```bash
# Convert SafeTensors to GGUF with Q4_0 quantization
python3 scripts/model_converter.py model.safetensors -f gguf -q Q4_0

# Convert GGUF to SafeTensors (requires dequantization)
python3 scripts/model_converter.py model.gguf -f safetensors

# Batch convert directory
python3 scripts/model_converter.py models_dir/ --batch -f gguf -q Q4_0

# Show model information
python3 scripts/model_converter.py model.safetensors --info
```

### Quantization Options
- **F16**: Full 16-bit precision (largest, highest quality)
- **Q8_0**: 8-bit quantization (minimal quality loss)
- **Q5_1**: 5-bit quantization (good balance)
- **Q4_1**: 4-bit quantization (significant size reduction)
- **Q4_0**: 4-bit quantization (smallest, some quality loss)

## Advanced Usage

### Command Line Options
```bash
# Bandwidth limiting
python3 scripts/universal_model_downloader.py --bandwidth-limit 10MB --interactive

# Maximum concurrent downloads
python3 scripts/universal_model_downloader.py --max-concurrent 5 --interactive

# Category-specific operations
python3 scripts/universal_model_downloader.py --list --category checkpoints --format gguf

# Non-interactive download
python3 scripts/universal_model_downloader.py --download "SDXL Base" --format safetensors
```

### Configuration Files

The system uses these configuration files:
- `configs/models_catalog.json`: Complete model catalog
- `configs/download_history.json`: Download tracking
- `configs/chroma_models.json`: Chroma-specific models
- `configs/models.txt`: Legacy format compatibility

### Integration with Existing Scripts

The enhanced downloader integrates seamlessly:
```bash
# Use existing script (will suggest enhanced version)
bash scripts/download_models.sh

# Enhanced version with all features
python3 scripts/universal_model_downloader.py --interactive
```

## Troubleshooting

### Common Issues

#### 1. Download Failures
```bash
# Check internet connection
ping -c 4 huggingface.co

# Resume failed downloads
python3 scripts/universal_model_downloader.py --interactive
# Select "Download queue status" to resume
```

#### 2. Storage Issues
```bash
# Check space
df -h /storage

# Clean temporary files
rm -rf /tmp/model_download_*

# Check model integrity
python3 scripts/model_converter.py model.safetensors --info
```

#### 3. Format Conversion Errors
```bash
# Install conversion dependencies
pip install torch transformers llama-cpp-python

# Check model format
python3 scripts/model_converter.py model.file --info
```

#### 4. Symlink Issues
```bash
# Recreate symlinks
python3 scripts/universal_model_downloader.py --create-symlinks

# Check symlink status
ls -la ComfyUI/models/
```

### Performance Tips

#### 1. Download Optimization
- Use aria2c for faster downloads (installed by setup script)
- Enable parallel downloads (default: 3 concurrent)
- Use bandwidth limiting to avoid overwhelming connection
- Download during off-peak hours for better speeds

#### 2. Storage Optimization
- Monitor storage regularly with `--storage-info`
- Use GGUF format for secondary models to save space
- Clean up old model versions periodically
- Consider external storage for large model collections

#### 3. Format Selection
- **SafeTensors** for GPU inference and maximum compatibility
- **GGUF Q4_0** for memory-constrained environments
- **GGUF F16** for CPU inference without quality loss
- Convert formats as needed rather than storing multiple versions

## Support and Updates

### Getting Help
- Use `--help` flag for command-line help
- Check download history for error details
- Monitor logs in `model_downloader.log`
- Use interactive mode for guided operations

### Updates
The model catalog is regularly updated with new models. To get the latest:
```bash
# Backup your custom catalog
cp configs/models_catalog.json configs/models_catalog.json.backup

# Update will happen automatically when new models are added to the catalog
```

### Contributing
To add new models to the catalog:
1. Follow the ModelInfo structure in the catalog
2. Include all required fields (url, format, size, description)
3. Add appropriate tags and use cases
4. Test download and verification

## Integration with Paperspace

### A6000 Optimization
The system is optimized for Paperspace A6000 instances:
- **48GB VRAM**: Can handle large models like Chroma v48
- **Flash Attention**: Enabled for memory efficiency
- **50GB Storage**: Smart recommendations for free tier
- **Tensorboard Integration**: Monitoring via Tensorboard URL

### Auto-restart Compatibility
The downloader works with Paperspace's 6-hour restart cycle:
- **Resume Downloads**: Automatically resume interrupted downloads
- **Persistent Storage**: Models stored in /storage survive restarts
- **Quick Recovery**: Symlinks recreated automatically on restart

## Best Practices

1. **Start Small**: Begin with essential models, expand as needed
2. **Monitor Storage**: Check space regularly with `--storage-info`
3. **Use Recommendations**: Let the system suggest optimal models
4. **Verify Downloads**: System automatically verifies model integrity
5. **Regular Maintenance**: Clean up old downloads and temporary files
6. **Format Selection**: Choose formats based on your specific use case
7. **Backup Configurations**: Keep backups of custom model catalogs
8. **Test Models**: Verify models work with your workflows after download

The Universal Model Downloader provides a comprehensive solution for managing ComfyUI models with advanced features while maintaining compatibility with existing systems.