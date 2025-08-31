#!/bin/bash
# download_models.sh - Download and link ComfyUI models
# This script handles model downloads and creates proper symlinks for ComfyUI
# Enhanced with universal model downloader integration

download_models() {
    echo "==> Starting model download and linking process..."
    
    # Define base storage directory
    STORAGE_BASE_DIR="/storage/ComfyUI/models"
    INIT_FLAG_FILE="/storage/ComfyUI/.init_done"
    
    # Create storage directory structure
    echo "==> Creating storage directory structure..."
    mkdir -p "$STORAGE_BASE_DIR"
    
    # Check for initialization flag
    if [ -f "$INIT_FLAG_FILE" ]; then
        echo "==> Models already downloaded (init flag found), skipping download process"
    else
        echo "==> No init flag found, proceeding with model downloads..."
        
        # Check if models.txt exists
        if [ ! -f "configs/models.txt" ]; then
            echo "==> Warning: configs/models.txt not found, skipping model downloads"
        else
            echo "==> Processing model downloads from configs/models.txt..."
            
            # Check for download tools
            if command -v aria2c >/dev/null 2>&1; then
                DOWNLOAD_CMD="aria2c -x 8 -s 8 -c"
                echo "==> Using aria2c for accelerated downloads"
            elif command -v wget >/dev/null 2>&1; then
                DOWNLOAD_CMD="wget -N -c"
                echo "==> Using wget for downloads (aria2c not found)"
            else
                echo "==> Error: Neither aria2c nor wget found, cannot download models"
                return 1
            fi
            
            # Process each line in models.txt
            while IFS= read -r line || [ -n "$line" ]; do
                # Skip empty lines and comments
                if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
                    continue
                fi
                
                # Parse line into target subdirectory and URL
                read -r target_subdir download_url <<< "$line"
                
                if [ -z "$target_subdir" ] || [ -z "$download_url" ]; then
                    echo "==> Warning: Invalid line format, skipping: $line"
                    continue
                fi
                
                # Create destination directory
                dest_dir="$STORAGE_BASE_DIR/$target_subdir"
                mkdir -p "$dest_dir"
                
                # Extract filename from URL
                filename=$(basename "$download_url")
                dest_file="$dest_dir/$filename"
                
                echo "==> Downloading $filename to $target_subdir..."
                
                # Change to destination directory and download
                (
                    cd "$dest_dir"
                    if [ "$DOWNLOAD_CMD" = "aria2c -x 8 -s 8 -c" ]; then
                        $DOWNLOAD_CMD "$download_url"
                    else
                        $DOWNLOAD_CMD "$download_url"
                    fi
                )
                
                if [ $? -eq 0 ]; then
                    echo "==> Successfully downloaded $filename"
                else
                    echo "==> Warning: Failed to download $filename"
                fi
                
            done < configs/models.txt
            
            # Create initialization flag after successful completion
            echo "==> Creating initialization flag..."
            touch "$INIT_FLAG_FILE"
            echo "==> Model downloads completed successfully"
        fi
    fi
    
    # Create symlinks regardless of download status
    echo "==> Creating symlinks from storage to ComfyUI models directory..."
    
    # Ensure ComfyUI models directory exists
    mkdir -p ComfyUI/models
    
    # Create symlinks for each subdirectory in storage
    if [ -d "$STORAGE_BASE_DIR" ]; then
        for storage_subdir in "$STORAGE_BASE_DIR"/*; do
            if [ -d "$storage_subdir" ]; then
                subdir_name=$(basename "$storage_subdir")
                link_target="ComfyUI/models/$subdir_name"
                
                # Remove existing symlink or directory if it exists
                if [ -L "$link_target" ] || [ -e "$link_target" ]; then
                    echo "==> Removing existing $link_target..."
                    rm -rf "$link_target"
                fi
                
                # Create symlink
                echo "==> Creating symlink: $link_target -> $storage_subdir"
                ln -s "$storage_subdir" "$link_target"
            fi
        done
    fi
    
    echo "==> Model linking completed successfully"
    
    # Suggest using the new universal downloader
    if [ -f "scripts/universal_model_downloader.py" ]; then
        echo ""
        echo "💡 TIP: Use the enhanced Universal Model Downloader for more features:"
        echo "   python3 scripts/universal_model_downloader.py --interactive"
        echo "   Features: GGUF support, smart recommendations, progress tracking, resume capability"
    fi
}