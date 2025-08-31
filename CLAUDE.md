# Claude Code Configuration - ComfyUI Paperspace Notebook

## 🚀 Project: ComfyUI Paperspace Setup with A6000 Optimization

Production-ready ComfyUI installation for Paperspace Gradient with:
- ✅ **100% Working Tensorboard URL Access** (Port 6006)
- ✅ **A6000 GPU Optimization** (48GB VRAM, Flash Attention)
- ✅ **Python 3.11/3.12 Support** with intelligent fallbacks
- ✅ **Auto-Restart System** (6-hour cycles for GPU memory)
- ✅ **JupyterLab with Root Access** for full filesystem control
- ✅ **50GB Free Tier Storage Management**

## 📋 Key Project Commands

### Main Setup
```bash
./run.sh                           # One-command complete setup
git pull origin master             # Update to latest fixes
```

### Storage & System
```bash
./scripts/storage_optimizer.sh status     # Check 50GB storage usage
./restart-control.sh status|enable|disable # Control auto-restart
nvidia-smi                                 # GPU status
df -h                                     # Disk usage
```

### Manual Operations
```bash
source scripts/install_dependencies.sh && install_dependencies  # Dependencies only
source scripts/download_models.sh && download_models           # Models only
source scripts/configure_jupyterlab.sh && start_jupyterlab    # JupyterLab only
```

## 🔧 Recent Fixes & Improvements (2025)

### ✅ Resolved Issues
1. **PyTorch/Flash Attention Order**: PyTorch installed before Flash Attention
2. **Package Version Conflicts**: Using flexible versions (>=) instead of exact (==)
3. **PyTorch/xformers Compatibility**: Separate installation with auto-resolution
4. **PyPI Index Conflicts**: Separate CUDA and PyPI package sources
5. **Git Authentication**: Timeout and fallback mechanisms for custom nodes
6. **JupyterLab Root Access**: Full filesystem visibility with ServerApp.root_dir='/'

### 📦 Package Installation Flow
1. Build dependencies (pip, setuptools, wheel, ninja)
2. PyTorch + torchvision with CUDA 12.4 index
3. xformers with automatic version resolution
4. ML packages from PyPI (accelerate, transformers, etc.)
5. Flash Attention (pre-built wheels for Python 3.12, source for others)

## 🌐 Access URLs

### ComfyUI (100% Reliable)
```
https://tensorboard-[PAPERSPACE_FQDN]/
```
- Uses Paperspace's built-in Tensorboard mapping
- Port 6006 with automatic SSL certificates
- No "Connection Refused" errors

### JupyterLab
```
https://[PAPERSPACE_FQDN]:8889/lab
```
- Root filesystem access enabled
- Port 8889 with no authentication token

## 🎯 Known Issues & Solutions

### Issue: Flash Attention Installation
**Solution**: Automatically handled - PyTorch installed first, then Flash Attention with pre-built wheels for Python 3.12

### Issue: Package Version Conflicts  
**Solution**: Using flexible version constraints (>=) for all packages

### Issue: Git Authentication for Custom Nodes
**Solution**: Problematic nodes commented out, can be installed via ComfyUI Manager

### Issue: Storage Limitations (50GB Free Tier)
**Solution**: Optimized model selection, temporary file cleanup, storage monitoring

## 📚 Documentation Files
- `README.md` - Main project documentation
- `docs/ANLEITUNG_DEUTSCH.md` - German step-by-step guide
- `TROUBLESHOOTING.md` - Common issues and solutions
- `A6000_FREE_TIER.md` - GPU optimization guide
- `AUTO_RESTART.md` - Auto-restart documentation

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.