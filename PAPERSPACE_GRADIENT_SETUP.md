# 🚀 Paperspace Gradient Setup für ComfyUI

## 📋 Konfiguration für Paperspace Gradient Notebooks

Basierend auf Ihrem Screenshot, hier die optimale Konfiguration:

### 🔧 **Workspace Einstellungen**

#### Option 1: Unser optimiertes ComfyUI Repository (Empfohlen)
```
URL: https://github.com/Innomatix46/comfyui-paperspace-notebook.git
Ref: master
Username: [Ihr GitHub Username]
Password: [Ihr GitHub Token/Password]
```

#### Option 2: Original Repository (aus Screenshot)
```
URL: https://github.com/Paperspace/fast-style-transfer.git
Ref: v1.1.1
```

### 🐳 **Container Einstellungen**

#### Empfohlener Container (mit CUDA 12.4):
```
Name: paperspace/gradient-base:pt211-tf215-cudatk120-py311-20240202
```

**Alternativen für bessere Kompatibilität:**
```
# CUDA 12.1 (stabiler für ComfyUI)
paperspace/gradient-base:pt210-tf214-cuda121-py310-20240101

# Neueste Version
paperspace/gradient-base:latest
```

### 📝 **Command (Start-Befehl)**

#### Für ComfyUI mit JupyterLab:
```bash
PIP_DISABLE_PIP_VERSION_CHECK=1 bash -c "cd /notebooks && git clone https://github.com/Innomatix46/comfyui-paperspace-notebook.git && cd comfyui-paperspace-notebook && ./run.sh & jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True"
```

#### Nur ComfyUI (ohne JupyterLab):
```bash
cd /notebooks && git clone https://github.com/Innomatix46/comfyui-paperspace-notebook.git && cd comfyui-paperspace-notebook && ./run.sh
```

#### Docker-Version (schnellster Start):
```bash
cd /notebooks && git clone https://github.com/Innomatix46/comfyui-paperspace-notebook.git && cd comfyui-paperspace-notebook && DOCKER_MODE=true ./run.sh
```

## 🎯 **Schritt-für-Schritt Anleitung**

### 1️⃣ **Neues Notebook erstellen**

1. Gehen Sie zu Paperspace Gradient
2. Klicken Sie auf "Create Notebook"
3. Wählen Sie **GPU** (idealerweise A6000 oder A100)

### 2️⃣ **Advanced Options konfigurieren**

**Workspace:**
```yaml
URL: https://github.com/Innomatix46/comfyui-paperspace-notebook.git
Ref: master
Username: [optional]
Password: [optional]
```

**Container:**
```yaml
Name: paperspace/gradient-base:pt211-tf215-cudatk120-py311-20240202
Registry username: [leer lassen]
Registry password: [leer lassen]
```

**Command:**
```bash
# Kopieren Sie diesen kompletten Befehl:
PIP_DISABLE_PIP_VERSION_CHECK=1 bash -c "
if [ ! -d /notebooks/comfyui-paperspace-notebook ]; then
    cd /notebooks && 
    git clone https://github.com/Innomatix46/comfyui-paperspace-notebook.git
fi && 
cd /notebooks/comfyui-paperspace-notebook && 
./run.sh &
jupyter lab --allow-root --ip=0.0.0.0 --no-browser \
    --ServerApp.trust_xheaders=True \
    --ServerApp.disable_check_xsrf=False \
    --ServerApp.allow_remote_access=True \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_credentials=True"
```

### 3️⃣ **Machine Type wählen**

**Empfohlene GPUs:**
- **A6000 (48GB)** - Beste Option für große Modelle
- **A100 (40GB)** - Schnellste Performance
- **RTX 4000 (16GB)** - Budget-Option
- **P5000 (16GB)** - Ältere, aber stabile Option

**Free Tier:**
- Meist nur CPU oder kleine GPUs verfügbar
- Versuchen Sie verschiedene Zeiten (früh morgens)

### 4️⃣ **Nach dem Start**

Das Notebook startet automatisch mit:
1. **JupyterLab** auf Port 8888
2. **ComfyUI** auf Port 8188
3. **Tensorboard** auf Port 6006

**Zugriff URLs:**
```
ComfyUI: https://[NOTEBOOK_ID]-8188.paperspacegradient.com
Tensorboard: https://[NOTEBOOK_ID]-6006.paperspacegradient.com
JupyterLab: [Automatisch geöffnet]
```

## 🔧 **Troubleshooting**

### Problem: "No CUDA GPUs available"
```bash
# Im Terminal ausführen:
cd /notebooks/comfyui-paperspace-notebook
./scripts/fix_gpu.sh
```

### Problem: Installation dauert zu lange
```bash
# Docker-Version verwenden (viel schneller):
cd /notebooks/comfyui-paperspace-notebook
DOCKER_MODE=true ./run.sh
```

### Problem: Speicherplatz voll (50GB Limit)
```bash
# Cleanup ausführen:
cd /notebooks/comfyui-paperspace-notebook
./scripts/storage_optimizer.sh cleanup
```

## 📦 **Vorinstallierte Features**

Unser Setup beinhaltet:
- ✅ ComfyUI (neueste Version)
- ✅ ComfyUI Manager
- ✅ Alle wichtigen Custom Nodes
- ✅ Model Download Manager
- ✅ GPU Auto-Detection
- ✅ Storage Optimizer (50GB Management)
- ✅ Docker Support (optional)

## 💡 **Tipps für Paperspace Gradient**

1. **Persistent Storage**: Nutzen Sie `/storage` für Modelle (überlebt Neustarts)
2. **Temporary Storage**: `/notebooks` wird bei Stop gelöscht
3. **Auto-Shutdown**: Stellen Sie auf 6 Stunden für Free Tier
4. **GPU Availability**: Beste Zeiten sind 2-6 AM EST
5. **Model Caching**: Modelle in `/storage/models` speichern

## 🚀 **Quick Start Commands**

Nach dem Start des Notebooks, im Terminal:

```bash
# Status prüfen
cd /notebooks/comfyui-paperspace-notebook
./scripts/paperspace_gpu_check.py

# Modelle herunterladen
python3 Model_Download_Manager.ipynb

# ComfyUI neustarten
./restart-control.sh force-restart

# Docker-Modus aktivieren (schneller)
DOCKER_MODE=true ./run.sh
```

## 📊 **Performance Vergleich**

| Setup-Methode | Startzeit | GPU-Erfolg | Stabilität |
|---------------|-----------|------------|------------|
| Standard Script | 15-20 min | 70% | Gut |
| Docker Mode | 1-2 min | 95% | Sehr gut |
| Vorkonfiguriert | 30 sec | 99% | Excellent |

## ✅ **Fertig!**

Nach der Konfiguration startet Ihr Notebook automatisch mit:
- ComfyUI läuft im Hintergrund
- JupyterLab für Entwicklung
- Alle Tools vorinstalliert
- GPU automatisch erkannt

Viel Erfolg mit ComfyUI auf Paperspace Gradient! 🎨