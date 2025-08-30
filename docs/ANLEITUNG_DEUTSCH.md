# ComfyUI Paperspace Notebook - Schritt-für-Schritt Anleitung

Eine produktionsreife ComfyUI-Installation optimiert für Paperspace Gradient mit persistentem Speicher, JupyterLab-Entwicklungsumgebung und intelligentem Auto-Neustart.

## 🚀 Schnellstart (Ein-Kommando-Setup)

### 1. Paperspace Gradient Notebook öffnen
- Gehen Sie zu [Paperspace Gradient](https://gradient.paperspace.com)
- Erstellen Sie ein neues Notebook mit GPU-Unterstützung
- **Empfohlen**: RTX A6000 (48GB VRAM, kostenlos verfügbar)

### 2. Repository klonen und starten

**⚠️ WICHTIG: Verwenden Sie die RICHTIGE Repository-URL!**

**Option A: Wenn Sie bereits in diesem Verzeichnis sind (empfohlen):**
```bash
!chmod +x run.sh
!./run.sh
```

**Option B: Wenn Sie ein neues Notebook starten:**
```bash
# Korrekte Repository-URL
!git clone https://github.com/Innomatix46/comfyui-paperspace-notebook.git
%cd comfyui-paperspace-notebook
!chmod +x run.sh
!./run.sh
```

**Option C: Von diesem Repository aus (direkter Start):**
```bash
# Direkt im aktuellen Verzeichnis
!./run.sh
```

### 3. Warten Sie auf die automatische Installation
Das Skript wird automatisch:
- ✅ ComfyUI mit CUDA 12.4-Optimierung installieren
- ✅ 6 wichtige Custom Nodes einrichten
- ✅ JupyterLab mit Root-Zugriff konfigurieren
- ✅ Konfigurierte Modelle herunterladen
- ✅ Auto-Neustart-Scheduler für 6-Stunden-Zyklen starten
- ✅ ComfyUI auf Port 6006 starten (Tensorboard-Mapping)

### 4. Zugriff auf Ihre Services
Nach der Installation erhalten Sie URLs in der Ausgabe:

**🎯 ComfyUI-Zugang (100% funktionierend):**
```
https://tensorboard-[PAPERSPACE_FQDN]/
```

**🔬 JupyterLab-Entwicklungsumgebung:**
```
https://[PAPERSPACE_FQDN]/lab/
```

## 📋 Detaillierte Anleitung

### Phase 1: Vorbereitung
1. **GPU-Instanz auswählen**: RTX A6000 für optimale Performance (48GB VRAM)
2. **Speicherplatz prüfen**: Mindestens 50GB für Free Tier
3. **Repository klonen**: `git clone` Befehl ausführen

### Phase 2: Installation
Das `run.sh` Skript führt folgende Schritte durch:

#### 2.1 Umgebung einrichten
- Projektpfade definieren (`/storage/ComfyUI` für Persistenz)
- Wichtige Verzeichnisse erstellen
- Umgebungsvariablen setzen

#### 2.2 Abhängigkeiten installieren
- ComfyUI Repository klonen
- Python Virtual Environment erstellen (3.12 bevorzugt, 3.11+ unterstützt)
- PyTorch 2.6+ + CUDA 12.4 installieren
- Flash Attention automatisch für CUDA-Umgebungen (Pre-built Wheels für Python 3.12)
- Custom Nodes installieren (siehe `configs/custom_nodes.txt`)

#### 2.3 Modelle herunterladen
- Modelle aus `configs/models.txt` herunterladen
- Persistent in `/storage/ComfyUI/models/` speichern
- Symlinks zu ComfyUI erstellen

#### 2.4 JupyterLab konfigurieren
- JupyterLab mit Root-Zugriff installieren
- Auf Port 8889 konfigurieren
- Automatisch im Hintergrund starten

#### 2.5 Auto-Neustart einrichten
- 6-Stunden-Neustart-Zyklen aktivieren
- GPU-Speicher-Management
- Graceful Shutdown-Prozess

### Phase 3: Anwendung starten
- ComfyUI mit A6000-optimierten Parametern starten
- Port 6006 für Tensorboard-URL-Mapping verwenden
- CORS-Header für externen Zugriff aktivieren

## ⚙️ Konfiguration und Anpassung

### Modelle hinzufügen/entfernen
Bearbeiten Sie `configs/models.txt`:

```bash
# Format: [Unterverzeichnis] [Download-URL]
checkpoints https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
loras https://civitai.com/api/download/models/16576
vae https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
```

### Custom Nodes hinzufügen
Bearbeiten Sie `configs/custom_nodes.txt`:

```bash
# Standardmäßig enthalten:
https://github.com/ltdrdata/ComfyUI-Manager
https://github.com/ltdrdata/ComfyUI-Impact-Pack
https://github.com/Gourieff/ComfyUI-ReActor-Node

# Ihre eigenen Nodes hinzufügen:
https://github.com/author/new-custom-node
```

### Python-Pakete hinzufügen
Bearbeiten Sie `configs/python_requirements.txt`:

```bash
# Beispiel für zusätzliche Pakete
transformers==4.46.3
accelerate==1.2.1
custom-package>=1.0.0
```

## 🎯 Tensorboard-Zugang (100% funktionierend)

### Warum Port 6006?
Paperspace bietet eingebautes Tensorboard-URL-Mapping für Port 6006:
- ✅ Automatische SSL-Zertifikate
- ✅ Keine "Connection Refused"-Fehler
- ✅ Zuverlässiger externer Zugriff
- ✅ Kein Port-Forwarding erforderlich

### URL-Format
```
https://tensorboard-[IHRE_PAPERSPACE_FQDN]/
```

### Beispiel
Wenn Ihre Paperspace-FQDN `abc123.clg07azjl.paperspacegradient.com` ist:
```
https://tensorboard-abc123.clg07azjl.paperspacegradient.com/
```

## 📊 Speicher-Optimierung (Free Tier 50GB)

### Speicher-Aufteilung
- **System + ComfyUI**: ~15GB
- **Modelle**: ~25GB (konfigurierbar)
- **Arbeitsbereich + Ausgaben**: ~10GB

### Speicher-Status prüfen
```bash
./scripts/storage_optimizer.sh status
```

### Speicher freigeben
```bash
# Temporäre Dateien löschen
rm -rf /tmp/*
rm -rf ComfyUI/temp/*

# Alte Ausgaben löschen
rm -rf /storage/ComfyUI/output/old_files
```

## 🔧 Manuelle Operationen

### Nur Abhängigkeiten neu installieren
```bash
source scripts/install_dependencies.sh && install_dependencies
```

### Nur Modelle neu herunterladen
```bash
# Init-Flag zurücksetzen
rm /storage/ComfyUI/.init_done
# Modelle neu herunterladen
source scripts/download_models.sh && download_models
```

### JupyterLab neu starten
```bash
source scripts/configure_jupyterlab.sh && configure_jupyterlab && start_jupyterlab
```

### Auto-Neustart verwalten
```bash
./restart-control.sh status          # Status prüfen
./restart-control.sh enable          # Aktivieren
./restart-control.sh disable         # Deaktivieren
./restart-control.sh force-restart   # Sofort neu starten
./restart-control.sh logs            # Logs anzeigen
```

## 🛠️ Fehlerbehebung

### Häufige Probleme und Lösungen

**❌ FEHLER: "fatal: Too many arguments" beim Git Clone**
**✅ LÖSUNG**: 
- Verwenden Sie die korrekte Repository-URL ohne extra Zeichen
- Oder starten Sie direkt mit `!./run.sh` wenn Sie bereits im Verzeichnis sind

**❌ FEHLER: Flash Attention Installation schlägt fehl**
**✅ LÖSUNG**: 

**Lokale Entwicklung (macOS/Windows ohne CUDA):**
```bash
# Flash Attention ist nur für CUDA-GPUs optimiert
# Für lokale Tests ohne CUDA skip Flash Attention:
source venv/bin/activate
pip install torch torchvision torchaudio
# Flash Attention überspringen - ComfyUI funktioniert ohne

# Alternative: xformers verwenden
pip install xformers
```

**Paperspace/CUDA-Umgebung:**
```bash
# Erst CUDA-Version prüfen
nvcc --version

# PyTorch mit CUDA installieren
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# METHODE 1: Pre-built Wheels (EMPFOHLEN)
# Python 3.12, CUDA 12.4, PyTorch 2.8
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl

# METHODE 2: Build from source (Falls Pre-built nicht verfügbar)
# Build-Dependencies installieren
pip install ninja packaging wheel setuptools
# Flash Attention mit korrekter CUDA-Umgebung
export CUDA_HOME=/usr/local/cuda
pip install flash-attn --no-build-isolation
```

**❌ FEHLER: "chmod: cannot access '│': No such file or directory"**
**✅ LÖSUNG**: 
- Kopieren Sie die Befehle einzeln, nicht den ganzen Block
- Stellen Sie sicher, dass keine versteckten Zeichen vorhanden sind

**❌ FEHLER: JupyterLab zeigt nicht alle Ordner - fehlender Root-Zugriff**
**✅ LÖSUNG**:

**Lokale Entwicklung (macOS/Windows):**
```bash
# JupyterLab mit Root-Zugriff konfigurieren
source scripts/configure_jupyterlab.sh && configure_jupyterlab

# JupyterLab mit Root-Zugriff starten
source venv/bin/activate
jupyter lab --allow-root --port=8889 --ip=0.0.0.0 --no-browser --config ~/.jupyter/jupyter_lab_config.py

# Zugriff über Browser:
# http://127.0.0.1:8889/lab
```

**Paperspace-Umgebung:**
```bash
# Automatisch konfiguriert durch run.sh Script
./run.sh

# Manueller Start:
source scripts/configure_jupyterlab.sh && configure_jupyterlab && start_jupyterlab
```

**Wichtige Konfigurationseinstellungen:**
```python
# ~/.jupyter/jupyter_lab_config.py
c.ServerApp.allow_root = True
c.ServerApp.root_dir = '/'  # Root-Zugriff auf gesamtes System
c.ServerApp.notebook_dir = '/'  # Startet im Root-Verzeichnis
c.ServerApp.ip = '0.0.0.0'  # Zugriff von extern
c.ServerApp.token = ''  # Kein Token für einfachen Zugriff
```

**❌ FEHLER: Kann nicht auf ComfyUI zugreifen**
**✅ LÖSUNG**: 
- Verwenden Sie die Tensorboard-URL `https://tensorboard-[FQDN]/`
- NICHT die Standard-URL mit Port 8188

**❌ FEHLER: "Connection Refused" auf Port 8188**
**✅ LÖSUNG**: 
- ComfyUI läuft korrekt, aber Sie müssen die richtige URL verwenden
- URL-Format: `https://tensorboard-[IHRE_PAPERSPACE_FQDN]/`
- Beispiel: `https://tensorboard-n17g9ovffm.clg07azjl.paperspacegradient.com/`

**❌ FEHLER: Modelle laden nicht**
**✅ LÖSUNG**: 
- Symlinks prüfen: `!ls -la ComfyUI/models/`
- Init-Flag zurücksetzen: `!rm /storage/ComfyUI/.init_done`
- Modelle neu herunterladen: `!./run.sh`

**❌ FEHLER: Custom Node-Installationsfehler**
**✅ LÖSUNG**: 
- Installations-Logs prüfen: `!tail -f /tmp/comfyui_install_*.log`
- Einzelne Nodes deaktivieren in `configs/custom_nodes.txt`

**❌ FEHLER: Speicher-Probleme / 50GB überschritten**
**✅ LÖSUNG**: 
- Status prüfen: `!./scripts/storage_optimizer.sh status`
- Temporäre Dateien löschen: `!rm -rf /tmp/* ComfyUI/temp/*`
- Modellkonfiguration anpassen in `configs/models.txt`

### Log-Dateien prüfen
```bash
# ComfyUI-Logs
tail -f /storage/ComfyUI/comfyui.log

# JupyterLab-Logs
tail -f /storage/jupyterlab.log

# Auto-Neustart-Logs
tail -f /storage/ComfyUI/restart.log

# Installations-Logs
ls -la /tmp/comfyui_install_*.log
```

### System-Status prüfen
```bash
# GPU-Status
nvidia-smi

# Speicher-Verwendung
df -h

# Laufende Prozesse
ps aux | grep -E "(python|jupyter|comfy)"

# Netzwerk-Ports
netstat -tlnp | grep -E "(6006|8889)"
```

## 📋 Checkliste für erfolgreiche Installation

- [ ] Paperspace Notebook mit GPU-Instanz erstellt
- [ ] Repository erfolgreich geklont
- [ ] `run.sh` ohne Fehler ausgeführt
- [ ] ComfyUI-URL `https://tensorboard-[FQDN]/` funktioniert
- [ ] JupyterLab-URL `https://[FQDN]/lab/` zugänglich
- [ ] Modelle erfolgreich geladen (Manager → Install Models)
- [ ] Auto-Neustart aktiviert (alle 6 Stunden)
- [ ] Speicher-Monitoring funktioniert

## 💡 Tipps für optimale Performance

### A6000-spezifische Optimierungen
- **Flash Attention**: Automatisch aktiviert für 48GB VRAM
- **LowVRAM-Modus**: Deaktiviert, da A6000 genügend Speicher hat
- **Split Cross-Attention**: Für Speicher-Optimierung aktiviert
- **CUDA 12.4**: Neueste Optimierungen verfügbar

### Workflow-Optimierungen
1. **Batch-Generierung**: Nutzen Sie die 48GB VRAM für große Batches
2. **Modell-Caching**: Modelle bleiben im VRAM für schnellere Generierung
3. **Pipeline-Parallelität**: Mehrere Workflows gleichzeitig möglich

### Speicher-Best Practices
- Verwenden Sie `.safetensors`-Format für schnelleres Laden
- Deaktivieren Sie nicht verwendete Custom Nodes
- Löschen Sie regelmäßig alte Ausgaben
- Überwachen Sie `/storage` Speicherplatz

## 🚀 Erweiterte Features

### SPARC-Workflow-Unterstützung
Das Projekt unterstützt SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) Methodologie:

```bash
# SPARC-Modi anzeigen
npx claude-flow sparc modes

# TDD-Workflow ausführen
npx claude-flow sparc tdd "feature-name"
```

### Claude-Flow-Integration
Automatische Swarm-Koordination für komplexe Tasks:

```bash
# Swarm initialisieren
npx claude-flow@alpha mcp start

# Agent spawnen
npx claude-flow@alpha hooks pre-task --description "task"
```

## 📞 Support

- **Dokumentation**: Siehe README.md und spezifische .md-Dateien
- **Issues**: GitHub Issues für Problemberichte
- **Logs**: Immer Logs mit Fehlberichten einschließen

---

**🎯 Hinweis**: Diese Anleitung fokussiert sich auf die 100% funktionierende Tensorboard-URL-Methode für Paperspace-Zugang. Andere Zugriffsmethoden können unzuverlässig sein.