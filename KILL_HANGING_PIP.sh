#!/bin/bash
# KILL_HANGING_PIP.sh - Beendet hängende pip Prozesse und startet neu

echo "🛑 KILLING HANGING PIP PROCESSES"
echo "================================="

# Kill all pip processes
echo "🔪 Killing all pip processes..."
pkill -9 -f pip
pkill -9 -f "python.*pip"
pkill -9 -f "python.*install"

# Kill specific Python processes that might be stuck
echo "🔪 Killing stuck Python processes..."
pkill -9 -f "python.*torchsde"
pkill -9 -f "python.*torchaudio"
pkill -9 -f "python.*scipy"

# Clean pip cache
echo "🧹 Cleaning pip cache..."
pip cache purge 2>/dev/null || rm -rf ~/.cache/pip/*

# Remove pip locks
echo "🔓 Removing pip locks..."
rm -f /tmp/pip-* 2>/dev/null
rm -rf ~/.cache/pip/selfcheck 2>/dev/null

echo ""
echo "✅ All pip processes killed!"
echo "================================="
echo ""
echo "Now you can run the installation again:"
echo "  ./scripts/quick_fix_dependencies.sh"
echo ""
echo "Or use the fast installer:"
echo "  ./scripts/install_dependencies_fast.sh"
echo ""
echo "Or skip pip and use Docker:"
echo "  DOCKER_MODE=true ./run.sh"