#!/usr/bin/env bash
# Two-model daily setup (tested): Qwen3.8-27B on port 8000 (share 0.34, 131k context)
# + Qwen3.6-35B-A3B on port 8001 (share 0.31, 131k context), each its own systemd service.
#   install:  sudo bash ~/ailab/safety/install-two-models.sh
#   undo:     sudo bash ~/ailab/safety/install-two-models.sh --undo   (back to 27B alone at 0.60 / 262k)
# Safe to re-run. Does not touch ssh, tailscaled or the other fail-safes.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
U=YOUR_USER
H=/home/$U
S=$H/ailab/safety
V=$H/ailab/vllm
SD=/etc/systemd/system
ok() { printf '  [ok] %s\n' "$*"; }

if [ "${1:-}" = "--undo" ]; then
  echo "Undo: Qwen3.6 off, Qwen3.8-27B back to 0.60 / 262k alone"
  systemctl disable --now ailab-vllm-qwen36.service >/dev/null 2>&1 || true
  sed -i 's/^MEM=0.34$/MEM=0.60/; s/^MAXLEN=131072$/MAXLEN=262144/' "$V/qwen38-27b.sh"
  sed -i 's|vllm-precheck.sh 65 ailab-vllm-qwen36:8001|vllm-precheck.sh 90|' "$S/ailab-vllm.service"
  install -m 0644 -o root -g root "$S/ailab-vllm.service" "$SD/ailab-vllm.service"
  systemctl daemon-reload
  systemctl restart ailab-vllm.service
  ok "ailab-vllm-qwen36 $(systemctl is-enabled ailab-vllm-qwen36 2>/dev/null || echo removed); ailab-vllm restarting alone (ready in ~5 min)"
  exit 0
fi

echo "1. Back up the current units"
B=$S/backup-$(date +%Y%m%d-%H%M%S)
mkdir -p "$B"
for f in ailab-vllm.service ailab-vllm-qwen36.service ailab-memwatch.service; do [ -f "$SD/$f" ] && cp -a "$SD/$f" "$B/"; done
cp -a "$V/qwen38-27b.sh" "$B/" 2>/dev/null || true
chown -R $U:$U "$B"
ok "$B"

echo "2. Install the units (memwatch now stops both model services in an emergency)"
chmod 0755 "$S/vllm-precheck.sh" "$V/qwen38-27b.sh" "$V/qwen36-35b.sh"
for f in ailab-memwatch.service ailab-vllm.service ailab-vllm-qwen36.service; do
  install -m 0644 -o root -g root "$S/$f" "$SD/$f"
done
systemctl daemon-reload
systemctl restart ailab-memwatch.service
sleep 2
ok "ailab-memwatch $(systemctl is-active ailab-memwatch), stops: $(systemctl show ailab-memwatch -p Environment --value)"

echo "3. Restart Qwen3.8-27B with the smaller share (port 8000 offline ~5 min)"
systemctl restart ailab-vllm.service
ok "ailab-vllm $(systemctl is-active ailab-vllm) / $(systemctl is-enabled ailab-vllm)"

echo "4. Enable and start Qwen3.6 (port 8001). It waits for 27B to finish loading, then ~4 min to load"
systemctl enable ailab-vllm-qwen36.service >/dev/null 2>&1
systemctl start --no-block ailab-vllm-qwen36.service
ok "ailab-vllm-qwen36 starting / $(systemctl is-enabled ailab-vllm-qwen36)"

echo
echo "Both models ready in ~10 min. Check with:"
echo "  systemctl status ailab-vllm ailab-vllm-qwen36"
echo "  journalctl -u ailab-vllm-qwen36 -f"
