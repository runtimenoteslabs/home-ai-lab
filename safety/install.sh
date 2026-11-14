#!/usr/bin/env bash
# ailab fail-safes. Run once with:  sudo bash ~/ailab/safety/install.sh
# Safe to re-run. Does NOT restart ssh or tailscaled (so your SSH session survives).
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
U=YOUR_USER
H=/home/$U
S=$H/ailab/safety
ok() { printf '  [ok] %s\n' "$*"; }

echo "1. Never suspend (a headless server must not sleep, e.g. at the login screen after a reboot)"
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target >/dev/null 2>&1
ok "sleep/suspend/hibernate targets masked"

echo "2. Reboot instead of hanging after a kernel panic or oops"
cat > /etc/sysctl.d/90-ailab-failsafe.conf <<'EOF'
# ailab: reboot 10 s after a kernel panic; treat an oops as a panic
kernel.panic = 10
kernel.panic_on_oops = 1
EOF
sysctl -q -p /etc/sysctl.d/90-ailab-failsafe.conf
ok "kernel.panic=$(sysctl -n kernel.panic) kernel.panic_on_oops=$(sysctl -n kernel.panic_on_oops)"

echo "3. Hardware watchdog: reboot the box if the system freezes"
mkdir -p /etc/systemd/system.conf.d
cat > /etc/systemd/system.conf.d/90-ailab-watchdog.conf <<'EOF'
[Manager]
RuntimeWatchdogSec=30s
RebootWatchdogSec=10min
EOF
systemctl daemon-reexec
sleep 2
ok "systemd $(systemctl show -p RuntimeWatchdogUSec --value) ; hw: $(cat /sys/class/watchdog/watchdog0/identity 2>/dev/null) state=$(cat /sys/class/watchdog/watchdog0/state 2>/dev/null) timeout=$(cat /sys/class/watchdog/watchdog0/timeout 2>/dev/null)s"

echo "4. Keep remote access alive under memory pressure (OOM killer avoids ssh/tailscaled)"
for svc in tailscaled ssh; do
  mkdir -p /etc/systemd/system/$svc.service.d
  printf '[Service]\nOOMScoreAdjust=-900\n' > /etc/systemd/system/$svc.service.d/90-ailab-oom.conf
done
systemctl daemon-reload
for p in $(pgrep -x tailscaled) $(pgrep -x sshd) $(pgrep -f '^sshd: ' || true); do echo -900 > /proc/$p/oom_score_adj 2>/dev/null || true; done
ok "tailscaled and sshd: oom_score_adj -900 (applied now and on future starts, no restart)"

echo "5. Memory watchdog (ailab-memwatch)"
install -m 0755 -o root -g root "$S/ailab-memwatch" /usr/local/sbin/ailab-memwatch
install -m 0644 -o root -g root "$S/ailab-memwatch.service" /etc/systemd/system/ailab-memwatch.service
install -d -o $U -g $U $H/ailab/logs $H/ailab/logs/memwatch $H/ailab/logs/vllm
systemctl daemon-reload
systemctl enable --now ailab-memwatch.service >/dev/null 2>&1
systemctl restart ailab-memwatch.service
sleep 2
ok "ailab-memwatch: $(systemctl is-active ailab-memwatch) / $(systemctl is-enabled ailab-memwatch)"

echo "6. Model server under systemd (starts at boot after Tailscale, restarts on crash, max 3 tries/hour)"
chmod 0755 "$S/vllm-precheck.sh"
install -m 0644 -o root -g root "$S/ailab-vllm.service" /etc/systemd/system/ailab-vllm.service
systemctl daemon-reload
if docker ps --format '{{.Names}}' | grep -qx vllm-qwen38-27b && ! systemctl is-active --quiet ailab-vllm; then
  echo "  stopping the hand-started container (model offline ~5 min while it restarts under systemd)"
  docker stop -t 30 vllm-qwen38-27b >/dev/null
  sleep 5
fi
systemctl enable ailab-vllm.service >/dev/null 2>&1
systemctl restart ailab-vllm.service
sleep 3
ok "ailab-vllm: $(systemctl is-active ailab-vllm) / $(systemctl is-enabled ailab-vllm) (model ready in ~5 min)"

echo "7. Open WebUI: recreate with log size limits"
sudo -u $U -H bash -c "$H/ailab/open-webui/run.sh" >/dev/null && ok "open-webui recreated"

echo
echo "Done. Check with:"
echo "  systemctl status ailab-memwatch ailab-vllm"
echo "  journalctl -u ailab-memwatch -f"
echo "Still manual (needs you at the box): BIOS 'Restore on AC power loss' -> Power On."
