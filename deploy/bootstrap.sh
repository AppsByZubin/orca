#!/usr/bin/env bash
# One-time preparation of a fresh Ubuntu/Debian VM for ORCA staging.
# Idempotent: safe to re-run.
#
#   scp deploy/bootstrap.sh user@vm:/tmp/
#   ssh user@vm 'sudo bash /tmp/bootstrap.sh'
#
# Installs Docker, creates the deploy user, adds swap (the image build needs
# more memory than the containers do), and closes the firewall. It deliberately
# does NOT open port 8000: ORCA has no authentication, so staging is reached
# over an SSH tunnel or a VPN, never a public port.
set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-orca}"
SWAP_GB="${SWAP_GB:-2}"

log() { printf '\n\033[36m==>\033[0m %s\n' "$1"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo bash $0)"; exit 1; }

log "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg rsync ufw sqlite3 ntpsec >/dev/null

log "clock (option expiry maths and token windows are time-sensitive)"
timedatectl set-timezone Asia/Kolkata || true
systemctl enable --now ntpsec 2>/dev/null || systemctl enable --now systemd-timesyncd 2>/dev/null || true
timedatectl | sed 's/^/    /'

if ! command -v docker >/dev/null 2>&1; then
  log "docker"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin >/dev/null
  systemctl enable --now docker
else
  log "docker already installed"
fi
docker --version | sed 's/^/    /'
docker compose version 2>/dev/null | sed 's/^/    /' || true

log "deploy user: $DEPLOY_USER"
id -u "$DEPLOY_USER" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$DEPLOY_USER"
usermod -aG docker "$DEPLOY_USER"
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0755 /opt/orca
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0700 "/home/$DEPLOY_USER/.ssh"
# Let the operator reach the deploy user with the same key they logged in with.
for src in /root/.ssh/authorized_keys "/home/${SUDO_USER:-nobody}/.ssh/authorized_keys"; do
  [ -f "$src" ] && cat "$src" >> "/home/$DEPLOY_USER/.ssh/authorized_keys" || true
done
if [ -f "/home/$DEPLOY_USER/.ssh/authorized_keys" ]; then
  sort -u "/home/$DEPLOY_USER/.ssh/authorized_keys" -o "/home/$DEPLOY_USER/.ssh/authorized_keys"
  chown -R "$DEPLOY_USER:$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh"
  chmod 600 "/home/$DEPLOY_USER/.ssh/authorized_keys"
fi

if [ ! -f /swapfile ] && [ "$SWAP_GB" -gt 0 ]; then
  log "swap (${SWAP_GB}G) - pip installing pandas/sklearn/pyarrow peaks well above idle"
  fallocate -l "${SWAP_GB}G" /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB*1024))
  chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
else
  log "swap already present or disabled"
fi
free -h | sed 's/^/    /'

log "firewall - SSH only, port 8000 stays closed"
ufw --force reset >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
ufw allow OpenSSH >/dev/null
ufw --force enable >/dev/null
ufw status verbose | sed 's/^/    /'

log "done"
cat <<EOF

  Next, from your workstation:

    make staging-deploy HOST=$DEPLOY_USER@<vm-ip>

  Then reach it over an SSH tunnel (port 8000 is firewalled off):

    ssh -N -L 8000:127.0.0.1:8000 $DEPLOY_USER@<vm-ip>
    open http://127.0.0.1:8000/docs

EOF
