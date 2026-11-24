#!/usr/bin/env bash
# Push this checkout to a staging VM, rebuild the image there, and restart.
#
#   ./deploy/deploy.sh orca@203.0.113.10          [remote-dir]
#   make staging-deploy HOST=orca@203.0.113.10
#
# rsync, not git: this project has no repository, so the working tree is the
# source of truth. The transfer is ~550 KB; the 785 MB image is built on the VM
# so nothing large crosses the wire.
set -euo pipefail

HOST="${1:-}"
REMOTE_DIR="${2:-/opt/orca}"
[ -n "$HOST" ] || { echo "usage: $0 user@host [remote-dir]"; exit 1; }

log() { printf '\n\033[36m==>\033[0m %s\n' "$1"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log "checking the VM"
ssh -o BatchMode=yes "$HOST" '
  command -v docker >/dev/null || { echo "docker missing - run deploy/bootstrap.sh first"; exit 1; }
  docker info >/dev/null 2>&1 || { echo "cannot talk to the docker daemon (is this user in the docker group? re-login after bootstrap)"; exit 1; }
' || exit 1

log "syncing source -> $HOST:$REMOTE_DIR"
ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude '.venv/' --exclude '.git/' --exclude '__pycache__/' \
  --exclude '*.pyc' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude 'data/' \
  --exclude '.env' \
  --include 'deploy/staging.env' \
  "$HERE/" "$HOST:$REMOTE_DIR/"

log "building the image on the VM (first run takes a few minutes)"
ssh "$HOST" "cd '$REMOTE_DIR' && docker build -t orca:staging ."

log "restarting containers"
# shellcheck disable=SC2029  # REMOTE_DIR is ours, expansion here is intended
ssh "$HOST" "cd '$REMOTE_DIR' && make staging-down >/dev/null 2>&1; make staging-up"

log "waiting for health"
ssh "$HOST" '
  for i in $(seq 1 24); do
    s=$(docker inspect -f "{{.State.Health.Status}}" orca-api 2>/dev/null || echo missing)
    [ "$s" = "healthy" ] && { echo "  healthy after $((i*5))s"; exit 0; }
    sleep 5
  done
  echo "  still not healthy - docker logs orca-api"; exit 1
' || { ssh "$HOST" 'docker logs --tail 40 orca-api'; exit 1; }

log "smoke test (on the VM, against loopback)"
ssh "$HOST" "cd '$REMOTE_DIR' && ./scripts/staging_smoke.sh http://127.0.0.1:8000" || true

cat <<EOF

  Deployed. Port 8000 is firewalled; reach it with a tunnel:

    ssh -N -L 8000:127.0.0.1:8000 $HOST
    open http://127.0.0.1:8000/docs

  Seed a trained model into the volume (optional):
    scp data/models/default.* $HOST:/tmp/ && ssh $HOST 'docker cp /tmp/default.joblib orca-api:/data/models/ && docker cp /tmp/default.json orca-api:/data/models/ && docker restart orca-api'

EOF
