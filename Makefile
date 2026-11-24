.PHONY: help env install serve worker test lint fmt scan snapshot track train demo clean \
        staging-build staging-up staging-down staging-logs staging-smoke staging-shell \
        staging-seed-model staging-deploy staging-bootstrap staging-tunnel

# ---------------------------------------------------------------------------
# Conda environment. Targets work with or without `conda activate orca`:
# every command runs the interpreter in the env by absolute path.
# Override with:  make test CONDA_ENV=orca-dev
# ---------------------------------------------------------------------------
CONDA_ENV ?= orca
PYTHON_VERSION ?= 3.12
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo $$HOME/anaconda3)
BIN := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin

PY := $(BIN)/python
PIP := $(BIN)/pip
ORCA := $(BIN)/orca

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

env: ## create the conda environment
	conda create -y -n $(CONDA_ENV) python=$(PYTHON_VERSION)

install: ## install ORCA with dev extras into the conda env
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

serve: ## run the JSON API at http://127.0.0.1:8000 (docs at /docs)
	$(ORCA) serve --reload

worker: ## run only the background jobs
	$(ORCA) worker

test: ## run the test suite
	$(BIN)/pytest -q

lint: ## ruff check
	$(BIN)/ruff check orca tests

fmt: ## ruff format + fix
	$(BIN)/ruff format orca tests
	$(BIN)/ruff check --fix orca tests

scan: ## rank candidates from the CLI
	$(ORCA) scan

snapshot: ## capture one chain snapshot per underlying
	$(ORCA) snapshot

track: ## mark open trades once
	$(ORCA) track

train: ## bootstrap the model on synthetic data
	$(ORCA) train --synthetic

demo: ## scan -> enter -> track -> exit on the mock feed
	$(ORCA) demo

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info

# ---------------------------------------------------------------------------
# Staging: two containers from one image, loopback-only, paper gateway.
# Uses plain `docker run` so it works without the compose v2 plugin;
# deploy/compose.yaml is the equivalent if you have it.
# ---------------------------------------------------------------------------
IMAGE   ?= orca:staging
NET     ?= orca-staging
VOLUME  ?= orca-staging-data
ENVFILE ?= deploy/staging.env
PORT    ?= 8000

staging-build: ## build the staging image
	docker build -t $(IMAGE) .

staging-up: staging-build ## start api + worker (http://127.0.0.1:$(PORT))
	-docker network create $(NET) 2>/dev/null
	-docker volume create $(VOLUME) 2>/dev/null
	-docker rm -f orca-api orca-worker 2>/dev/null
	docker run -d --name orca-api --network $(NET) \
	  --env-file $(ENVFILE) -e ORCA_SCHEDULER_ENABLED=false \
	  -v $(VOLUME):/data -p 127.0.0.1:$(PORT):8000 \
	  --restart unless-stopped $(IMAGE)
	@# --no-healthcheck: the worker runs no web server, so the image's HTTP
	@# probe can never pass. A crashed worker exits, and restart handles that.
	docker run -d --name orca-worker --network $(NET) \
	  --env-file $(ENVFILE) -e ORCA_SCHEDULER_ENABLED=true \
	  -v $(VOLUME):/data --no-healthcheck \
	  --restart unless-stopped $(IMAGE) orca worker
	@echo "api:    http://127.0.0.1:$(PORT)/docs"
	@echo "verify: make staging-smoke"

staging-down: ## stop and remove the staging containers (volume kept)
	-docker rm -f orca-api orca-worker

staging-logs: ## follow both containers
	docker logs -f --tail 50 orca-api & docker logs -f --tail 50 orca-worker; wait

staging-shell: ## shell into the api container
	docker exec -it orca-api /bin/bash

staging-seed-model: ## copy the local trained model into the staging volume
	@test -f data/models/$(or $(MODEL),default).joblib \
	  || { echo "no data/models/$(or $(MODEL),default).joblib - run 'make train' first"; exit 1; }
	docker cp data/models/$(or $(MODEL),default).joblib orca-api:/data/models/
	docker cp data/models/$(or $(MODEL),default).json    orca-api:/data/models/
	@# get_model() caches the model per process, so the API must restart to see it
	docker restart orca-api >/dev/null && echo "model seeded; api restarted"

staging-smoke: ## assert staging is healthy and safely configured
	./scripts/staging_smoke.sh http://127.0.0.1:$(PORT)

staging-bootstrap: ## one-time VM prep: make staging-bootstrap HOST=user@ip
	@test -n "$(HOST)" || { echo "usage: make staging-bootstrap HOST=user@ip"; exit 1; }
	scp deploy/bootstrap.sh $(HOST):/tmp/orca-bootstrap.sh
	ssh -t $(HOST) 'sudo bash /tmp/orca-bootstrap.sh'

staging-deploy: ## push + rebuild + restart on a VM: make staging-deploy HOST=user@ip
	@test -n "$(HOST)" || { echo "usage: make staging-deploy HOST=user@ip"; exit 1; }
	./deploy/deploy.sh $(HOST) $(or $(REMOTE_DIR),/opt/orca)

staging-tunnel: ## forward the VM's API to localhost:$(PORT)
	@test -n "$(HOST)" || { echo "usage: make staging-tunnel HOST=user@ip"; exit 1; }
	@echo "http://127.0.0.1:$(PORT)/docs  (Ctrl-C to close)"
	ssh -N -L $(PORT):127.0.0.1:8000 $(HOST)
