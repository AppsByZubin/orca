# ORCA

**O**ptions **R**isk & **C**andidate **A**nalyzer — a web-based desk for selling index options on
NIFTY and SENSEX.

ORCA does three things:

1. **Pulls market data** from Upstox (option chains with greeks, OI, quotes, daily candles).
2. **Ranks option-selling candidates** with a machine-learning model — "if I sell this structure
   now, will it reach my profit target before my stop?"
3. **Tracks the multi-leg trade** once it is live: leg-level marks, structure-level P&L and greeks,
   breakeven/stop/target rules, partial-leg exits, and a stored MTM history.

Orders are placed on **Zerodha Kite**; market data comes from **Upstox**. Those are two separate
interfaces (`orca/feeds/` and `orca/execution/`), so either side can be swapped without touching
the rest of the system.

Out of the box it runs on a **mock feed** and a **paper gateway** — no broker credentials, no real
orders, full end-to-end behaviour.

---

## Quick start

```bash
make env                # conda create -n orca python=3.12
make install            # installs ORCA into that env (editable)
cp .env.example .env    # defaults: mock feed, paper orders, trading disabled
make train              # bootstrap the scoring model on synthetic data
make serve              # JSON API at http://127.0.0.1:8000 (docs at /docs)
```

The `make` targets call the env's interpreter by absolute path, so they work
whether or not you have activated it. Override the env name with
`make test CONDA_ENV=orca-dev`.

Without `make`:

```bash
conda create -n orca python=3.12 -y
conda activate orca
pip install -e ".[dev]"
orca init-db
orca serve
```

Python 3.12 is the floor in practice, not just the `requires-python >= 3.11`
in `pyproject.toml`: numpy 2.5+ requires 3.12, so a 3.11 env silently resolves
to an older numpy. Do **not** install the `.[kite]` extra - nothing imports
`kiteconnect` (`execution/kite.py` uses `httpx` directly) and it hard-pins
`autobahn==19.11.2`, which does not belong in a 3.12 env.

Useful CLI commands:

| Command | What it does |
| --- | --- |
| `orca health` | feed / gateway / model wiring check |
| `orca chain NIFTY` | print the nearest-expiry chain around ATM |
| `orca scan --top 10` | rank candidates right now |
| `orca snapshot` | capture one chain snapshot per underlying (training data) |
| `orca track` | mark every open trade once, apply exit rules |
| `orca train --synthetic` | train the model (synthetic bootstrap) |
| `orca demo` | scan → enter → track → exit on the mock feed |
| `orca serve` / `orca worker` | web app (with jobs) / jobs only |

Tests and linting: `make test`, `make lint`.

---

## Layout

```
orca/
├── config.py            env settings (.env) + YAML universe/strategy config
├── domain.py            core types: OptionQuote, ChainSnapshot, LegSpec, Candidate
├── symbols.py           Upstox instrument_key  <->  Kite tradingsymbol
├── db.py                SQLAlchemy engine/session (SQLite by default)
├── models/              tables: chain_snapshots, option_quotes, underlying_daily,
│                                signals, trades, trade_legs, broker_orders, trade_snapshots
├── feeds/               MARKET DATA IN
│   ├── base.py          MarketDataProvider interface
│   ├── upstox.py        Upstox v2 adapter + daily-token login helper
│   └── mock.py          synthetic chain (Black-Scholes + vol smile) for offline dev
├── execution/           ORDERS OUT
│   ├── base.py          OrderGateway interface, OrderRequest/OrderResult
│   ├── kite.py          Zerodha Kite REST adapter + basket margin
│   └── paper.py         simulated fills (default)
├── analytics/           pure math, no I/O
│   ├── greeks.py        Black-Scholes price/greeks, implied vol, realized vol
│   ├── payoff.py        multi-leg expiry payoff, breakevens, probability of profit
│   ├── margin.py        SPAN+exposure estimate (replace with broker margin API)
│   └── charges.py       brokerage/STT/GST cost model
├── strategies/          chain -> candidate leg-sets (registry + templates)
│   ├── short_strangle.py  iron_condor.py  credit_spread.py
├── ml/
│   ├── features.py      regime + structure features (FEATURE_NAMES is the contract)
│   ├── labels.py        replay forward snapshots -> did it hit target before stop?
│   ├── dataset.py       training frame from captured snapshots (+ synthetic bootstrap)
│   ├── model.py         sklearn pipelines, save/load, scoring
│   ├── train.py         chronological split, AUC/Brier/log-loss
│   └── heuristic.py     transparent fallback score, and the baseline to beat
├── services/            orchestration
│   ├── market_data.py   fetch + persist chains, IV history, regime context
│   ├── scanner.py       candidates -> features -> score -> ranked signals
│   ├── trade_service.py build / enter / exit trades and legs
│   ├── tracker.py       multi-leg marking, greeks, exit rules, portfolio summary
│   └── risk.py          pre-trade guardrails (lots, open trades, day loss)
├── jobs/scheduler.py    APScheduler: snapshot, tracker, scan, daily rollup
├── web/
│   ├── app.py           FastAPI factory (JSON API under /api, no bundled UI)
│   └── api/             health, market, signals, trades, model, auth routers
└── cli.py               typer CLI
config/
├── instruments.yaml     NIFTY + SENSEX: lot size, strike step, exchanges, keys
└── strategies.yaml      which structures to generate and their parameters
```

---

## How the pieces fit

```
Upstox ──► feeds ──► services/market_data ──► chain_snapshots (DB)
                               │                     │
                               ▼                     ▼
                         strategies  ──────►  ml/features ──► ml/model ──► score
                               │                                            │
                               ▼                                            ▼
                          Candidate ─────────► signals (DB) ──────►  /api/signals
                                                     │
                                                     ▼  "Take"
                                        services/trade_service ──► execution ──► Kite
                                                     │
                                                     ▼
                                        services/tracker  (every 15s)
                                        trades / trade_legs / trade_snapshots
```

### Multi-leg tracking

A `Trade` is the structure (one strangle, one condor); `TradeLeg` rows are its legs. Every tracker
tick marks each open leg off the live chain, then aggregates at the **trade** level:

- P&L: `realized + unrealized − charges`, with peak/trough retained (`max_profit_seen`, `max_loss_seen`)
- greeks: net delta / gamma / theta / vega, quantity- and direction-weighted
- rules: target hit, stop hit, breakeven trail, delta breach, spot outside breakevens, expiry-day clock
- actions: `hold` / `adjust` / `exit` — and auto-exit if the trade has `auto_exit` set

Legs can be exited individually (`POST /api/trades/{id}/legs/{leg_id}/exit`), which is how a
strangle gets reduced to a single short leg mid-session. Every tick writes a `trade_snapshots` row,
so the P&L curve and the ML outcome labels both come from the same source.

### The ML layer

| Stage | Where | Notes |
| --- | --- | --- |
| Capture | `jobs.snapshot_job` every 60s | intraday chains with greeks are not available historically — you have to collect them |
| Features | `ml/features.py` | regime (IV rank, IV−RV spread, skew, PCR, trend, time of day) + structure (short delta, net greeks, credit/margin, breakeven distance in sigmas) |
| Label | `ml/labels.py` | replay forward snapshots: target (50% of credit) before stop (2× credit)? |
| Train | `ml/train.py` | `HistGradientBoosting`, chronological split, reports AUC / Brier / log-loss |
| Score | `services/scanner.py` | model probability, or the `ml/heuristic.py` blend until a model exists |

`orca train --synthetic` exists so the whole path is exercisable on day one. Models trained that way
are tagged `synthetic: true` and shown as such in the UI — **do not trade off them**. Collect a few
weeks of real snapshots, then `orca train`, and check the hold-out AUC beats the heuristic baseline.

---

## Staging

A containerised deployment with the broker unreachable by construction
(`paper` gateway, `ORCA_TRADING_ENABLED=false`) and its own database and model
directory, so it cannot touch your dev `data/`.

```bash
make staging-up            # build + run api (loopback:8000) and worker
make staging-seed-model    # copy data/models/default.* into the staging volume
make staging-smoke         # 15 assertions: liveness, safety, wiring, a real scan
make staging-logs          # follow both containers
make staging-down          # stop (the volume, and its data, survives)
```

`deploy/compose.yaml` is the same topology for `docker compose`, if you have the
v2 plugin; the Makefile targets issue plain `docker run` so they work without it.

Two containers from one image, on purpose:

| container | `ORCA_SCHEDULER_ENABLED` | why |
| --- | --- | --- |
| `orca-api` | `false` | restarting the API must not interrupt tracking |
| `orca-worker` | `true` | one owner for the jobs, so they never double up |

`make staging-smoke` is the part worth keeping. It asserts `trading_enabled=false`,
`gateway=paper` and `gateway.live=false` **before** it checks anything works - a
staging box quietly pointed at a live gateway is worse than one that is down. It
also warns when the active model is `synthetic`.

Configuration lives in `deploy/staging.env`. Set `ORCA_FEED_PROVIDER=upstox` plus a
daily token there for the genuinely useful staging mode: real market data, simulated
fills. Note the API is bound to `127.0.0.1` because **it still has no
authentication** - put it behind a VPN or an authenticating proxy before it is
reachable from anywhere else.

### On a VM

```bash
make staging-bootstrap HOST=orca@<vm-ip>   # once: docker, deploy user, swap, firewall
make staging-deploy    HOST=orca@<vm-ip>   # rsync + build + restart + smoke
make staging-tunnel    HOST=orca@<vm-ip>   # reach it at http://127.0.0.1:8000/docs
```

Sizing, from measurement: the image is **785 MB**, the two containers idle at
**~325 MiB** together, and the source transfer is **~550 KB** (the image is built
on the VM, not shipped). 2 vCPU / 4 GB / 20 GB is comfortable. Go to 8 GB only if
you intend to run `orca train` there - loading a month of 5-minute chains costs
roughly 2 GB of RAM.

`deploy/bootstrap.sh` leaves port 8000 **closed** and sets `ufw` to SSH-only,
because the API is protected only by a single-account login (`ORCA_AUTH_USERNAME` /
`ORCA_AUTH_PASSWORD`) over plain HTTP. Reach it through the SSH tunnel above, or a
VPN. `deploy/deploy.sh` syncs with rsync rather than git (this tree is the source
of truth) and excludes `data/` and your dev `.env`, so local credentials and your
local database never reach the VM.

Deployed environments should set `ORCA_LOG_FORMAT=full` (adds the date) or `json`;
the `plain` default prints wall-clock time only, which is useless for
reconstructing a session after the fact.

---

## Developing a UI against this API

The API is bound to loopback on whichever host runs it, so a local UI reaches a
VM deployment through an SSH tunnel:

```bash
make staging-tunnel HOST=orca@<vm-ip>      # VM's :8000 -> your 127.0.0.1:8000
```

Then **proxy `/api` through your dev server** rather than calling the API
cross-origin. The browser only ever talks to its own origin, so CORS stops being
a consideration at all - no preflight, and `ORCA_CORS_ORIGINS` can stay at its
default:

```ts
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      "/api":  { target: process.env.ORCA_API ?? "http://127.0.0.1:8000", changeOrigin: true },
      "/docs": { target: process.env.ORCA_API ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
```

`ORCA_API` then switches the same UI between a local ORCA (`make serve`, mock feed,
fast and always available) and the VM (real Upstox data, but its daily token
expires and the tunnel has to be up).

If you call the API directly instead of proxying, set the origin explicitly - it
is enforced, not advisory:

```bash
# deploy/staging.env  (or .env locally)
ORCA_CORS_ORIGINS=http://localhost:5173
```

The contract is `/openapi.json`; generate a typed client from it instead of
hand-writing fetch calls. `GET /docs` is the browsable version.

---

## Going live (in this order)

1. **Upstox token** — `GET /api/auth/upstox/login` gives the login URL; the redirect comes back with
   `?code=`, and `GET /api/auth/upstox/callback?code=...` exchanges it. Put the token in `.env` as
   `UPSTOX_ACCESS_TOKEN` and set `ORCA_FEED_PROVIDER=upstox`. Upstox tokens expire daily (~03:30 IST).
2. **Kite token** — same shape: `/api/auth/kite/login`, then `/api/auth/kite/callback?request_token=...`
   → `KITE_ACCESS_TOKEN`. Leave `ORCA_EXECUTION_GATEWAY=paper` while you compare paper fills against
   what the market actually did.
3. **Verify symbol mapping** — `orca/symbols.py` formats Kite tradingsymbols, but
   `resolve_from_instruments()` looks them up in Kite's own instrument dump. Use the lookup before
   sending live orders; the formatter is a convenience, not an authority.
4. **Replace the margin estimate** — `analytics/margin.py` is an approximation. `KiteGateway.basket_margin()`
   calls `POST /margins/basket` for the real SPAN+exposure number.
5. **Flip the switch** — `ORCA_EXECUTION_GATEWAY=kite` and `ORCA_TRADING_ENABLED=true`. Until that
   flag is true, `KiteGateway` refuses to send anything (`TradingDisabled`).

Guardrails that are already enforced: `ORCA_MAX_LOTS_PER_TRADE`, `ORCA_MAX_OPEN_TRADES`,
`ORCA_MAX_DAY_LOSS`, plus a check that the structure has either a defined max loss or a margin
estimate (`services/risk.py`).

### Values you must verify yourself

These change by exchange circular and are config/constants on purpose, not facts baked into logic:

- lot sizes, strike steps, weekly expiry weekdays — `config/instruments.yaml`
  (ORCA prefers the expiry list returned by the feed; the weekday rule is only a fallback)
- brokerage / STT / exchange / GST rates — `analytics/charges.py`
- margin percentages — `analytics/margin.py`

---

## API

```
GET  /api/health                        feed, gateway, model, market status
GET  /api/market/underlyings            configured universe
GET  /api/market/expiries?symbol=NIFTY  available expiries
GET  /api/market/chain?symbol=NIFTY     chain around ATM (optionally persist it)
GET  /api/market/context?symbol=NIFTY   regime features (IV rank, RV, trend...)
POST /api/signals/scan                  generate + score + rank candidates
GET  /api/signals                       recent scored candidates
POST /api/trades/from-signal            take a candidate (risk-checked, then placed)
POST /api/trades/manual                 build a trade from explicit legs
POST /api/trades/{id}/enter             send the entry basket for a DRAFT trade
POST /api/trades/{id}/track             mark now
PATCH /api/trades/{id}                  stop / target / auto-exit / trail flags
POST /api/trades/{id}/exit              close every open leg
POST /api/trades/{id}/legs/{leg}/exit   close one leg
GET  /api/trades/{id}/payoff            expiry payoff curve + breakevens
GET  /api/trades/{id}/history           MTM time series
GET  /api/portfolio                     aggregate open risk
POST /api/portfolio/track               tracker tick for every open trade
GET  /api/model  POST /api/model/train  model info / retrain
```

Interactive docs at `/docs`.

---

## Extending

**A new strategy** — add a template and reference it from `config/strategies.yaml`:

```python
# orca/strategies/jade_lizard.py
@template("jade_lizard")
def build(chain, underlying, params, lots=1):
    ...
    return [build_candidate("jade_lizard", "jade_lizard", chain, underlying, legs)]
```

Import it in `orca/strategies/__init__.py` and it shows up in the scanner, the features, the labels
and the training set — no other changes.

**A new underlying** — add a block to `config/instruments.yaml` (BANKNIFTY is there, commented).

**A new feature** — append to `FEATURE_NAMES` in `ml/features.py` and compute it in
`market_context()` or `candidate_features()`, then retrain. Saved models record the feature list
they were trained with.

**The front end** — ORCA ships no UI. Everything is exposed as JSON under `/api` (browse it at
`/docs`), so the UI is a separate project against that contract. Point `ORCA_CORS_ORIGINS` at its
origin; `*` is the default and is only safe on localhost.

**Login** — every `/api` route except `/api/health`, `/api/session/login` and the broker OAuth
callbacks requires a bearer token. Get one with
`POST /api/session/login {"username": "Admin", "password": "passwordpassword"}` and send it as
`Authorization: Bearer <access_token>`. The account comes from `ORCA_AUTH_USERNAME` /
`ORCA_AUTH_PASSWORD` (defaults shown) — **change the password** anywhere beyond localhost.
Tokens last `ORCA_AUTH_TOKEN_TTL_HOURS` (12) and are signed with `ORCA_AUTH_SECRET`, or with a
key derived from the credentials when that is empty.

---

## Status

Implemented and tested end to end on the mock feed: data capture, candidate generation and scoring,
risk checks, order placement (paper), multi-leg tracking with greeks and exit rules, partial-leg and
full exits, payoff/P&L history, model training and inspection, JSON API, CLI, background jobs.

Not done yet (deliberate next steps): Upstox websocket streaming instead of polling, Alembic
migrations, adjustment/roll workflows, backtesting over captured snapshots, **auth on the API**,
per-session reporting. The UI lives in a separate project.

> This is trading software operating on your own money. Everything ships disabled: mock feed, paper
> fills, `ORCA_TRADING_ENABLED=false`. Keep it that way until you have verified each layer against
> your broker yourself. The ML score is a ranking aid, not advice.
