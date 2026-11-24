"""Smoke test of the HTTP surface with the mock feed + paper gateway."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orca.web.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


def test_health(client) -> None:
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["feed"]["provider"] == "mock"
    assert body["trading_enabled"] is False
    assert "NIFTY" in body["universe"]


def test_underlyings_and_chain(client) -> None:
    symbols = [u["symbol"] for u in client.get("/api/market/underlyings").json()]
    assert {"NIFTY", "SENSEX"} <= set(symbols)

    chain = client.get("/api/market/chain?symbol=NIFTY&strikes=5").json()
    assert chain["spot"] > 0
    assert chain["options"]
    assert {o["option_type"] for o in chain["options"]} == {"CE", "PE"}


def test_scan_then_take_then_exit(client) -> None:
    candidates = client.post(
        "/api/signals/scan", json={"symbols": ["NIFTY"], "lots": 1, "limit": 5}
    ).json()
    assert candidates
    signal_id = candidates[0]["signal_id"]
    assert signal_id

    trade = client.post(
        "/api/trades/from-signal", json={"signal_id": signal_id, "lots": 1, "place": True}
    ).json()
    assert trade["status"] == "OPEN"
    assert trade["legs"]

    tracked = client.post(f"/api/trades/{trade['id']}/track").json()
    assert tracked["last_tracked_at"]

    payoff = client.get(f"/api/trades/{trade['id']}/payoff").json()
    assert payoff["curve"]

    closed = client.post(f"/api/trades/{trade['id']}/exit?reason=test").json()
    assert closed["status"] == "CLOSED"


def test_model_train_synthetic(client) -> None:
    body = client.post(
        "/api/model/train", json={"name": "test-api", "synthetic": True, "rows": 800}
    ).json()
    assert body["metrics"]["n_train"] > 0
