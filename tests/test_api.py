"""Smoke test of the HTTP surface with the mock feed + paper gateway."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orca.web.app import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        resp = c.post(
            "/api/session/login", json={"username": "Admin", "password": "passwordpassword"}
        )
        assert resp.status_code == 200, resp.text
        c.headers["Authorization"] = f"Bearer {resp.json()['access_token']}"
        yield c


@pytest.fixture(scope="module")
def anon():
    with TestClient(create_app()) as c:
        yield c


def test_login_rejects_bad_credentials(anon) -> None:
    for user, pwd in [("Admin", "wrong"), ("admin", "passwordpassword"), ("", "")]:
        resp = anon.post("/api/session/login", json={"username": user, "password": pwd})
        assert resp.status_code == 401


def test_protected_routes_need_token(anon) -> None:
    for path in ["/api/market/underlyings", "/api/portfolio", "/api/model", "/api/session/me"]:
        assert anon.get(path).status_code == 401, path
    bad = {"Authorization": "Bearer not-a-token"}
    assert anon.get("/api/portfolio", headers=bad).status_code == 401
    # Probes and the login itself stay open.
    assert anon.get("/api/health").status_code == 200


def test_token_identifies_user(client) -> None:
    assert client.get("/api/session/me").json() == {"username": "Admin"}


def test_tampered_or_expired_token_rejected() -> None:
    from orca.config import get_settings
    from orca.web.security import issue_token, verify_token

    token, _ = issue_token("Admin")
    assert verify_token(token) == "Admin"
    body, sig = token.split(".")
    assert verify_token(body + "x." + sig) is None
    assert verify_token(token[:-2] + "AA") is None

    settings = get_settings().model_copy(update={"auth_token_ttl_hours": -1})
    expired, _ = issue_token("Admin", settings)
    assert verify_token(expired, settings) is None


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


def test_probability_model(client) -> None:
    body = client.get("/api/model/probability?symbol=NIFTY&strikes=5").json()
    assert body["spot"] > 0 and body["atm_iv"] > 0 and body["sigma"] > 0
    assert [r["sigmas"] for r in body["ranges"]] == [1, 2, 3]
    assert 0 < len(body["strikes"]) <= 11 and body["curve"]
    for s in body["strikes"]:
        assert s["p_below"] + s["p_above"] == pytest.approx(1.0)
        assert 0.0 <= s["p_touch"] <= 1.0
    below = [s["p_below"] for s in body["strikes"]]
    assert below == sorted(below)  # further up the chain, likelier to finish below


def test_broker_callback_served_at_registered_path(anon) -> None:
    # Upstox/Kite redirect to /auth/<broker>/callback (no /api prefix). The route
    # must exist there (422 = missing ?code, not 404) and need no dashboard login.
    assert anon.get("/auth/upstox/callback").status_code == 422
    assert anon.get("/auth/kite/callback").status_code == 422
    assert anon.get("/auth/upstox/login").status_code == 401
