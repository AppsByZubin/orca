"""ORCA - Options Risk & Candidate Analyzer.

Layers (each one is replaceable in isolation):

    feeds/        market data in   (Upstox)
    execution/    orders out       (Zerodha Kite / paper)
    analytics/    pricing, greeks, payoff, margin  - pure functions
    strategies/   turn a chain into candidate leg-sets
    ml/           features, labels, training, scoring
    services/     orchestration: snapshot, scan, track
    web/          FastAPI JSON API (no bundled UI)
"""

__version__ = "0.1.0"
