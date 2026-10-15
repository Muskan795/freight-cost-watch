"""Core cost logic: clean data, weekly cost per tonne-km, baselines, flags."""
import pandas as pd

import config as cfg

SHIPMENT_COLUMNS = [
    "shipment_id", "origin", "destination", "route_type",
    "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date",
]
NUMERIC = ["quantity_tonnes", "distance_km", "freight_cost_inr"]


def load_shipments(path):
    """Read shipments, drop invalid rows (and count them) instead of crashing."""
    df = pd.read_csv(path)
    missing = set(SHIPMENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing shipment columns: {sorted(missing)}")

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")
    for col in ["origin", "destination", "route_type"]:
        df[col] = df[col].astype("string").str.strip()

    invalid = (
        df[SHIPMENT_COLUMNS].isnull().any(axis=1)
        | (df["quantity_tonnes"] <= 0)
        | (df["distance_km"] <= 0)
        | (df["freight_cost_inr"] < 0)
    )
    report = {"rows_read": len(df), "dropped_invalid": int(invalid.sum())}
    df = df[~invalid]

    dupes = df.duplicated("shipment_id", keep="first")
    report["dropped_duplicate_ids"] = int(dupes.sum())
    df = df[~dupes]
    report["rows_used"] = len(df)
    return df.reset_index(drop=True), report


def build_weekly(shipments):
    """One row per route per Monday-Sunday week.

    cost_per_tonne_km = total cost / total tonne-km (pooled, not an average of ratios).
    """
    df = shipments.copy()
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    df["week_of"] = (
        df["shipment_date"].dt.normalize()
        - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    )
    df["route"] = df["origin"] + "-" + df["destination"]

    weekly = (
        df.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(total_cost=("freight_cost_inr", "sum"), total_tonne_km=("tonne_km", "sum"))
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    weekly = weekly.sort_values(["route", "week_of"]).reset_index(drop=True)

    if weekly.duplicated(["route", "week_of"]).any():
        raise ValueError("A route appears with more than one route_type in the same week.")
    return weekly


def _pct_change(value, baseline):
    pct = (value / baseline - 1) * 100
    return pct.where(baseline > 0)          # avoid inf when baseline is 0


def add_baselines(weekly):
    """Add the two comparison baselines exactly as defined in the brief."""
    w = weekly.copy()

    # vs own history: mean of up to HISTORY_WEEKS previous OBSERVED weeks of this route
    # (shift(1) excludes the current week -> no look-ahead).
    prior = w.groupby("route")["cost_per_tonne_km"].transform(
        lambda s: s.shift(1).rolling(cfg.HISTORY_WEEKS, min_periods=1).mean()
    )
    prior_n = w.groupby("route")["cost_per_tonne_km"].transform(
        lambda s: s.shift(1).rolling(cfg.HISTORY_WEEKS, min_periods=1).count()
    )
    w["own_history_avg"] = prior
    w["own_history_weeks"] = prior_n.fillna(0).astype(int)
    w["vs_own_history"] = _pct_change(w["cost_per_tonne_km"], w["own_history_avg"])

    # vs similar routes: same route_type, same week, excluding the route itself.
    grp = w.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    total, count = grp.transform("sum"), grp.transform("count")
    w["peer_avg"] = ((total - w["cost_per_tonne_km"]) / (count - 1)).where(count > 1)
    w["vs_similar_routes"] = _pct_change(w["cost_per_tonne_km"], w["peer_avg"])
    return w


def add_flags(weekly):
    """Flag rows at/above the threshold; also build the numeric part of the reason."""
    w = weekly.copy()
    t = cfg.ANOMALY_THRESHOLD_PCT
    own_hit = w["vs_own_history"] >= t
    peer_hit = w["vs_similar_routes"] >= t
    flagged = own_hit | peer_hit
    if cfg.REQUIRE_RISE_VS_OWN_HISTORY:
        flagged &= w["vs_own_history"].fillna(0) > 0
    w["flagged"] = flagged.fillna(False).astype(bool)

    def describe(row):
        parts = []
        if row["vs_own_history"] >= t:
            parts.append(f"{row['vs_own_history']:.1f}% above its trailing "
                         f"{row['own_history_weeks']}-week average")
        if row["vs_similar_routes"] >= t:
            parts.append(f"{row['vs_similar_routes']:.1f}% above similar "
                         f"{row['route_type']} routes that week")
        return "Cost per tonne-km is " + " and ".join(parts)

    w["anomaly_text"] = [describe(r) if r["flagged"] else "" for _, r in w.iterrows()]
    return w
