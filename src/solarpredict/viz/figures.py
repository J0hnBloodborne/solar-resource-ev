"""Figure generators for the paper (matplotlib, headless).

Each function takes plain arrays/frames and writes a PNG, returning its path, so
the pipeline stays a thin caller. Tiers are colour-coded consistently across
figures (naive/statistical, classical ML, deep learning, foundation model).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

TIER_COLORS = {
    "tier-0": "#9e9e9e",
    "tier-1": "#1f77b4",
    "tier-2": "#2ca02c",
    "tier-3": "#d62728",
}
TIER_LABELS = {
    "tier-0": "naive/statistical",
    "tier-1": "classical ML",
    "tier-2": "deep learning",
    "tier-3": "foundation model",
}


def _save(fig: plt.Figure, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def model_comparison_bar(
    table: pd.DataFrame, *, path: str | Path, metric: str = "skill", title: str = ""
) -> Path:
    """Bar chart of each model's ``metric``, coloured by tier (best first)."""
    ranked = table.sort_values(metric, ascending=(metric != "skill"))
    colors = [TIER_COLORS.get(t, "#333333") for t in ranked["tier"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ranked["model"], ranked[metric], color=colors)
    ylabel = "Forecast skill score" if metric == "skill" else metric.upper()
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(title or f"Hour-ahead GHI — model {metric}")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(
        handles=[Patch(color=c, label=TIER_LABELS[t]) for t, c in TIER_COLORS.items()],
        fontsize=8,
        title="tier",
    )
    return _save(fig, path)


def predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    path: str | Path,
    name: str,
    mask: np.ndarray | None = None,
    sample: int = 4000,
) -> Path:
    """Scatter of predicted vs actual GHI for one model (1:1 line)."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    if mask is not None:
        yt, yp = yt[mask], yp[mask]
    if len(yt) > sample:
        idx = np.linspace(0, len(yt) - 1, sample).astype(int)
        yt, yp = yt[idx], yp[idx]

    lim = float(max(yt.max(), yp.max())) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=4, alpha=0.3, color="#1f77b4")
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual GHI (W/m$^2$)")
    ax.set_ylabel("Predicted GHI (W/m$^2$)")
    ax.set_title(f"Predicted vs actual — {name}")
    return _save(fig, path)


def feature_importance(
    names: list[str], importances: np.ndarray, *, path: str | Path, top: int = 15
) -> Path:
    """Horizontal bar chart of the top feature importances."""
    order = np.argsort(np.asarray(importances))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order], [importances[i] for i in order], color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance — best model")
    return _save(fig, path)


def data_efficiency_lines(
    df: pd.DataFrame, *, path: str | Path, metric: str = "skill"
) -> Path:
    """Line chart of ``metric`` vs training-set size, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("train_years")
        ax.plot(sub["train_years"], sub[metric], marker="o", label=model)
    ax.set_xlabel("Training data (years)")
    ax.set_ylabel("Forecast skill score" if metric == "skill" else metric.upper())
    ax.set_title(
        "Data efficiency — accuracy vs training-set size (Karachi, fixed test)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def city_ranking_bar(
    summary: pd.DataFrame,
    *,
    path: str | Path,
    value: str = "annual_kwh_m2",
    highlight: tuple[str, ...] = (),
) -> Path:
    """Bar chart of cities by solar yield; highlighted cities are coloured red."""
    ranked = summary.sort_values(value, ascending=False)
    chosen = set(highlight)
    colors = ["#d62728" if c in chosen else "#1f77b4" for c in ranked["city"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ranked["city"], ranked[value], color=colors)
    ax.set_ylabel("Annual GHI yield (kWh/m$^2$/yr)")
    ax.set_xlabel("City")
    ax.set_title("Solar resource by city (5-year mean)")
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, ranked[value], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return _save(fig, path)


def city_ghi_map(
    summary: pd.DataFrame, *, path: str | Path, highlight: tuple[str, ...] = ()
) -> Path:
    """Geographic scatter of cities (lon/lat) coloured by annual GHI yield."""
    chosen = set(highlight)
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        summary["longitude"],
        summary["latitude"],
        c=summary["annual_kwh_m2"],
        s=220,
        cmap="YlOrRd",
        edgecolor="black",
        zorder=3,
    )
    for _, row in summary.iterrows():
        weight = "bold" if row["city"] in chosen else "normal"
        ax.annotate(
            f"{row['city']}\n{row['annual_kwh_m2']:.0f}",
            (row["longitude"], row["latitude"]),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=8,
            fontweight=weight,
        )
    fig.colorbar(scatter, ax=ax, label="Annual GHI yield (kWh/m$^2$/yr)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Solar resource across Pakistani cities")
    ax.grid(alpha=0.2)
    return _save(fig, path)


def seasonal_ghi_lines(
    monthly: pd.DataFrame, *, path: str | Path, cities: tuple[str, ...] | None = None
) -> Path:
    """Mean daytime GHI by month, one line per city."""
    names = cities if cities is not None else tuple(monthly["city"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for city in names:
        sub = monthly[monthly["city"] == city].sort_values("month")
        ax.plot(sub["month"], sub["ghi_day"], marker="o", label=city)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean daytime GHI (W/m$^2$)")
    ax.set_title("Seasonal GHI by city")
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def _yellow_red(values: np.ndarray) -> np.ndarray:
    span = float(values.max() - values.min())
    frac = 0.35 + 0.55 * (
        (values - values.min()) / span if span > 0 else np.full_like(values, 0.5)
    )
    return plt.get_cmap("YlOrRd")(frac)


def site_suitability_bar(
    df: pd.DataFrame, *, path: str | Path, city: str = "Karachi"
) -> Path:
    """Bar chart of within-city sites by suitability score (best first)."""
    ranked = df.sort_values("suitability", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ranked["site"],
        ranked["suitability"],
        color=_yellow_red(ranked["suitability"].to_numpy()),
    )
    ax.set_ylabel("Suitability score (0-1)")
    ax.set_xlabel("Site")
    ax.set_title(f"Solar-EV site suitability — {city}")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def ev_locations_map(
    df: pd.DataFrame, *, path: str | Path, city: str = "Karachi", top: int = 3
) -> Path:
    """Map of sites coloured by suitability; top-N starred as recommended."""
    ranked = df.sort_values("suitability", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(
        ranked["longitude"],
        ranked["latitude"],
        c=ranked["suitability"],
        s=260,
        cmap="YlOrRd",
        edgecolor="black",
        zorder=3,
    )
    for i, row in ranked.iterrows():
        recommended = i < top
        ax.annotate(
            f"{'★ ' if recommended else ''}{row['site']}",
            (row["longitude"], row["latitude"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9,
            fontweight="bold" if recommended else "normal",
        )
    fig.colorbar(scatter, ax=ax, label="Suitability score")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Recommended solar-EV charging sites — {city} (top {top} starred)")
    ax.grid(alpha=0.2)
    return _save(fig, path)


def ghi_heatmap_hour_month(
    df: pd.DataFrame,
    *,
    path: str | Path,
    ghi_col: str = "y",
    time_col: str = "ds",
    tz: str = "Asia/Karachi",
    title: str = "GHI heatmap — local hour of day vs month (Karachi)",
) -> Path:
    """2-D heatmap of mean GHI by local hour-of-day (y) and month (x)."""
    ts = pd.DatetimeIndex(df[time_col])
    ts = (ts.tz_localize("UTC") if ts.tz is None else ts).tz_convert(tz)
    grid = (
        pd.DataFrame(
            {"hour": ts.hour, "month": ts.month, "ghi": df[ghi_col].to_numpy(float)}
        )
        .pivot_table(index="hour", columns="month", values="ghi", aggfunc="mean")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        grid.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=(0.5, 12.5, -0.5, 23.5),
    )
    fig.colorbar(image, ax=ax, label="Mean GHI (W/m$^2$)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day (PKT)")
    ax.set_title(title)
    ax.set_xticks(range(1, 13))
    return _save(fig, path)


def daily_ghi_profile(
    df: pd.DataFrame,
    *,
    path: str | Path,
    ghi_col: str = "y",
    time_col: str = "ds",
    tz: str = "Asia/Karachi",
    title: str = "Mean daily GHI and PV potential (Karachi)",
) -> Path:
    """Average GHI by local hour of day, with a PV-power axis (kW per kW-peak)."""
    ts = pd.DatetimeIndex(df[time_col])
    ts = (ts.tz_localize("UTC") if ts.tz is None else ts).tz_convert(tz)
    profile = pd.Series(df[ghi_col].to_numpy(float)).groupby(ts.hour).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(profile.index, profile.to_numpy(), marker="o", color="#d62728")
    ax.fill_between(profile.index, profile.to_numpy(), alpha=0.2, color="#d62728")
    ax.set_xlabel("Hour of day (PKT)")
    ax.set_ylabel("Mean GHI (W/m$^2$)")
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)

    low, high = ax.get_ylim()
    power = ax.twinx()
    power.set_ylim(low / 1000.0, high / 1000.0)  # power ~ GHI/1000 for a 1 kWp panel
    power.set_ylabel("PV power (kW per kW-peak)")
    return _save(fig, path)


def grid_ghi_heatmap(
    grid: pd.DataFrame, *, path: str | Path, city: str = "Karachi"
) -> Path:
    """2-D heatmap of mean daytime GHI across a city's ~7 km sampling grid."""
    pivot = grid.pivot_table(
        index="latitude", columns="longitude", values="mean_ghi_day"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.pcolormesh(
        pivot.columns, pivot.index, pivot.to_numpy(), cmap="YlOrRd", shading="auto"
    )
    fig.colorbar(mesh, ax=ax, label="Mean daytime GHI (W/m$^2$)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Intra-city GHI gradient — {city} (~7 km grid)")
    return _save(fig, path)
