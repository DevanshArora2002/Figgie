"""Charts for a finished tournament. Every figure is drawn from the CSV tables."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from figures import apply_style
from tournament import ENTRANTS, aggregate_games, bootstrap_mean_ci, league_duels

STRATEGY_COLOR = {
    "greedy": "#1d4e89",
    "random": "#c44536",
    "chartist": "#1f7a4d",
    "bottom_feeder": "#6b3fa0",
    "fundamentalist": "#b86e00",
    "fundamentalist2": "#0f6f78",
    "fundamentalist3": "#9b3a4a",
}

STAGE_COLUMNS = [
    ("table_1", "Table 1"),
    ("table_2", "Table 2"),
    ("table_3", "Table 3"),
    ("league", "League"),
]


def write_figures(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    games = _read_games(out_dir / "games.csv")
    standings = _read_csv(out_dir / "standings.csv")
    league = _read_csv(out_dir / "league.csv")
    head_to_head = _read_csv(out_dir / "head_to_head.csv")
    if not league:
        raise SystemExit(f"{out_dir / 'league.csv'} is empty")

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    order = [row["strategy"] for row in sorted(league, key=lambda row: int(row["rank"]))]
    charts = {
        "bracket.png": bracket_figure(standings, league),
        "league_table.png": league_table_figure(league, games, order),
        "head_to_head.png": head_to_head_figure(head_to_head, order),
        "payoff_distributions.png": payoff_distributions_figure(games, order),
        "points_race.png": points_race_figure(games, order),
        "mix_effect.png": mix_effect_figure(games, order),
    }
    written = []
    for name, fig in charts.items():
        path = figures_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def bracket_figure(standings, league) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.4, 8.0))
    present = {}
    for row in standings:
        present.setdefault(row["stage"], {})[row["strategy"]] = row
    champions = {row["strategy"] for row in league if _flag(row.get("champion"))}

    box_w = 2.35
    box_h = 0.72
    gap_x = 3.05
    gap_y = 0.98
    origin_x = 0.35
    origin_y = 0.55
    positions = {}
    for column, (stage, title) in enumerate(STAGE_COLUMNS):
        x = origin_x + column * gap_x
        ax.text(x, origin_y + len(ENTRANTS) * gap_y + 0.08, title, fontsize=12, fontweight="bold", ha="left")
        for index, strategy in enumerate(ENTRANTS):
            row = present.get(stage, {}).get(strategy)
            if row is None:
                continue
            y = origin_y + (len(ENTRANTS) - 1 - index) * gap_y
            positions[(stage, strategy)] = (x, y)
            advanced = _flag(row.get("advanced"))
            color = STRATEGY_COLOR.get(strategy, "#555555")
            face = color if advanced else "#f3f3f3"
            edge = "#111111" if strategy in champions and stage == "league" else color
            width = 2.2 if strategy in champions and stage == "league" else 1.2
            ax.add_patch(FancyBboxPatch(
                (x, y),
                box_w,
                box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=face,
                edgecolor=edge,
                linewidth=width,
            ))
            payoff = float(row["mean_payoff"])
            text = f"{strategy}\n{payoff:+.1f}"
            ax.text(
                x + box_w / 2,
                y + box_h / 2,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if advanced else "#444444",
            )

    stage_names = [stage for stage, _title in STAGE_COLUMNS]
    for (stage, strategy), (x, y) in positions.items():
        start = stage_names.index(stage)
        for later in stage_names[start + 1:]:
            dest = positions.get((later, strategy))
            if dest is None:
                continue
            x2, y2 = dest
            ax.add_patch(FancyArrowPatch(
                (x + box_w + 0.02, y + box_h / 2),
                (x2 - 0.02, y2 + box_h / 2),
                arrowstyle="-|>",
                mutation_scale=9,
                color=STRATEGY_COLOR.get(strategy, "#555555"),
                linewidth=1.15,
                alpha=0.85,
                shrinkA=0,
                shrinkB=0,
            ))
            break

    ax.set_xlim(0, origin_x + 3 * gap_x + box_w + 0.45)
    ax.set_ylim(0, origin_y + len(ENTRANTS) * gap_y + 0.55)
    ax.axis("off")
    ax.set_title("Who reached the top-5 league", loc="left", pad=8)
    fig.legend(
        handles=[
            Patch(facecolor="#1d4e89", edgecolor="#1d4e89", label="advanced"),
            Patch(facecolor="#f3f3f3", edgecolor="#1d4e89", label="eliminated"),
            Patch(facecolor="#0f6f78", edgecolor="#111111", linewidth=2, label="league champion"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    return fig


def league_table_figure(league, games, order) -> plt.Figure:
    apply_style()
    by_name = {row["strategy"]: row for row in league}
    ranked = list(reversed(order))
    y = np.arange(len(ranked))
    points = [float(by_name[name]["points"]) for name in ranked]
    labels = [f"{int(by_name[name]['rank'])}  {name}" for name in ranked]
    colors = [STRATEGY_COLOR[name] for name in ranked]
    edges = ["#111111" if _flag(by_name[name].get("champion")) else color for name, color in zip(ranked, colors)]
    widths = [2.0 if _flag(by_name[name].get("champion")) else 0.0 for name in ranked]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
    ax = axes[0]
    ax.barh(y, points, color=colors, edgecolor=edges, linewidth=widths, zorder=2)
    for index, name in enumerate(ranked):
        row = by_name[name]
        record = f"{int(row['duel_wins'])}-{int(row['duel_draws'])}-{int(row['duel_losses'])}"
        ax.text(float(row["points"]) + 0.6, index, record, va="center", ha="left", fontsize=8, color="#333333", clip_on=False)
    ax.set_yticks(y, labels)
    ax.set_xlabel("league points")
    ax.set_xlim(0, max(points) * 1.38 if points else 1)
    ax.set_title("Points, with wins-draws-losses", loc="left")
    ax.xaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)

    means, lows, highs = _league_payoff_intervals(games, ranked)
    ax = axes[1]
    low_err = np.array(means) - np.array(lows)
    high_err = np.array(highs) - np.array(means)
    ax.barh(y, means, color=colors, edgecolor=edges, linewidth=widths, xerr=[low_err, high_err], capsize=3, ecolor="#333333", zorder=2)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xlabel("mean payoff")
    ax.set_title("Mean payoff and 95% interval", loc="left")
    ax.xaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)
    fig.suptitle("Top-5 league", fontsize=14)
    fig.tight_layout()
    return fig


def head_to_head_figure(records, order) -> plt.Figure:
    apply_style()
    size = len(order)
    index = {name: place for place, name in enumerate(order)}
    diff = np.full((size, size), np.nan)
    share = np.full((size, size), np.nan)
    lookup = {(row["strategy_a"], row["strategy_b"]): row for row in records}
    for row_name in order:
        for column_name in order:
            if row_name == column_name:
                continue
            record = lookup[(row_name, column_name)]
            diff[index[row_name], index[column_name]] = float(record["mean_payoff_diff"])
            share[index[row_name], index[column_name]] = float(record["a_above_share"])
    limit = max(1.0, float(np.nanmax(np.abs(diff))))
    masked = np.ma.array(diff, mask=np.eye(size))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("#f4f4f4")

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    image = ax.imshow(masked, cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(size), order, rotation=30, ha="right")
    ax.set_yticks(range(size), order)
    for row_index in range(size):
        for column_index in range(size):
            if row_index == column_index:
                continue
            strength = abs(diff[row_index, column_index]) / limit
            color = "white" if strength > 0.55 else "#1a1a1a"
            ax.text(
                column_index,
                row_index,
                f"{diff[row_index, column_index]:+.0f}\n{share[row_index, column_index]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    ax.set_title("Head to head. Color and the top number are row minus column.", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="mean payoff difference")
    fig.tight_layout()
    return fig


def payoff_distributions_figure(games, order) -> plt.Figure:
    apply_style()
    league_games = [game for game in aggregate_games(games) if game["stage"] == "league"]
    samples = [[game["payoff"] for game in league_games if game["strategy"] == name] for name in order]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    positions = np.arange(1, len(order) + 1)
    boxes = ax.boxplot(
        samples,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
        zorder=2,
    )
    rng = np.random.default_rng(0)
    for position, name, sample, box in zip(positions, order, samples, boxes["boxes"]):
        color = STRATEGY_COLOR[name]
        box.set_facecolor(color)
        box.set_alpha(0.35)
        box.set_edgecolor(color)
        jitter = rng.uniform(-0.12, 0.12, size=len(sample))
        ax.scatter(np.full(len(sample), position) + jitter, sample, s=14, color=color, alpha=0.55, zorder=3)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(positions, order, rotation=20, ha="right")
    ax.set_ylabel("payoff in one game")
    ax.set_title("How widely each league result spreads", loc="left")
    ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def points_race_figure(games, order) -> plt.Figure:
    apply_style()
    duels = league_duels(games)
    seeds = sorted({int(duel["seed"]) for duel in duels})
    earned = {seed: {name: 0.0 for name in order} for seed in seeds}
    for duel in duels:
        seed = int(duel["seed"])
        diff = duel["payoff_diff"]
        first = duel["strategy_a"]
        second = duel["strategy_b"]
        if diff > 0:
            earned[seed][first] += 1
        elif diff < 0:
            earned[seed][second] += 1
        else:
            earned[seed][first] += 0.5
            earned[seed][second] += 0.5

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    running = {name: 0.0 for name in order}
    series = {name: [] for name in order}
    for seed in seeds:
        for name in order:
            running[name] += earned[seed][name]
            series[name].append(running[name])
    for name in order:
        ax.plot(seeds, series[name], marker="o", linewidth=2, color=STRATEGY_COLOR[name], label=name)
    ax.set_xticks(seeds, [str(seed) for seed in seeds])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("seed")
    ax.set_ylabel("cumulative league points")
    ax.set_title("How the league lead changed as seeds were added", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.yaxis.grid(True, color="#f0f0f0")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def mix_effect_figure(games, order) -> plt.Figure:
    apply_style()
    buckets = {}
    grouped = {}
    for game in aggregate_games(games):
        if game["stage"] != "league":
            continue
        grouped.setdefault((game["lineup"], game["seed"]), {})[game["strategy"]] = game["payoff"]
    for (lineup, _seed), payoffs in grouped.items():
        names = lineup.split("+")
        distinct = list(dict.fromkeys(names))
        if len(distinct) != 2:
            continue
        first, second = distinct
        buckets.setdefault((first, second, names.count(first)), []).append(payoffs[first] - payoffs[second])

    pairs = list(combinations(order, 2))
    seat_counts = (3, 2, 1)
    heights = {seats: [] for seats in seat_counts}
    for first, second in pairs:
        for seats in seat_counts:
            heights[seats].append(_mix_mean(buckets, first, second, seats))

    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    x = np.arange(len(pairs))
    width = 0.24
    shades = {3: "#1d4e89", 2: "#5b8fc7", 1: "#b9d2ea"}
    for offset, seats in zip((-width, 0, width), seat_counts):
        ax.bar(
            x + offset,
            heights[seats],
            width,
            color=shades[seats],
            edgecolor="white",
            linewidth=0.6,
            label=f"A has {seats} seats",
            zorder=2,
        )
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(x, [f"{first}\nvs {second}" for first, second in pairs], fontsize=7.5)
    ax.set_ylabel("mean payoff, A minus B")
    ax.set_title("A is the higher-ranked strategy. The pale bar is A outnumbered 1 to 3.", loc="left")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _mix_mean(buckets, first, second, seats):
    direct = buckets.get((first, second, seats))
    if direct:
        return float(np.mean(direct))
    opposite = buckets.get((second, first, 4 - seats))
    if opposite:
        return -float(np.mean(opposite))
    raise ValueError(f"No {seats}-seat games for {first} against {second}")


def _league_payoff_intervals(games, strategies):
    grouped = {name: [] for name in strategies}
    for game in aggregate_games(games):
        if game["stage"] == "league" and game["strategy"] in grouped:
            grouped[game["strategy"]].append(game["payoff"])
    rng = np.random.default_rng(0)
    means, lows, highs = [], [], []
    for name in strategies:
        mean, low, high = bootstrap_mean_ci(grouped[name], rng)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    return means, lows, highs


def _flag(value) -> bool:
    return value is True or str(value) == "True"


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_games(path: Path) -> list[dict]:
    rows = _read_csv(path)
    for row in rows:
        row["seed"] = int(row["seed"])
        row["cash"] = float(row["cash"])
        row["payoff"] = float(row["payoff"])
        row["rank"] = float(row["rank"])
        row["win_share"] = float(row["win_share"])
    return rows
