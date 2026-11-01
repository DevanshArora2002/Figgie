"""Play the Figgie strategy bracket and write the result tables."""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from game_engine import CLASS_REGISTRY, GameEngine
from utils import (
    STARTING_CASH,
    TOURNAMENT_ROUND_TIME,
    TOURNAMENT_SEEDS,
    TOURNAMENT_TICK,
    TOURNAMENT_TOP5_SEEDS,
)

ENTRANTS = [
    "greedy",
    "random",
    "chartist",
    "bottom_feeder",
    "fundamentalist",
    "fundamentalist2",
    "fundamentalist3",
]

STAGE_ORDER = ["table_1", "table_2", "table_3", "league"]
# Each seed: 4 four-way tables x 3 opponents, plus 4 opponents x 3 mixes.
DUELS_PER_SEED = 24

GAMES_FIELDS = [
    "stage",
    "table",
    "lineup",
    "seed",
    "strategy",
    "cash",
    "payoff",
    "rank",
    "win_share",
]
STANDINGS_FIELDS = [
    "stage",
    "table",
    "strategy",
    "mean_payoff",
    "std_error",
    "win_rate",
    "mean_rank",
    "advanced",
]
SUMMARY_FIELDS = [
    "strategy",
    "games_played",
    "mean_payoff",
    "payoff_ci_low",
    "payoff_ci_high",
    "win_rate",
    "mean_rank",
    "stage_reached",
]
HEAD_TO_HEAD_FIELDS = [
    "strategy_a",
    "strategy_b",
    "games",
    "mean_payoff_diff",
    "a_above_share",
]
LEAGUE_FIELDS = [
    "rank",
    "strategy",
    "games",
    "points",
    "duels",
    "duel_wins",
    "duel_draws",
    "duel_losses",
    "mean_payoff",
    "std_error",
    "win_rate",
    "mean_rank",
    "champion",
]


def play_one(lineup, seed, round_time=None, tick=None):
    """Play one game and return one row per seat."""
    import player as player_module

    if round_time is None:
        round_time = TOURNAMENT_ROUND_TIME
    if tick is None:
        tick = TOURNAMENT_TICK
    if len(lineup) != 4:
        raise ValueError(f"A table seats 4 strategies, got {lineup}")

    os.environ["FIGGIE_LOG_LEVEL"] = "WARNING"
    for name in dir(player_module):
        if name.endswith("_TICK"):
            setattr(player_module, name, tick)

    engine = GameEngine(
        *lineup,
        seed=seed,
        round_time=round_time,
        logging_file=None,
        log_file="",
    )
    asyncio.run(engine.play_game())
    return _seat_rows(engine.results())


def _seat_rows(results):
    cashes = [result["cash"] for result in results]
    ranks, win_shares = _ranks_and_win_shares(cashes)
    rows = []
    for result, rank, win_share in zip(results, ranks, win_shares):
        rows.append({
            "strategy": result["strategy"],
            "cash": result["cash"],
            "payoff": result["cash"] - STARTING_CASH,
            "rank": rank,
            "win_share": win_share,
        })
    return rows


def _ranks_and_win_shares(cashes):
    """Average rank for cash ties, and an even split of the win among the top cash."""
    order = sorted(range(len(cashes)), key=lambda index: cashes[index], reverse=True)
    ranks = [0.0] * len(cashes)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and cashes[order[end]] == cashes[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    best = max(cashes)
    winners = sum(1 for cash in cashes if cash == best)
    share = 1.0 / winners
    win_shares = [share if cash == best else 0.0 for cash in cashes]
    return ranks, win_shares


def play_table(stage, lineups, seeds, round_time=None, tick=None, workers=None, table=None):
    """Play every line-up and seed in parallel. Return seat rows and standings."""
    if round_time is None:
        round_time = TOURNAMENT_ROUND_TIME
    if tick is None:
        tick = TOURNAMENT_TICK
    if workers is None:
        workers = os.cpu_count() or 1
    if table is None:
        table = stage

    lineups = _as_lineups(lineups)
    jobs = [(tuple(lineup), seed) for lineup in lineups for seed in seeds]
    print(
        f"Playing {stage}: {len(lineups)} line-up(s) x {len(seeds)} seed(s) on {workers} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        seat_groups = pool.map(
            play_one,
            [lineup for lineup, _seed in jobs],
            [seed for _lineup, seed in jobs],
            [round_time] * len(jobs),
            [tick] * len(jobs),
        )
        rows = []
        for (lineup, seed), seat_rows in zip(jobs, seat_groups):
            lineup_label = "+".join(lineup)
            for seat in seat_rows:
                rows.append({
                    "stage": stage,
                    "table": table,
                    "lineup": lineup_label,
                    "seed": seed,
                    **seat,
                })
    return rows, compute_standings(rows)


def _as_lineups(lineups):
    if not lineups:
        raise ValueError("A table needs at least one line-up")
    if isinstance(lineups[0], str):
        return [tuple(lineups)]
    return [tuple(lineup) for lineup in lineups]


def aggregate_games(rows):
    """One observation per strategy per game: mean payoff, total win share, mean rank."""
    grouped = {}
    for row in rows:
        key = (row["stage"], row["lineup"], row["seed"], row["strategy"])
        grouped.setdefault(key, []).append(row)
    games = []
    for (stage, lineup, seed, strategy), seats in grouped.items():
        games.append({
            "stage": stage,
            "lineup": lineup,
            "seed": seed,
            "strategy": strategy,
            "payoff": float(np.mean([float(seat["payoff"]) for seat in seats])),
            "win_share": float(np.sum([float(seat["win_share"]) for seat in seats])),
            "rank": float(np.mean([float(seat["rank"]) for seat in seats])),
        })
    return games


def compute_standings(rows):
    grouped = {}
    for game in aggregate_games(rows):
        grouped.setdefault(game["strategy"], []).append(game)
    standings = []
    for strategy, games in grouped.items():
        payoffs = np.array([game["payoff"] for game in games], dtype=float)
        count = len(payoffs)
        if count > 1:
            std_error = float(payoffs.std(ddof=1) / np.sqrt(count))
        else:
            std_error = 0.0
        standings.append({
            "strategy": strategy,
            "mean_payoff": float(payoffs.mean()),
            "std_error": std_error,
            "win_rate": float(np.mean([game["win_share"] for game in games])),
            "mean_rank": float(np.mean([game["rank"] for game in games])),
        })
    standings.sort(key=lambda row: (-row["mean_payoff"], -row["win_rate"], row["strategy"]))
    return standings


def _mark_advancement(standings, advancing):
    advanced = []
    staying = []
    for index, row in enumerate(standings):
        row["advanced"] = index < advancing
        if row["advanced"]:
            advanced.append(row["strategy"])
        else:
            staying.append(row["strategy"])
    return advanced, staying


def entry_order(seeds, order=None):
    if order:
        names = [name.lower() for name in order]
        if sorted(names) != sorted(ENTRANTS):
            raise SystemExit(f"--order must list each entrant once: {', '.join(ENTRANTS)}")
        return names
    names = ENTRANTS[:]
    random.Random(seeds[0]).shuffle(names)
    return names


def league_lineups(top5):
    """Five four-way tables, plus 3v1, 2v2, and 1v3 for every pair."""
    if len(top5) != 5 or len(set(top5)) != 5:
        raise ValueError(f"The league needs 5 strategies, got {top5}")
    lineups = [tuple(group) for group in combinations(top5, 4)]
    for first, second in combinations(top5, 2):
        lineups.append((first, first, first, second))
        lineups.append((first, first, second, second))
        lineups.append((first, second, second, second))
    return lineups


def league_duels(rows):
    """One comparison per pair of strategies seated in a league game."""
    games = {}
    for game in aggregate_games(rows):
        if game["stage"] != "league":
            continue
        games.setdefault((game["lineup"], game["seed"]), []).append(game)
    duels = []
    for (lineup, seed), seats in games.items():
        payoffs = {seat["strategy"]: seat["payoff"] for seat in seats}
        for first, second in combinations(sorted(payoffs), 2):
            duels.append({
                "lineup": lineup,
                "seed": seed,
                "strategy_a": first,
                "strategy_b": second,
                "payoff_diff": payoffs[first] - payoffs[second],
            })
    return duels


def compute_league(rows):
    """Rank the top 5 by duel points, then payoff, win rate, and name."""
    league_rows = [row for row in rows if row["stage"] == "league"]
    grouped = {}
    for game in aggregate_games(league_rows):
        grouped.setdefault(game["strategy"], []).append(game)
    totals = {
        strategy: {"points": 0.0, "duels": 0, "wins": 0, "draws": 0, "losses": 0}
        for strategy in grouped
    }
    for duel in league_duels(league_rows):
        first = duel["strategy_a"]
        second = duel["strategy_b"]
        diff = duel["payoff_diff"]
        for strategy in (first, second):
            totals[strategy]["duels"] += 1
        if diff > 0:
            totals[first]["points"] += 1
            totals[first]["wins"] += 1
            totals[second]["losses"] += 1
        elif diff < 0:
            totals[second]["points"] += 1
            totals[second]["wins"] += 1
            totals[first]["losses"] += 1
        else:
            totals[first]["points"] += 0.5
            totals[second]["points"] += 0.5
            totals[first]["draws"] += 1
            totals[second]["draws"] += 1

    table = []
    for strategy, games in grouped.items():
        payoffs = np.array([game["payoff"] for game in games], dtype=float)
        count = len(payoffs)
        if count > 1:
            std_error = float(payoffs.std(ddof=1) / np.sqrt(count))
        else:
            std_error = 0.0
        record = totals[strategy]
        table.append({
            "strategy": strategy,
            "games": count,
            "points": record["points"],
            "duels": record["duels"],
            "duel_wins": record["wins"],
            "duel_draws": record["draws"],
            "duel_losses": record["losses"],
            "mean_payoff": float(payoffs.mean()),
            "std_error": std_error,
            "win_rate": float(np.mean([game["win_share"] for game in games])),
            "mean_rank": float(np.mean([game["rank"] for game in games])),
        })
    table.sort(key=lambda row: (-row["points"], -row["mean_payoff"], -row["win_rate"], row["strategy"]))
    for index, row in enumerate(table, start=1):
        row["rank"] = index
        row["champion"] = index == 1
    return table


def run_bracket(seeds, top5_seeds, round_time, tick, workers, order=None):
    """Tables 1-3 qualify five strategies. The league then ranks those five."""
    names = entry_order(seeds, order)
    print("Entry order: " + ", ".join(names), flush=True)
    game_rows = []
    standing_rows = []

    def play(stage, table, lineups, stage_seeds, advancing):
        rows, standings = play_table(
            stage,
            lineups,
            stage_seeds,
            round_time=round_time,
            tick=tick,
            workers=workers,
            table=table,
        )
        advanced, rest = _mark_advancement(standings, advancing)
        for row in standings:
            standing_rows.append({"stage": stage, "table": table, **row})
        game_rows.extend(rows)
        _print_standings(stage, standings)
        return advanced, rest

    advanced, rest = play("table_1", 1, names[0:4], seeds, 2)
    qualified = list(advanced)
    print(f"Advanced: {', '.join(advanced)}", flush=True)
    print(f"To table 2: {', '.join(rest)}", flush=True)

    advanced, rest = play("table_2", 2, rest + names[4:6], seeds, 1)
    qualified.extend(advanced)
    print(f"Advanced: {', '.join(advanced)}", flush=True)
    print(f"To table 3: {', '.join(rest)}", flush=True)

    advanced, eliminated = play("table_3", 3, rest + names[6:7], seeds, 2)
    qualified.extend(advanced)
    print(f"Advanced: {', '.join(advanced)}", flush=True)
    print(f"Eliminated: {', '.join(eliminated)}", flush=True)
    print(f"Qualified ({len(qualified)}): {', '.join(qualified)}", flush=True)
    if len(qualified) != 5:
        raise RuntimeError(f"Expected 5 qualified strategies, got {qualified}")

    lineups = league_lineups(qualified)
    if len(lineups) != 35:
        raise RuntimeError(f"Expected 35 league line-ups, got {len(lineups)}")
    rows, standings = play_table(
        "league",
        lineups,
        top5_seeds,
        round_time=round_time,
        tick=tick,
        workers=workers,
        table="league",
    )
    game_rows.extend(rows)
    league = compute_league(rows)
    expected_duels = DUELS_PER_SEED * len(top5_seeds)
    for row in league:
        if row["duels"] != expected_duels:
            raise RuntimeError(
                f"{row['strategy']} fought {row['duels']} duels, expected {expected_duels}"
            )
    by_strategy = {row["strategy"]: row for row in standings}
    for entry in league:
        standing = by_strategy[entry["strategy"]]
        standing["advanced"] = True
        standing_rows.append({"stage": "league", "table": "league", **standing})
    print_league(league)
    print(
        f"Stage sizes: entrants={len(names)} qualified={len(qualified)} league={len(league)}",
        flush=True,
    )
    return {
        "order": names,
        "qualified": qualified,
        "top5": [row["strategy"] for row in league],
        "rows": game_rows,
        "standings": standing_rows,
        "league": league,
    }


def bootstrap_mean_ci(values, rng, n_boot=10_000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return mean, float(low), float(high)


def build_summary(rows):
    grouped = {}
    stages = {}
    for game in aggregate_games(rows):
        grouped.setdefault(game["strategy"], []).append(game)
        stages.setdefault(game["strategy"], set()).add(game["stage"])
    rng = np.random.default_rng(0)
    summary = []
    for strategy in sorted(grouped):
        games = grouped[strategy]
        payoffs = [game["payoff"] for game in games]
        mean, low, high = bootstrap_mean_ci(payoffs, rng)
        reached = max(stages[strategy], key=STAGE_ORDER.index)
        summary.append({
            "strategy": strategy,
            "games_played": len(games),
            "mean_payoff": mean,
            "payoff_ci_low": low,
            "payoff_ci_high": high,
            "win_rate": float(np.mean([game["win_share"] for game in games])),
            "mean_rank": float(np.mean([game["rank"] for game in games])),
            "stage_reached": reached,
        })
    summary.sort(key=lambda row: (-row["mean_payoff"], -row["win_rate"], row["strategy"]))
    return summary


def build_head_to_head(rows, top5):
    games = {}
    for row in rows:
        if row["stage"] != "league":
            continue
        games.setdefault((row["lineup"], row["seed"]), []).append(row)
    records = []
    for first, second in combinations(top5, 2):
        diffs = []
        for seats in games.values():
            present = {seat["strategy"] for seat in seats}
            if first not in present or second not in present:
                continue
            payoff_first = float(np.mean([float(seat["payoff"]) for seat in seats if seat["strategy"] == first]))
            payoff_second = float(np.mean([float(seat["payoff"]) for seat in seats if seat["strategy"] == second]))
            diffs.append(payoff_first - payoff_second)
        if not diffs:
            continue
        count = len(diffs)
        mean_diff = float(np.mean(diffs))
        first_above = sum(diff > 0 for diff in diffs) / count
        second_above = sum(diff < 0 for diff in diffs) / count
        records.append({
            "strategy_a": first,
            "strategy_b": second,
            "games": count,
            "mean_payoff_diff": mean_diff,
            "a_above_share": first_above,
        })
        records.append({
            "strategy_a": second,
            "strategy_b": first,
            "games": count,
            "mean_payoff_diff": -mean_diff,
            "a_above_share": second_above,
        })
    return records


def write_results(out_dir, rows, standings, summary, head_to_head, league):
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "games.csv", GAMES_FIELDS, rows)
    _write_csv(out_dir / "standings.csv", STANDINGS_FIELDS, standings)
    _write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary)
    _write_csv(out_dir / "head_to_head.csv", HEAD_TO_HEAD_FIELDS, head_to_head)
    _write_csv(out_dir / "league.csv", LEAGUE_FIELDS, league)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_standings(stage, standings):
    print(f"\n{stage}", flush=True)
    print(
        f"{'strategy':<18}{'payoff':>10}{'se':>10}{'win_rate':>10}{'rank':>10}{'advanced':>10}",
        flush=True,
    )
    for row in standings:
        advanced = "yes" if row.get("advanced") else "no"
        print(
            f"{row['strategy']:<18}{row['mean_payoff']:>10.2f}{row['std_error']:>10.2f}"
            f"{row['win_rate']:>10.2f}{row['mean_rank']:>10.2f}{advanced:>10}",
            flush=True,
        )


def print_league(league):
    print("\nTop-5 league", flush=True)
    print(
        f"{'rank':>4}  {'strategy':<18}{'points':>8}{'W':>6}{'D':>6}{'L':>6}{'payoff':>10}{'win_rate':>10}",
        flush=True,
    )
    champion = None
    for row in league:
        if row["champion"]:
            champion = row["strategy"]
        print(
            f"{row['rank']:>4}  {row['strategy']:<18}{row['points']:>8.1f}"
            f"{row['duel_wins']:>6}{row['duel_draws']:>6}{row['duel_losses']:>6}"
            f"{row['mean_payoff']:>10.2f}{row['win_rate']:>10.2f}",
            flush=True,
        )
    print(f"Champion: {champion}", flush=True)


def print_leaderboard(summary):
    print("\nOverall leaderboard", flush=True)
    print(
        f"{'strategy':<18}{'games':>7}{'payoff':>10}{'95% CI':>22}{'win_rate':>10}{'rank':>8}  stage",
        flush=True,
    )
    for row in summary:
        interval = f"[{row['payoff_ci_low']:.2f}, {row['payoff_ci_high']:.2f}]"
        print(
            f"{row['strategy']:<18}{row['games_played']:>7}{row['mean_payoff']:>10.2f}"
            f"{interval:>22}{row['win_rate']:>10.2f}{row['mean_rank']:>8.2f}  {row['stage_reached']}",
            flush=True,
        )


def print_head_to_head(top5, records):
    lookup = {(row["strategy_a"], row["strategy_b"]): row for row in records}
    print("\nHead to head: mean payoff of row minus column", flush=True)
    _print_matrix(top5, lookup, "mean_payoff_diff")
    print("\nHead to head: share of games where row finishes above column", flush=True)
    _print_matrix(top5, lookup, "a_above_share")


def _print_matrix(names, lookup, field):
    label_width = max(18, max(len(name) for name in names))
    column_width = label_width
    header = f"{'':<{label_width}}" + "".join(f"{name:>{column_width}}" for name in names)
    print(header, flush=True)
    for row_name in names:
        cells = []
        for column_name in names:
            if row_name == column_name:
                cells.append(f"{'-':>{column_width}}")
            else:
                value = lookup[(row_name, column_name)][field]
                cells.append(f"{value:>{column_width}.2f}")
        print(f"{row_name:<{label_width}}" + "".join(cells), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the Figgie strategy tournament")
    parser.add_argument("--seeds", type=int, nargs="+", default=TOURNAMENT_SEEDS)
    parser.add_argument("--top5-seeds", type=int, nargs="+", default=TOURNAMENT_TOP5_SEEDS)
    parser.add_argument("--round-time", type=float, default=TOURNAMENT_ROUND_TIME)
    parser.add_argument("--tick", type=float, default=TOURNAMENT_TICK)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--order", nargs="+", default=None, help="Fix the 7-strategy entry order")
    parser.add_argument("--plot-only", default=None, help="Rebuild charts from a results folder")
    args = parser.parse_args(argv)
    if args.plot_only:
        return args
    if len(args.seeds) < 2:
        parser.error("provide at least 2 --seeds")
    if len(args.top5_seeds) < 1:
        parser.error("provide at least 1 --top5-seeds")
    missing = [name for name in ENTRANTS if name not in CLASS_REGISTRY]
    if missing:
        parser.error(f"missing from the class registry: {', '.join(missing)}")
    return args


def main(argv=None):
    os.environ["FIGGIE_LOG_LEVEL"] = "WARNING"
    args = parse_args(argv)
    if args.plot_only:
        from tournament_figures import write_figures

        out_dir = Path(args.plot_only).expanduser().resolve()
        if not out_dir.is_dir():
            raise SystemExit(f"No results folder at {out_dir}")
        figures = write_figures(out_dir)
        print(f"Wrote {len(figures)} charts to {out_dir / 'figures'}", flush=True)
        return out_dir

    from tournament_figures import write_figures

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "results" / stamp
    print(
        f"Round time {args.round_time}s | tick {args.tick}s | "
        f"seeds {args.seeds} | top-5 seeds {args.top5_seeds}",
        flush=True,
    )
    bracket = run_bracket(
        seeds=args.seeds,
        top5_seeds=args.top5_seeds,
        round_time=args.round_time,
        tick=args.tick,
        workers=args.workers,
        order=args.order,
    )
    summary = build_summary(bracket["rows"])
    head_to_head = build_head_to_head(bracket["rows"], bracket["top5"])
    write_results(out_dir, bracket["rows"], bracket["standings"], summary, head_to_head, bracket["league"])
    print_leaderboard(summary)
    print_head_to_head(bracket["top5"], head_to_head)
    figures = write_figures(out_dir)
    print(f"\nWrote {out_dir}", flush=True)
    print(f"Wrote {len(figures)} charts to {out_dir / 'figures'}", flush=True)
    return out_dir


if __name__ == "__main__":
    main()
