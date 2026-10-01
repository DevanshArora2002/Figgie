"""Charts that show how the greedy bots play a logged match."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import analyse

_TIME = "%Y-%m-%d %H:%M:%S.%f"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d0d0d0",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def _elapsed(timestamp: str, origin: str) -> float:
    return (
        datetime.strptime(timestamp, _TIME) - datetime.strptime(origin, _TIME)
    ).total_seconds()


def _finish(fig: plt.Figure) -> plt.Figure:
    fig.tight_layout()
    return fig


def _hand_panel(ax, rnd: analyse.Round, player: int) -> None:
    hand = rnd.hands[player]
    anchor = rnd.anchors[player]
    goal = rnd.goals[player]
    colors = [analyse.SUIT_COLOR[suit] for suit in analyse.SUITS]
    counts = [hand[suit] for suit in analyse.SUITS]
    ax.bar(analyse.SUITS, counts, color=colors, width=0.72, zorder=2)
    ax.set_ylim(0, max(counts) + 2.4)
    ax.set_ylabel("cards dealt")
    ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)
    for index, suit in enumerate(analyse.SUITS):
        tags = []
        if suit == anchor:
            tags.append("biggest pile")
        if suit == goal:
            tags.append("bot buys this")
        if suit == rnd.golden:
            tags.append("actual prize")
        if not tags:
            continue
        ax.text(index, counts[index] + 0.12, "\n".join(tags), ha="center", va="bottom", fontsize=8, color="#333333")
    correct = goal == rnd.golden
    verdict = "and that suit is the prize" if correct else "but that suit is not the prize"
    ax.set_title(f"Player {player} buys {goal}, {verdict}", loc="left", pad=8)


def case_studies(match: analyse.Match) -> tuple[tuple[analyse.Round, int], tuple[analyse.Round, int]]:
    """The clearest correct guess, and the player who was dealt the most prize cards and sold them."""
    worked = None
    worked_score = (-1, -1)
    failed = None
    failed_prize = -1
    for rnd in match.rounds:
        if not rnd.goals or not rnd.hands:
            continue
        for player in rnd.players():
            if player not in rnd.goals:
                continue
            pile = max(rnd.hands[player].values())
            held_at_end = rnd.exit_hands.get(player, {}).get(rnd.golden, 0)
            if rnd.goals[player] == rnd.golden and (pile, held_at_end) > worked_score:
                worked = (rnd, player)
                worked_score = (pile, held_at_end)
            prize_dealt = rnd.hands[player][rnd.golden]
            if rnd.goals[player] != rnd.golden and prize_dealt > failed_prize:
                failed = (rnd, player)
                failed_prize = prize_dealt
    fallback = (match.rounds[0], match.rounds[0].players()[0])
    return worked or fallback, failed or worked or fallback


def _deck_caption(rnd: analyse.Round) -> str:
    twelve = max(analyse.SUITS, key=lambda suit: rnd.counts[suit])
    return f"Round {rnd.number}  ·  {rnd.counts[twelve]} {twelve}s, prize is {rnd.golden}"


def rule_figure(match: analyse.Match) -> plt.Figure:
    """Same decision rule on a hand where it works and a hand where it sells the prize."""
    apply_style()
    (worked_round, worked_player), (failed_round, failed_player) = case_studies(match)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    _hand_panel(axes[0], worked_round, worked_player)
    _hand_panel(axes[1], failed_round, failed_player)
    axes[0].set_xlabel(_deck_caption(worked_round))
    axes[1].set_xlabel(_deck_caption(failed_round))

    buy_suit = failed_round.goals[failed_player]
    sell_suit = failed_round.golden
    quotes = [
        quote
        for quote in failed_round.quotes
        if quote.player == failed_player and quote.suit in (buy_suit, sell_suit)
    ]
    ax = axes[2]
    if quotes:
        origin = quotes[0].timestamp
        for suit, side, label in (
            (sell_suit, "SELL", f"offers in {sell_suit}, the prize"),
            (buy_suit, "BUY", f"bids in {buy_suit}, the guess"),
        ):
            series = [quote for quote in quotes if quote.suit == suit and quote.side == side]
            if not series:
                continue
            ax.step(
                [_elapsed(quote.timestamp, origin) for quote in series],
                [quote.price for quote in series],
                where="post",
                color=analyse.SUIT_COLOR[suit],
                linewidth=2,
                label=label,
            )
        ax.set_ylim(0, 11)
        ax.set_xlim(left=0)
        ax.set_xlabel("seconds after the first quote")
        ax.set_ylabel("price")
        ax.legend(loc="upper right")
        ax.yaxis.grid(True, color="#f0f0f0")
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "No quote ladder in this log", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Then it walks the price until someone crosses", loc="left", pad=8)
    fig.suptitle("One rule: buy the color partner of your biggest pile", fontsize=14, y=1.02)
    return _finish(fig)


def scoreboard_figure(match: analyse.Match) -> plt.Figure:
    apply_style()
    rounds = [rnd for rnd in match.rounds if rnd.standings]
    players = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    ax = axes[0]
    x = np.arange(len(rounds))
    width = 0.18
    for offset, player in enumerate(players):
        profits = [analyse.round_profit(rnd)[player] for rnd in rounds]
        positions = x + (offset - 1.5) * width
        ax.bar(positions, profits, width, color=analyse.PLAYER_COLOR[player], label=f"Player {player}", zorder=2)
        for position, rnd, profit in zip(positions, rounds, profits):
            if rnd.goals.get(player) == rnd.golden:
                ax.scatter(position, profit, s=18, color="white", zorder=3)
                ax.scatter(position, profit, s=8, color="#111111", zorder=4)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(x, [f"Round {rnd.number}\nprize {rnd.golden}" for rnd in rounds])
    ax.set_ylabel("chips won this round")
    ax.set_title("Round result. A dot means that player named the prize.", loc="left")
    ax.legend(ncol=4, loc="upper right")
    ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    for player in players:
        cash = [350] + [rnd.standings[player] for rnd in rounds]
        ax.plot(
            range(len(cash)),
            cash,
            color=analyse.PLAYER_COLOR[player],
            marker="o",
            linewidth=2,
            label=f"Player {player}",
        )
    ax.set_xticks(range(len(rounds) + 1), ["deal"] + [f"after {rnd.number}" for rnd in rounds])
    ax.set_ylabel("cash")
    ax.set_title("Match cash. Early sweeps still lead after the later give-backs.", loc="left")
    ax.legend(ncol=4, loc="upper left")
    ax.yaxis.grid(True, color="#f0f0f0")
    ax.set_axisbelow(True)
    return _finish(fig)


def inventory_figure(match: analyse.Match) -> plt.Figure:
    apply_style()
    rounds = match.rounds
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), sharey=True)
    for ax, rnd in zip(axes.ravel(), rounds):
        players = rnd.players()
        start = [rnd.hands[player][rnd.golden] for player in players]
        end = [
            rnd.exit_hands.get(player, rnd.hands[player])[rnd.golden]
            for player in players
        ]
        x = np.arange(len(players))
        width = 0.36
        ax.bar(x - width / 2, start, width, color="#d9d9d9", label="dealt", zorder=2)
        ax.bar(x + width / 2, end, width, color="#243b53", label="after the auction", zorder=2)
        buyers = [f"P{player}" for player in players if rnd.goals.get(player) == rnd.golden]
        buyer_text = ", ".join(buyers) if buyers else "nobody"
        ax.set_title(f"Round {rnd.number}: prize is {rnd.golden}. Buying it: {buyer_text}", loc="left", fontsize=11)
        ax.set_xticks(x, [f"P{player}" for player in players])
        for position, before, after in zip(x, start, end):
            ax.text(position - width / 2, before + 0.15, str(before), ha="center", va="bottom", fontsize=8, color="#555555")
            ax.text(position + width / 2, after + 0.15, str(after), ha="center", va="bottom", fontsize=8, color="#243b53")
        ax.set_ylim(0, 13)
        ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
        ax.set_axisbelow(True)
        if ax is axes[0, 0]:
            ax.legend(loc="upper right")
            ax.set_ylabel("prize cards")
        if ax is axes[1, 0]:
            ax.set_ylabel("prize cards")
    fig.suptitle("Prize cards move to whoever decided to buy that suit", fontsize=14)
    return _finish(fig)


def auction_figure(rnd: analyse.Round) -> plt.Figure:
    """Trade prices and who ends up holding the prize, for one round."""
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    if rnd.trades:
        origin = rnd.trades[0].timestamp
    elif rnd.quotes:
        origin = rnd.quotes[0].timestamp
    else:
        origin = None

    ax = axes[0]
    holdings = {player: rnd.hands[player][rnd.golden] for player in rnd.players()}
    series = {player: [holdings[player]] for player in rnd.players()}
    times = [0.0]
    for trade in rnd.trades:
        if trade.suit == rnd.golden:
            holdings[trade.seller] -= 1
            holdings[trade.buyer] += 1
        times.append(_elapsed(trade.timestamp, origin))
        for player in rnd.players():
            series[player].append(holdings[player])
    for player in rnd.players():
        ax.step(times, series[player], where="post", color=analyse.PLAYER_COLOR[player], linewidth=2, label=f"Player {player}")
    ax.set_ylabel(f"{rnd.golden} cards held")
    ax.set_title(f"Round {rnd.number}: who is holding the prize while the auction runs", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.yaxis.grid(True, color="#f0f0f0")
    ax.set_axisbelow(True)

    ax = axes[1]
    for suit in analyse.SUITS:
        xs = [_elapsed(trade.timestamp, origin) for trade in rnd.trades if trade.suit == suit]
        ys = [trade.price for trade in rnd.trades if trade.suit == suit]
        size = 46 if suit == rnd.golden else 28
        ax.scatter(xs, ys, s=size, color=analyse.SUIT_COLOR[suit], label=suit, zorder=3 if suit == rnd.golden else 2)
    ax.set_ylabel("trade price")
    ax.set_xlabel("seconds from the first trade")
    ax.set_title("Fills. The prize suit is drawn larger. The price is just where the two ladders met.", loc="left")
    ax.legend(ncol=4, loc="upper right")
    ax.set_ylim(0, 10)
    ax.yaxis.grid(True, color="#f0f0f0")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.55, top=0.86)
    return fig


def belief_figure(rnd: analyse.Round) -> plt.Figure:
    apply_style()
    players = rnd.players()
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 4.2), sharey=True)
    x = np.arange(len(analyse.SUITS))
    width = 0.38
    for ax, player in zip(axes, players):
        own = analyse.posterior(rnd.hands[player])
        market = analyse.market_view(rnd, player)
        colors = [analyse.SUIT_COLOR[suit] for suit in analyse.SUITS]
        ax.bar(x - width / 2, [own[suit] for suit in analyse.SUITS], width, color=colors, alpha=0.35, zorder=2)
        ax.bar(x + width / 2, [market[suit] for suit in analyse.SUITS], width, color=colors, zorder=2)
        goal = rnd.goals[player]
        goal_index = analyse.SUITS.index(goal)
        marker_y = max(own[goal], market[goal]) + 0.06
        ax.scatter([goal_index], [marker_y], marker="v", color="#111111", s=32, zorder=4, clip_on=False)
        prize_index = analyse.SUITS.index(rnd.golden)
        ax.axvline(prize_index, color="#e6c200", linewidth=6, alpha=0.28, zorder=0)
        crowd = {}
        for other in players:
            if other == player:
                continue
            crowd[rnd.goals[other]] = crowd.get(rnd.goals[other], 0) + 1
        agreed = [f"{count} opponents bid {suit}" for suit, count in crowd.items() if count >= 2]
        crowd_text = ", ".join(agreed) if agreed else "no two opponents agree"
        picked = analyse.favoured(market)
        if picked == rnd.golden and goal != rnd.golden:
            hint = "crowd corrects the guess"
        elif picked == rnd.golden and goal == rnd.golden:
            hint = "still on the prize"
        else:
            hint = "still misses the prize"
        ax.set_title(f"P{player} bids {goal}\n{hint}", loc="left", fontsize=10)
        ax.set_xticks(x, analyse.SUITS, rotation=30, ha="right")
        ax.set_xlabel(crowd_text, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color="#f0f0f0", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("P(this suit is the prize)")
    handles = [
        Patch(facecolor="#2a9d8f", alpha=0.35, label="from this hand only"),
        Patch(facecolor="#2a9d8f", label="after the other bids"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#111111", markersize=7, label="suit the bot buys"),
        Patch(facecolor="#e6c200", alpha=0.45, label="actual prize"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Round {rnd.number}: what the hand says, and what changes when you read the other three bids",
        fontsize=13,
        y=1.12,
    )
    fig.tight_layout()
    return fig


def corrections(match: analyse.Match) -> list[str]:
    found = []
    for rnd in match.rounds:
        for player in rnd.players():
            market_pick = analyse.favoured(analyse.market_view(rnd, player))
            greedy_pick = rnd.goals[player]
            if greedy_pick != rnd.golden and market_pick == rnd.golden:
                found.append(
                    f"round {rnd.number} player {player} (bid {greedy_pick}, the others point at {market_pick})"
                )
    return found


def tie_break_sales(match: analyse.Match) -> list[str]:
    """Hands where a tie made the bot sell a suit that was actually the prize."""
    found = []
    for rnd in match.rounds:
        for player in rnd.players():
            hand = rnd.hands[player]
            top = max(hand.values())
            tied = [suit for suit in analyse.SUITS if hand[suit] == top]
            if len(tied) < 2 or rnd.golden not in tied or rnd.goals.get(player) == rnd.golden:
                continue
            found.append(
                f"round {rnd.number} player {player} "
                f"(tied on {', '.join(tied)}, bought {rnd.goals[player]}, prize was {rnd.golden})"
            )
    return found


def match_note(match: analyse.Match) -> str:
    hands = 0
    greedy_hits = 0
    market_hits = 0
    for rnd in match.rounds:
        for player in rnd.players():
            hands += 1
            if rnd.goals.get(player) == rnd.golden:
                greedy_hits += 1
            if analyse.favoured(analyse.market_view(rnd, player)) == rnd.golden:
                market_hits += 1
    fixed = corrections(match)
    undone = regressions(match)
    fixed_text = "; ".join(fixed) if fixed else "none"
    undone_text = "; ".join(undone) if undone else "none"
    ties = tie_break_sales(match)
    tie_text = "; ".join(ties) if ties else "none"
    return (
        f"On this log the opening rule names the prize in {greedy_hits} of {hands} hands. "
        f"Rescoring those hands with the other three bids names it in {market_hits} of {hands}. "
        f"The hands that flip onto the prize are: {fixed_text}. "
        f"Hands where that rescoring pulls a correct guess off the prize: {undone_text}. "
        f"Ties where list order made the bot sell the prize: {tie_text}."
    )


def regressions(match: analyse.Match) -> list[str]:
    found = []
    for rnd in match.rounds:
        for player in rnd.players():
            market_pick = analyse.favoured(analyse.market_view(rnd, player))
            if rnd.goals[player] == rnd.golden and market_pick != rnd.golden:
                found.append(f"round {rnd.number} player {player}")
    return found
