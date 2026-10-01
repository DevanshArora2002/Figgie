"""Read a Figgie match log and score the greedy policy against the deck."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

SUITS = ["heart", "diamond", "club", "spade"]
PARTNER = {"heart": "diamond", "diamond": "heart", "club": "spade", "spade": "club"}
SUIT_COLOR = {
    "heart": "#d62828",
    "diamond": "#e09f3e",
    "club": "#2a9d8f",
    "spade": "#1d3557",
}
PLAYER_COLOR = {1: "#264653", 2: "#e76f51", 3: "#2a9d8f", 4: "#c47d0e"}
BONUS_PER_CARD = 10
BOOT = 50
POT = 200

_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| (?P<level>\w+)\s+\| \S+ \| (?P<msg>.*)$"
)
_DEALT = re.compile(
    r"Round (?P<n>\d+) dealt \| counts=(?P<counts>\{.*?\}) \| golden_suit=(?P<golden>\w+) \((?P<gn>\d+) cards\) \| seats=(?P<seats>\[.*?\])"
)
_HAND = re.compile(
    r"Round (?P<n>\d+) hand \| P(?P<pid>\d+)\(cash=(?P<cash>[-0-9.]+), heart=(?P<heart>\d+),diamond=(?P<diamond>\d+),club=(?P<club>\d+),spade=(?P<spade>\d+)\)"
)
_STRATEGY = re.compile(
    r"Player (?P<pid>\d+) strategy \| goal=(?P<goal>\w+) \| anchor=(?P<anchor>\w+) \| counts=(?P<counts>\{.*?\}) \| cash=(?P<cash>[-0-9.]+)"
)
_TRADE = re.compile(
    r"Trade (?P<oid>order-\d+) \| (?P<suit>\w+) x1 @ (?P<price>[-0-9.]+) \| buyer=(?P<buyer>\d+) cash=(?P<bcash>[-0-9.]+) \| seller=(?P<seller>\d+) cash=(?P<scash>[-0-9.]+)"
)
_EXIT = re.compile(
    r"Player (?P<pid>\d+) left the market \| P(?P<pid2>\d+)\(cash=(?P<cash>[-0-9.]+), heart=(?P<heart>\d+),diamond=(?P<diamond>\d+),club=(?P<club>\d+),spade=(?P<spade>\d+)\)"
)
_BONUS = re.compile(
    r"Round (?P<n>\d+) bonus \| player=(?P<pid>\d+) \| golden_cards=(?P<cards>\d+) \| bonus=(?P<bonus>[-0-9.]+) \| cash=(?P<cash>[-0-9.]+)"
)
_POT = re.compile(
    r"Round (?P<n>\d+) pot awarded \| winners=(?P<winners>\[.*?\]) \| most_golden=(?P<most>\d+) \| share=(?P<share>[-0-9.]+) \| remainder_pot=(?P<rem>[-0-9.]+)"
)
_STANDINGS = re.compile(r"Round (?P<n>\d+) standings \| (?P<body>.*)")
_START = re.compile(r"Starting round (?P<n>\d+) of (?P<total>\d+)")
_QUOTE = re.compile(
    r"Player (?P<pid>\d+) quoting (?P<side>BUY|SELL) (?P<suit>\w+) @ (?P<price>[-0-9.]+)"
)


def _counts_from_groups(group) -> dict[str, int]:
    return {suit: int(group[suit]) for suit in SUITS}


def greedy_choice(hand: dict[str, int]) -> tuple[str, str]:
    """Same rule as GreedyPlayer: highest count, ties broken by suit order."""
    anchor = max(SUITS, key=lambda suit: hand[suit])
    return anchor, PARTNER[anchor]


def _worlds() -> list[tuple[str, str, str, dict[str, int]]]:
    worlds = []
    for twelve in SUITS:
        for eight in SUITS:
            if eight == twelve:
                continue
            counts = {suit: 10 for suit in SUITS}
            counts[twelve] = 12
            counts[eight] = 8
            worlds.append((twelve, eight, PARTNER[twelve], counts))
    return worlds


WORLDS = _worlds()


def posterior(hand: dict[str, int]) -> dict[str, float]:
    """P(suit is the prize | this 10-card hand).

    The deck is one of 12 equally likely layouts: one suit has 12, a different
    suit has 8, and the other two have 10. The prize is the same-color partner
    of the 12-card suit. The hand is a random 10-card draw from that deck.
    """
    weights = []
    for _twelve, _eight, _golden, counts in WORLDS:
        if any(hand[suit] > counts[suit] for suit in SUITS):
            weights.append(0)
            continue
        weight = 1
        for suit in SUITS:
            weight *= comb(counts[suit], hand[suit])
        weights.append(weight)
    total = sum(weights) or 1
    prize = {suit: 0.0 for suit in SUITS}
    for (_twelve, _eight, golden, _counts), weight in zip(WORLDS, weights):
        prize[golden] += weight / total
    return prize


def _goal_distribution(population: tuple[int, ...]) -> dict[str, float]:
    """How often the greedy rule names each prize, for a random 10-card hand."""
    goal_weight = {suit: 0.0 for suit in SUITS}
    total = 0.0

    def walk(index: int, left: int, acc: list[int]) -> None:
        nonlocal total
        if index == 3:
            count = left
            if count < 0 or count > population[3]:
                return
            weight = comb(population[3], count)
            hand = {}
            for suit, held, cap in zip(SUITS, [*acc, count], population):
                if held > cap:
                    return
                weight *= 1 if suit == SUITS[3] else comb(cap, held)
                hand[suit] = held
            if weight:
                goal_weight[greedy_choice(hand)[1]] += weight
                total += weight
            return
        for take in range(min(left, population[index]) + 1):
            acc.append(take)
            walk(index + 1, left - take, acc)
            acc.pop()

    walk(0, 10, [])
    if total == 0:
        return {suit: 0.0 for suit in SUITS}
    return {suit: goal_weight[suit] / total for suit in SUITS}


_GOAL_CACHE: dict[tuple[int, ...], dict[str, float]] = {}


def market_posterior(hand: dict[str, int], opponent_goals: list[str]) -> dict[str, float]:
    """P(suit is the prize | my hand, and the suits the other greedy bots are buying).

    Each opponent is assumed to use the same rule. Their bid is scored against
    the cards still in the deck after my hand is removed. The three opponents
    are treated as separate draws, which is a simplification of the real deal.
    """
    weights = []
    goldens = []
    for _twelve, _eight, golden, counts in WORLDS:
        if any(hand[suit] > counts[suit] for suit in SUITS):
            weights.append(0.0)
            goldens.append(golden)
            continue
        weight = 1.0
        for suit in SUITS:
            weight *= comb(counts[suit], hand[suit])
        remain = tuple(counts[suit] - hand[suit] for suit in SUITS)
        if remain not in _GOAL_CACHE:
            _GOAL_CACHE[remain] = _goal_distribution(remain)
        likelihood = _GOAL_CACHE[remain]
        for goal in opponent_goals:
            weight *= likelihood[goal]
        weights.append(weight)
        goldens.append(golden)
    total = sum(weights) or 1.0
    prize = {suit: 0.0 for suit in SUITS}
    for weight, golden in zip(weights, goldens):
        prize[golden] += weight / total
    return prize


def bonus_floor(hand: dict[str, int]) -> dict[str, float]:
    """Lowest settlement value of one card, before any chance at the pot.

    Every prize card pays 10 to whoever holds it. A card of suit S is worth
    at least 10 * P(S is the prize). The pot contest can only add value.
    """
    belief = posterior(hand)
    return {suit: BONUS_PER_CARD * belief[suit] for suit in SUITS}


@dataclass
class Quote:
    player: int
    side: str
    suit: str
    price: float
    timestamp: str


@dataclass
class Trade:
    order_id: str
    suit: str
    price: float
    buyer: int
    seller: int
    buyer_cash: float
    seller_cash: float
    timestamp: str


@dataclass
class Round:
    number: int
    counts: dict[str, int]
    golden: str
    golden_n: int
    seats: list[int]
    hands: dict[int, dict[str, int]] = field(default_factory=dict)
    deal_cash: dict[int, float] = field(default_factory=dict)
    goals: dict[int, str] = field(default_factory=dict)
    anchors: dict[int, str] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    exit_cash: dict[int, float] = field(default_factory=dict)
    exit_hands: dict[int, dict[str, int]] = field(default_factory=dict)
    bonus_cards: dict[int, int] = field(default_factory=dict)
    bonus_cash: dict[int, float] = field(default_factory=dict)
    winners: list[int] = field(default_factory=list)
    share: float | None = None
    remainder: float | None = None
    standings: dict[int, float] = field(default_factory=dict)

    @property
    def settled(self) -> bool:
        return bool(self.standings)

    def players(self) -> list[int]:
        return sorted(self.hands)


@dataclass
class Match:
    rounds: list[Round]
    scheduled_rounds: int | None = None

    def settled_rounds(self) -> list[Round]:
        return [rnd for rnd in self.rounds if rnd.settled]


def parse_log(path: str | Path) -> Match:
    rounds: dict[int, Round] = {}
    current: Round | None = None
    scheduled = None
    for line in Path(path).read_text().splitlines():
        parsed = _LINE.match(line)
        if parsed is None or parsed.group("level") not in ("INFO", "DEBUG"):
            continue
        msg = parsed.group("msg")
        ts = parsed.group("ts")
        if parsed.group("level") == "DEBUG":
            quote = _QUOTE.match(msg)
            if quote and current is not None:
                current.quotes.append(
                    Quote(
                        player=int(quote.group("pid")),
                        side=quote.group("side"),
                        suit=quote.group("suit"),
                        price=float(quote.group("price")),
                        timestamp=ts,
                    )
                )
            continue

        started = _START.match(msg)
        if started:
            scheduled = int(started.group("total"))
            current = None
            continue

        dealt = _DEALT.match(msg)
        if dealt:
            number = int(dealt.group("n"))
            current = Round(
                number=number,
                counts=ast.literal_eval(dealt.group("counts")),
                golden=dealt.group("golden"),
                golden_n=int(dealt.group("gn")),
                seats=ast.literal_eval(dealt.group("seats")),
            )
            rounds[number] = current
            continue
        if current is None:
            continue

        hand = _HAND.match(msg)
        if hand and int(hand.group("n")) == current.number:
            pid = int(hand.group("pid"))
            current.hands[pid] = _counts_from_groups(hand)
            current.deal_cash[pid] = float(hand.group("cash"))
            continue

        strategy = _STRATEGY.match(msg)
        if strategy:
            pid = int(strategy.group("pid"))
            current.goals[pid] = strategy.group("goal")
            current.anchors[pid] = strategy.group("anchor")
            continue

        trade = _TRADE.match(msg)
        if trade:
            current.trades.append(
                Trade(
                    order_id=trade.group("oid"),
                    suit=trade.group("suit"),
                    price=float(trade.group("price")),
                    buyer=int(trade.group("buyer")),
                    seller=int(trade.group("seller")),
                    buyer_cash=float(trade.group("bcash")),
                    seller_cash=float(trade.group("scash")),
                    timestamp=ts,
                )
            )
            continue

        exit_ = _EXIT.match(msg)
        if exit_:
            pid = int(exit_.group("pid"))
            current.exit_cash[pid] = float(exit_.group("cash"))
            current.exit_hands[pid] = _counts_from_groups(exit_)
            continue

        bonus = _BONUS.match(msg)
        if bonus and int(bonus.group("n")) == current.number:
            pid = int(bonus.group("pid"))
            current.bonus_cards[pid] = int(bonus.group("cards"))
            current.bonus_cash[pid] = float(bonus.group("bonus"))
            continue

        pot = _POT.match(msg)
        if pot and int(pot.group("n")) == current.number:
            current.winners = ast.literal_eval(pot.group("winners"))
            current.share = float(pot.group("share"))
            current.remainder = float(pot.group("rem"))
            continue

        standings = _STANDINGS.match(msg)
        if standings and int(standings.group("n")) == current.number:
            current.standings = {
                int(pid): float(cash)
                for pid, cash in re.findall(r"p(\d+)=([-0-9.]+)", standings.group("body"))
            }

    ordered = [rounds[number] for number in sorted(rounds)]
    for rnd in ordered:
        for player, hand in rnd.hands.items():
            if player in rnd.goals and player in rnd.anchors:
                continue
            anchor, goal = greedy_choice(hand)
            rnd.anchors.setdefault(player, anchor)
            rnd.goals.setdefault(player, goal)
    return Match(rounds=ordered, scheduled_rounds=scheduled)


def settle(hands: dict[int, dict[str, int]], cash: dict[int, float], golden: str) -> dict[int, float]:
    """Apply the engine payout: 10 per prize card, then the leftover pot."""
    bonus = {pid: hands[pid][golden] * BONUS_PER_CARD for pid in hands}
    remainder = POT - sum(bonus.values())
    most = max(hands[pid][golden] for pid in hands)
    winners = [pid for pid in hands if hands[pid][golden] == most]
    share = remainder / len(winners)
    return {pid: cash[pid] + bonus[pid] + (share if pid in winners else 0.0) for pid in hands}


def replay(rnd: Round, keep) -> tuple[dict[int, dict[str, int]], dict[int, float]]:
    """Replay the tape from the deal. `keep(trade, seller_hand_at_deal)` drops trades."""
    hands = {pid: dict(counts) for pid, counts in rnd.hands.items()}
    cash = {pid: rnd.deal_cash[pid] - BOOT for pid in rnd.hands}
    opening = {pid: dict(counts) for pid, counts in rnd.hands.items()}
    for trade in rnd.trades:
        if not keep(trade, opening[trade.seller]):
            continue
        hands[trade.seller][trade.suit] -= 1
        hands[trade.buyer][trade.suit] += 1
        cash[trade.buyer] -= trade.price
        cash[trade.seller] += trade.price
    return hands, cash


def actual_replay_matches_log(rnd: Round) -> bool:
    if not rnd.exit_hands:
        return True
    hands, cash = replay(rnd, lambda trade, _hand: True)
    for pid, counts in rnd.exit_hands.items():
        if hands[pid] != counts:
            return False
        if abs(cash[pid] - rnd.exit_cash[pid]) > 0.02:
            return False
    if rnd.standings:
        final = settle(hands, cash, rnd.golden)
        for pid, standing in rnd.standings.items():
            if abs(final[pid] - standing) > 0.02:
                return False
    return True


@dataclass
class TradeNote:
    trade: Trade
    seller_p: float
    buyer_p: float
    seller_floor: float
    buyer_floor: float
    is_prize: bool

    @property
    def given_up(self) -> float:
        return max(0.0, self.seller_floor - self.trade.price)

    @property
    def premium(self) -> float:
        return max(0.0, self.trade.price - self.buyer_floor)


def annotate(rnd: Round) -> list[TradeNote]:
    seller_belief = {pid: posterior(rnd.hands[pid]) for pid in rnd.hands}
    notes = []
    for trade in rnd.trades:
        seller_p = seller_belief[trade.seller][trade.suit]
        buyer_p = seller_belief[trade.buyer][trade.suit]
        notes.append(
            TradeNote(
                trade=trade,
                seller_p=seller_p,
                buyer_p=buyer_p,
                seller_floor=BONUS_PER_CARD * seller_p,
                buyer_floor=BONUS_PER_CARD * buyer_p,
                is_prize=trade.suit == rnd.golden,
            )
        )
    return notes


def floor_counterfactual(rnd: Round) -> dict[int, float]:
    """Standings if sellers refuse any trade priced under their bonus floor."""
    def keep(trade: Trade, opening: dict[str, int]) -> bool:
        floor = BONUS_PER_CARD * posterior(opening)[trade.suit]
        return trade.price + 1e-9 >= floor

    hands, cash = replay(rnd, keep)
    return settle(hands, cash, rnd.golden)


def round_profit(rnd: Round) -> dict[int, float]:
    if rnd.standings:
        return {pid: rnd.standings[pid] - rnd.deal_cash[pid] for pid in rnd.standings}
    if len(rnd.exit_hands) == len(rnd.hands):
        final = settle(rnd.exit_hands, rnd.exit_cash, rnd.golden)
        return {pid: final[pid] - rnd.deal_cash[pid] for pid in final}
    return {}


def guessed_right(rnd: Round) -> dict[int, bool]:
    return {pid: rnd.goals.get(pid) == rnd.golden for pid in rnd.hands}


def favoured(dist: dict[str, float]) -> str:
    return max(SUITS, key=lambda suit: dist[suit])


def opponent_goals(rnd: Round, player: int) -> list[str]:
    return [rnd.goals[other] for other in rnd.players() if other != player]


def market_view(rnd: Round, player: int) -> dict[str, float]:
    return market_posterior(rnd.hands[player], opponent_goals(rnd, player))
