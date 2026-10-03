from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import logging
import math
from utils import *
from card import Card

from typing import TYPE_CHECKING
import itertools
import numpy as np

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union
    from order_book import OrderBookManager

logger = logging.getLogger(__name__)


def _build_decks() -> list[dict]:
    """The 12 Figgie decks: each suit count, the goal suit, its majority size, and its pot share."""
    decks = []
    seen: set[tuple[int, ...]] = set()
    for counts in itertools.permutations(NO_CARDS):
        if counts in seen:
            continue
        seen.add(counts)
        counts_by_suit = dict(zip(SUITS, counts))
        twelve_suit = next(suit for suit, count in counts_by_suit.items() if count == 12)
        goal = next(suit for suit in color_to_suit[suit_to_color[twelve_suit]] if suit != twelve_suit)
        goal_count = counts_by_suit[goal]
        decks.append({
            "counts": counts_by_suit,
            "goal": goal,
            "x": goal_count // 2 + 1,
            "p": 4 * BOOT_AMOUNT - CASH_GAIN_PER_GOLDEN_CARD * goal_count,
        })
    return decks


DECKS = _build_decks()


class Player(ABC):
    _id_counter = itertools.count(1)
    def __init__(self):
        self.id = next(self._id_counter)
        self.cash = STARTING_CASH
        self.cards_lst: List[Card] = []
        logger.debug("Player %s created | cash=%s", self.id, self.cash)

    def __repr__(self) -> str:
        counts = ",".join(f"{suit}={self.get_cards_by_suit(suit)}" for suit in SUITS)
        return f"P{self.id}(cash={self.cash:.2f}, {counts})"

    def add_card(self, card: Card) -> None:
        """Add one dealt or bought card to this player's hand."""
        self.cards_lst.append(card)
        logger.debug("Player %s took %s | hand_size=%s", self.id, card, len(self.cards_lst))

    def remove_card(self, card: Card) -> None:
        """Remove one sold card from this player's hand."""
        self.cards_lst.remove(card)
        logger.debug("Player %s gave %s | hand_size=%s", self.id, card, len(self.cards_lst))

    def get_cards_by_suit(self, suit: str) -> int:
        return sum(1 for card in self.cards_lst if card.suit == suit)

    def give_boot(self, amount: float) -> None:
        self.cash -= amount
        logger.debug("Player %s paid boot %s | cash=%.2f", self.id, amount, self.cash)

    def get_cash(self, amount: float) -> None:
        self.cash += amount
        logger.debug("Player %s received %.2f | cash=%.2f", self.id, amount, self.cash)

    @abstractmethod
    async def play(self, order_book: OrderBookManager) -> None:
        pass

import heapq

class MaxHeapItem:
    def __init__(self, val, data=None):
        self.val = val
        self.data = data

    # Reverse '<' comparison so heapq treats larger values as smaller
    def __lt__(self, other):
        return self.val > other.val

    def __repr__(self):
        return f"MaxHeapItem(val={self.val}, data={self.data})"



class GreedyPlayer(Player):
    """
    A simple greedy player that:
    - Identifies a goal suit based on its starting hand (same color as most-held suit)
    - Bids to buy goal suit cards starting at price 1, increasing on rejection
    - Offers to sell other suits starting at price 10, decreasing on rejection
    """

    def _get_precedence(self, suit_counts: dict[str, int], suit: str) -> int:
        suit_color = suit_to_color[suit]
        partner = next(other for other in color_to_suit[suit_color] if other != suit)
        return suit_counts[partner] + 10 - suit_counts[suit]

    
    async def play(self, order_book: OrderBookManager) -> None:
        from order_book import OrderStatus, RejectReason, OrderHandle

        # Determine goal suit: the other suit of the same color as our most-held suit
        suit_counts = {suit: self.get_cards_by_suit(suit) for suit in SUITS}
        heap = []
        items = [(self._get_precedence(suit_counts, suit), suit) for suit in SUITS]

        for priority, task in items:
            heapq.heappush(heap, MaxHeapItem(priority, task))
        

        precedence = [heapq.heappop(heap).data for _ in range(len(SUITS))]
        goal_suit = precedence[0]
        sell_suit_precedence = list(reversed(precedence[1:]))
        anchor = max(SUITS, key=lambda suit: suit_counts[suit])

        logger.info(
            "Player %s strategy | goal=%s | anchor=%s | counts=%s | cash=%.2f",
            self.id,
            goal_suit,
            anchor,
            suit_counts,
            self.cash,
        )

        # Initialize prices and handles
        # price[suit] = current price for that suit
        # handle[suit] = current OrderHandle or None
        # For goal_suit: we BID (buy)
        # For other suits: we OFFER (sell)
        prices: dict[str, float] = {}
        handles: dict[str, OrderHandle | None] = {}
        active: set[str] = set()

        # Goal suit: bid starting at GREEDY_START_BID
        prices[goal_suit] = GREEDY_START_BID
        handles[goal_suit] = None
        active.add(goal_suit)

        # Other suits: offer starting at GREEDY_START_OFFER
        for suit in sell_suit_precedence:
            prices[suit] = GREEDY_START_OFFER
            handles[suit] = None
            # Only active if we have cards to sell
            if self.get_cards_by_suit(suit) > 0:
                active.add(suit)

        # Track how long each order has been resting
        resting_ticks: dict[str, int] = {suit: 0 for suit in SUITS}
        RESTING_PATIENCE = 3  # Improve price after this many ticks of resting

        try:
            while active:
                for suit in list(active):
                    handle = handles[suit]
                    is_buy = (suit == goal_suit)
                    should_submit = False
                    improve_price = False

                    if handle is None:
                        # No active handle, submit a new order
                        should_submit = True

                    elif handle.status == OrderStatus.RESTING:
                        # Order is resting, wait for it to get filled or improve after patience
                        resting_ticks[suit] += 1
                        if resting_ticks[suit] >= RESTING_PATIENCE:
                            logger.debug(
                                "Player %s quote on %s rested %s ticks, improving",
                                self.id,
                                suit,
                                resting_ticks[suit],
                            )
                            # Order has been resting too long, improve price and resubmit
                            improve_price = True
                            should_submit = True
                            resting_ticks[suit] = 0

                    elif handle.is_terminal():
                        # Handle terminal states
                        resting_ticks[suit] = 0
                        logger.debug(
                            "Player %s saw %s on %s -> %s%s",
                            self.id,
                            handle.order_id,
                            suit,
                            handle.status.name,
                            f" ({handle.reason.name})" if handle.reason else "",
                        )

                        if handle.status == OrderStatus.FILLED:
                            # Great! Re-submit at same price
                            should_submit = True

                        elif handle.status == OrderStatus.CANCELLED:
                            # Our order was cancelled (trade happened elsewhere), re-submit at same price
                            should_submit = True

                        elif handle.status == OrderStatus.REJECTED:
                            reason = handle.reason

                            if reason == RejectReason.WORSE_PRICE:
                                # Our price wasn't good enough, improve it
                                improve_price = True
                                should_submit = True

                            elif reason == RejectReason.NO_CARD:
                                # No card to sell, drop this suit
                                logger.info("Player %s stopped selling %s | no cards left", self.id, suit)
                                active.discard(suit)
                                handles[suit] = None
                                continue

                            elif reason == RejectReason.NO_CASH:
                                # Not enough cash to buy, wait for a sale
                                handles[suit] = None
                                if self.cash < prices[suit]:
                                    continue
                                should_submit = True

                            elif reason == RejectReason.SELF_TRADE:
                                # Tried to trade with ourselves, just retry
                                should_submit = True

                    # Improve price if needed
                    if improve_price:
                        previous_price = prices[suit]
                        if is_buy:
                            prices[suit] = min(prices[suit] + 1, MAX_BID)
                            logger.debug(
                                "Player %s bid %s price %s -> %s",
                                self.id,
                                suit,
                                previous_price,
                                prices[suit],
                            )
                            if prices[suit] >= MAX_BID:
                                logger.info("Player %s stopped buying %s | bid reached %s", self.id, suit, prices[suit])
                                active.discard(suit)
                                handles[suit] = None
                                continue
                        else:
                            prices[suit] = max(prices[suit] - 1, MIN_OFFER)
                            logger.debug(
                                "Player %s offer %s price %s -> %s",
                                self.id,
                                suit,
                                previous_price,
                                prices[suit],
                            )
                            if prices[suit] <= MIN_OFFER:
                                logger.info("Player %s stopped selling %s | offer reached %s", self.id, suit, prices[suit])
                                active.discard(suit)
                                handles[suit] = None
                                continue

                    # Submit new order if needed
                    if should_submit:
                        # For sells, check if we still have cards
                        if not is_buy and self.get_cards_by_suit(suit) == 0:
                            logger.info("Player %s stopped selling %s | no cards left", self.id, suit)
                            active.discard(suit)
                            handles[suit] = None
                            continue

                        # For buys, check if we have enough cash
                        if is_buy and self.cash < prices[suit]:
                            handles[suit] = None
                            continue

                        price = prices[suit]
                        side = "BUY" if is_buy else "SELL"
                        logger.debug("Player %s quoting %s %s @ %s", self.id, side, suit, price)
                        handles[suit] = await order_book.submit_order(suit, side, price, self)

                await asyncio.sleep(GREEDY_TICK)
        finally:
            logger.info("Player %s left the market | %s", self.id, self)

class RandomPlayer(Player):
    """
    A simple random player whose task is to chose the reward randomly from a uniform distribution.
    """

    async def play(self, order_book: OrderBookManager) -> None:
        np.set.seed(SEED)
        while True:
            prob = np.random.uniform(0, 1)
            if prob < 0.5:
                side = "BUY"
            else:
                side = "SELL"
            
            if side == "BUY":
                price = int(np.random.uniform(RANDOM_MIN_OFFER, min(self.cash, RANDOM_MAX_BID)))
                suit = np.random.choice(SUITS)
                logger.debug(
                        "Player %s quoting %s %s @ %s",
                        self.id,
                        side,
                        suit,
                        price,
                    )
                await order_book.submit_order(suit, side, price, self)
            elif side == "SELL":
                suit = np.random.choice(SUITS)
                if self.get_cards_by_suit(suit) > 0:
                    price = int(np.random.uniform(RANDOM_MIN_OFFER, RANDOM_MAX_BID))
                    logger.debug(
                        "Player %s quoting %s %s @ %s",
                        self.id,
                        side,
                        suit,
                        price,
                    )
                    await order_book.submit_order(suit, side, price, self)
            await asyncio.sleep(RANDOM_TICK)

class ChartistPlayer(Player):
    """
    Estimates each suit's future value from the average return of its trades.
    Buy and sell are chosen at random. The quote is that future value, rounded.
    """

    def _trade_prices(self, order_book: OrderBookManager, suit: str) -> list[float]:
        return [
            trade["PRICE"]
            for trade in order_book.order_executed
            if trade["ASSET"] == suit
        ]

    def _expected_value(self, prices: list[float]) -> float | None:
        """
        Future value p_t * exp(r_bar * tau).

        r_bar is the average log return over the last tau trades, which
        telescopes to (1 / tau) * ln(p_{t-1} / p_{t-tau-1}).
        """
        tau = CHARTIST_TAU
        if len(prices) < tau + 2:
            return None
        current = prices[-1]
        previous = prices[-2]
        horizon_start = prices[-(tau + 2)]
        if current <= 0 or previous <= 0 or horizon_start <= 0:
            return None
        average_return = (1.0 / tau) * np.log(previous / horizon_start)
        expected = float(current * np.exp(average_return * tau))
        if not np.isfinite(expected):
            return None
        return expected

    async def play(self, order_book: OrderBookManager) -> None:
        np.random.seed(SEED)
        needed = CHARTIST_TAU + 2
        while True:
            prob = np.random.uniform(0, 1)
            side = "BUY" if prob < 0.5 else "SELL"

            ready = [
                suit
                for suit in SUITS
                if len(self._trade_prices(order_book, suit)) >= needed
            ]
            if ready:
                suit = str(np.random.choice(ready))
                expected = self._expected_value(self._trade_prices(order_book, suit))
                price = round(expected) if expected is not None else 0
                can_buy = side == "BUY" and price >= 1 and self.cash >= price
                can_sell = side == "SELL" and price >= 1 and self.get_cards_by_suit(suit) > 0
                if can_buy or can_sell:
                    logger.debug(
                        "Player %s quoting %s %s @ %s | value=%.2f",
                        self.id,
                        side,
                        suit,
                        price,
                        expected,
                    )
                    await order_book.submit_order(suit, side, float(price), self)

            await asyncio.sleep(CHARTIST_TICK)

class BottomFeederPlayer(Player):
    """
    Estimates each suit's value as the average of other players' quotes.
    An agent counts only after k buy orders and k sell orders in that suit.
    Buy and sell are chosen at random. The quote is that value, rounded.
    """

    def _expected_value(self, order_book: OrderBookManager, suit: str) -> float | None:
        """
        Average, over prey, of the mid between that prey's mean buy and mean sell.

        Prey are the other agents with at least k buy orders and k sell orders
        in this suit. Each mean uses that agent's k most recent orders on that side.
        """
        k = BOTTOM_FEEDER_K
        buys: dict[int, list[float]] = {}
        sells: dict[int, list[float]] = {}
        for order in order_book.order_history:
            if order["ASSET"] != suit or order["PLAYER"] == self.id:
                continue
            side_prices = buys if order["SIDE"] == "BUY" else sells
            side_prices.setdefault(order["PLAYER"], []).append(order["PRICE"])

        prey_values = []
        for player_id in set(buys) & set(sells):
            buy_prices = buys[player_id]
            sell_prices = sells[player_id]
            if len(buy_prices) < k or len(sell_prices) < k:
                continue
            mean_buy = sum(buy_prices[-k:]) / k
            mean_sell = sum(sell_prices[-k:]) / k
            prey_values.append(0.5 * (mean_buy + mean_sell))

        if not prey_values:
            return None
        expected = sum(prey_values) / len(prey_values)
        if not np.isfinite(expected):
            return None
        return expected

    async def play(self, order_book: OrderBookManager) -> None:
        np.random.seed(SEED)
        while True:
            prob = np.random.uniform(0, 1)
            side = "BUY" if prob < 0.5 else "SELL"

            values = {}
            for suit in SUITS:
                expected = self._expected_value(order_book, suit)
                if expected is not None:
                    values[suit] = expected
            if values:
                suit = str(np.random.choice(list(values)))
                expected = values[suit]
                price = round(expected)
                can_buy = side == "BUY" and price >= 1 and self.cash >= price
                can_sell = side == "SELL" and price >= 1 and self.get_cards_by_suit(suit) > 0
                if can_buy or can_sell:
                    logger.debug(
                        "Player %s quoting %s %s @ %s | value=%.2f",
                        self.id,
                        side,
                        suit,
                        price,
                        expected,
                    )
                    await order_book.submit_order(suit, side, float(price), self)

            await asyncio.sleep(BOTTOM_FEEDER_TICK)   

class FundamentalistPlayer(Player):
    """
    Estimates each suit from the cards seen in the deal and in later trades.
    Buy and sell use different values. The order price is drawn around the value
    that matches the side, then rounded.
    """

    def _inventory_update(
        self,
        order_book: OrderBookManager,
        inventory: dict[str, dict[int, int]],
        history_obs: int = 0,
    ) -> dict[str, dict[int, int]] | None:
        if len(order_book.order_executed) <= history_obs:
            return None
        for his in order_book.order_executed[history_obs:]:
            asset = his["ASSET"]
            buyer_id = his["BUYER"]
            seller_id = his["SELLER"]
            asset_inventory = inventory.get(asset.lower())
            asset_inventory.setdefault(seller_id, 0)
            asset_inventory.setdefault(buyer_id, 0)
            asset_inventory[seller_id] -= 1 if asset_inventory[seller_id] > 0 else 0
            asset_inventory[buyer_id] += 1
        return inventory

    def _deck_distribution(self, inventory: dict[str, dict[int, int]]) -> list[float]:
        """Posterior over the 12 decks, given how many distinct cards of each suit have been seen."""
        seen = {suit: sum(inventory[suit].values()) for suit in SUITS}
        weights = []
        for deck in DECKS:
            weight = 1
            for suit in SUITS:
                weight *= math.comb(deck["counts"][suit], seen[suit])
            weights.append(weight)
        total = sum(weights)
        if total == 0:
            return [0.0] * len(DECKS)
        return [weight / total for weight in weights]

    def _get_posterior_probability(
        self,
        inventory: dict[str, dict[int, int]],
        suit: str,
        deck_index: int | None = None,
    ) -> float:
        distribution = self._deck_distribution(inventory)
        if deck_index is not None:
            return distribution[deck_index]
        return sum(prob for prob, deck in zip(distribution, DECKS) if deck["goal"] == suit)

    def _card_value(self, deck: dict, suit: str, cards_held: int) -> float:
        """Value of one more card of suit if this deck is the one in play."""
        if deck["goal"] != suit or cards_held < 0:
            return 0.0
        if cards_held >= deck["x"]:
            majority = 0.0
        else:
            growth = FUNDAMENTALIST_R
            base = deck["p"] * (1 - growth) / (1 - growth ** deck["x"])
            majority = base * (growth ** cards_held)
        return CASH_GAIN_PER_GOLDEN_CARD + majority

    def _get_expected_value(
        self,
        inventory: dict[str, dict[int, int]],
        suit: str,
        cards_held: int | None = None,
    ) -> float:
        if cards_held is None:
            cards_held = self.get_cards_by_suit(suit)
        distribution = self._deck_distribution(inventory)
        return sum(
            prob * self._card_value(deck, suit, cards_held)
            for prob, deck in zip(distribution, DECKS)
        )

    async def play(self, order_book: OrderBookManager) -> None:
        np.random.seed(SEED)
        inventory = {suit: {self.id: self.get_cards_by_suit(suit)} for suit in SUITS}
        history_obs = 0
        while True:
            updated = self._inventory_update(order_book, inventory, history_obs)
            if updated is not None:
                inventory = updated
            history_obs = len(order_book.order_executed)

            suit = str(np.random.choice(SUITS))
            held = self.get_cards_by_suit(suit)
            buy_value = self._get_expected_value(inventory, suit, cards_held=held)
            prob = np.random.uniform(0, 1)
            side = "BUY" if prob < 0.5 else "SELL"

            if side == "BUY" and buy_value > 0:
                price = round(float(np.random.uniform(0, buy_value)))
                offer = order_book.books[suit].ask
                if offer is not None:
                    price = min(price, round(offer.order_price))
                if price >= 1 and self.cash >= price:
                    logger.debug(
                        "Player %s quoting %s %s @ %s | value=%.2f",
                        self.id,
                        side,
                        suit,
                        price,
                        buy_value,
                    )
                    await order_book.submit_order(suit, side, float(price), self)
            elif side == "SELL" and held > 0:
                sell_value = self._get_expected_value(inventory, suit, cards_held=held - 1)
                if sell_value > 0:
                    price = round(float(np.random.uniform(sell_value, 2 * sell_value)))
                    bid = order_book.books[suit].bid
                    if bid is not None:
                        price = max(price, round(bid.order_price))
                    if price >= 1:
                        logger.debug(
                            "Player %s quoting %s %s @ %s | value=%.2f",
                            self.id,
                            side,
                            suit,
                            price,
                            sell_value,
                        )
                        await order_book.submit_order(suit, side, float(price), self)

            await asyncio.sleep(FUNDAMENTALIST_TICK)


class FundamentalistPlayer2(FundamentalistPlayer):
    """
    A more directional play approach by combining a little bit of greedy approach.
    """

    async def play(self, order_book: OrderBookManager) -> None:
        """
        Value-maximising Fundamentalist.

        At every decision:
        1. Update known-card inventory from observed trades.
        2. Compute the correct posterior over the 12 possible decks.
        3. Compute buy EV and sell EV for every suit.
        4. Compare each EV with the current best ask / best bid.
        5. Execute the feasible action with the largest positive expected surplus.

        Buy edge:
            buy_value - best_ask

        Sell edge:
            best_bid - sell_value
        """
        inventory = {suit: {self.id: self.get_cards_by_suit(suit)}for suit in SUITS}
        history_obs = 0
        while True:
            updated = self._inventory_update(order_book,inventory,history_obs)
            if updated is not None:
                inventory = updated
            history_obs = len(order_book.order_executed)
            distribution = FundamentalistPlayer._deck_distribution(self,inventory)
            
            if sum(distribution) <= 0:
                await asyncio.sleep(FUNDAMENTALIST_TICK)
                continue

            actions = []
            for suit in SUITS:
                held = self.get_cards_by_suit(suit)
                buy_value = sum(prob * self._card_value(deck,suit,held) for prob, deck in zip(distribution, DECKS))
                ask = order_book.books[suit].ask

                if ask is not None:
                    ask_price = float(ask.order_price)
                    if (
                        buy_value > 0
                        and ask_price >= 1
                        and self.cash >= ask_price
                    ):
                        buy_edge = buy_value - ask_price

                        actions.append({
                            "side": "BUY",
                            "suit": suit,
                            "edge": buy_edge,
                            "price": ask_price,
                            "value": buy_value,
                        })
                if held > 0:
                    sell_value = sum(prob * self._card_value(deck,suit,held - 1) for prob, deck in zip(distribution, DECKS))
                    bid = order_book.books[suit].bid
                    
                    if bid is not None:
                        bid_price = float(bid.order_price)
                        
                        if bid_price >= 1:
                            sell_edge = bid_price - sell_value

                            actions.append({
                                "side": "SELL",
                                "suit": suit,
                                "edge": sell_edge,
                                "price": bid_price,
                                "value": sell_value,
                            })
            profitable_actions = [
                action
                for action in actions
                if action["edge"] > 0
            ]
            if not profitable_actions:
                await asyncio.sleep(FUNDAMENTALIST_TICK)
                continue
            best_action = max(profitable_actions,key=lambda action: action["edge"])

            side = best_action["side"]
            suit = best_action["suit"]
            price = best_action["price"]
            value = best_action["value"]
            edge = best_action["edge"]

            logger.debug(
                (
                    "Player %s VALUE-MAX action | "
                    "%s %s @ %.2f | fundamental_value=%.2f | edge=%.2f"
                ),
                self.id,
                side,
                suit,
                price,
                value,
                edge,
            )
            await order_book.submit_order(
                suit,
                side,
                price,
                self
            )
            await asyncio.sleep(FUNDAMENTALIST_TICK)

class FundamentalistPlayer3(FundamentalistPlayer):
    """
    Directional Fundamentalist.

    Strategy:
    1. Use card counting to calculate posterior probabilities
       over the 12 possible decks.
    2. Convert those deck probabilities into:
           P(suit is goal | observed information)
    3. BUY the suit(s) with maximum goal probability.
    4. SELL the suit(s) with minimum goal probability,
       if currently held.
    5. Buy price is sampled between 1 and (buy_EV - 1).
    6. Sell price is sampled between sell_EV and 2 * sell_EV.
    """

    async def play(self, order_book: OrderBookManager) -> None:
        inventory = {suit: {self.id: self.get_cards_by_suit(suit)} for suit in SUITS}
        history_obs = 0
        while True:
            updated = self._inventory_update(order_book,inventory,history_obs)

            if updated is not None:
                inventory = updated

            history_obs = len(order_book.order_executed)
            distribution = self._deck_distribution(inventory)

            if sum(distribution) <= 0:
                await asyncio.sleep(FUNDAMENTALIST_TICK)
                continue

            goal_probability = {suit: 0.0 for suit in SUITS}

            for prob, deck in zip(distribution, DECKS):
                goal_suit = deck["goal"]

                goal_probability[goal_suit] += prob

            max_probability = max(goal_probability.values())
            min_probability = min(goal_probability.values())

            buy_suits = [
                suit
                for suit in SUITS
                if np.isclose(
                    goal_probability[suit],
                    max_probability
                )
            ]

            sell_suits = [
                suit
                for suit in SUITS
                if np.isclose(
                    goal_probability[suit],
                    min_probability
                )
            ]

            for suit in buy_suits:

                held = self.get_cards_by_suit(suit)

                buy_value = self._get_expected_value(
                    inventory,
                    suit,
                    cards_held=held
                )
                
                if buy_value <= 2:
                    continue

                price = float(
                    np.random.uniform(
                        1,
                        buy_value - 1
                    )
                )

                price = round(price)

                if price < 1:
                    continue

                
                if self.cash < price:
                    continue

                logger.debug(
                    (
                        "Player %s F3 BUY %s @ %s | "
                        "P(goal)=%.4f | "
                        "buy_EV=%.2f | held=%s"
                    ),
                    self.id,
                    suit,
                    price,
                    goal_probability[suit],
                    buy_value,
                    held,
                )

                await order_book.submit_order(
                    suit,
                    "BUY",
                    float(price),
                    self
                )


            for suit in sell_suits:

                held = self.get_cards_by_suit(suit)

                if held <= 0:
                    continue
                
                sell_value = self._get_expected_value(
                    inventory,
                    suit,
                    cards_held=held - 1
                )

                if sell_value <= 0:
                    continue

                price = float(
                    np.random.uniform(
                        sell_value,
                        2 * sell_value
                    )
                )

                price = round(price)

                if price < 1:
                    continue

                logger.debug(
                    (
                        "Player %s F3 SELL %s @ %s | "
                        "P(goal)=%.4f | "
                        "sell_EV=%.2f | held=%s"
                    ),
                    self.id,
                    suit,
                    price,
                    goal_probability[suit],
                    sell_value,
                    held,
                )

                await order_book.submit_order(
                    suit,
                    "SELL",
                    float(price),
                    self
                )

            await asyncio.sleep(FUNDAMENTALIST_TICK)