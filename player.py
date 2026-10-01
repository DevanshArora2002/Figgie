from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
import logging
from utils import *
from card import Card

from typing import TYPE_CHECKING
import itertools

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union
    from order_book import OrderBookManager

logger = logging.getLogger(__name__)

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

