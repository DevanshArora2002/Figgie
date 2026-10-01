from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from card import Card
    from player import Player


class OrderStatus(Enum):
    PENDING = auto()
    RESTING = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


class RejectReason(Enum):
    WORSE_PRICE = auto()
    NO_CARD = auto()
    NO_CASH = auto()
    SELF_TRADE = auto()


@dataclass
class OrderHandle:
    """Receipt for a submitted order. Only the matching worker mutates it."""
    order_id: str
    asset: str
    side: str  # "BUY" or "SELL"
    price: float
    party: Player
    status: OrderStatus = OrderStatus.PENDING
    reason: Optional[RejectReason] = None
    trade: Optional[dict] = None
    _done: Optional[asyncio.Future] = field(default=None)

    def _get_future(self) -> asyncio.Future:
        """Lazily create the future when needed."""
        if self._done is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            self._done = loop.create_future()
        return self._done

    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)

    async def wait(self) -> OrderHandle:
        """Block until the handle reaches a terminal state."""
        if self.is_terminal():
            return self
        await self._get_future()
        return self

    def _resolve(self, status: OrderStatus, reason: Optional[RejectReason] = None, trade: Optional[dict] = None) -> None:
        self.status = status
        self.reason = reason
        self.trade = trade
        if self.is_terminal() and self._done is not None and not self._done.done():
            self._done.set_result(self)


@dataclass
class Order:
    """A resting quote in the book."""
    order_type: str
    order_price: float
    order_party: Player
    order_id: str
    handle: OrderHandle


class OrderBook:
    """One displayed bid and one displayed offer for a single suit."""

    def __init__(self, asset: str):
        self.asset = asset
        self.bid: Optional[Order] = None
        self.ask: Optional[Order] = None

    def clear(self) -> list[OrderHandle]:
        """Clear all quotes and return their handles for cancellation."""
        handles = []
        if self.bid is not None:
            handles.append(self.bid.handle)
            self.bid = None
        if self.ask is not None:
            handles.append(self.ask.handle)
            self.ask = None
        return handles

    def process_order(self, handle: OrderHandle) -> dict:
        """
        Process an incoming order.
        Returns a result dict with keys:
          - "outcome": "trade", "rested", or "rejected"
          - "trade": trade dict if outcome is "trade"
          - "reason": RejectReason if outcome is "rejected"
          - "resting_handle": handle of the resting order that was filled (if any)
          - "stale_handle": handle of a stale resting order that was removed (if any)
          - "stale_reason": RejectReason for the stale order (if any)
        """
        if handle.side == "BUY":
            return self._process_buy(handle)
        if handle.side == "SELL":
            return self._process_sell(handle)
        return {"outcome": "rejected", "reason": RejectReason.WORSE_PRICE}

    def _process_buy(self, handle: OrderHandle) -> dict:
        price = handle.price
        party = handle.party

        # Check if we can cross with the ask
        if self.ask is not None and price >= self.ask.order_price:
            # Self-trade check
            if party is self.ask.order_party:
                return {"outcome": "rejected", "reason": RejectReason.SELF_TRADE}

            # Check if resting seller still has a card
            if not self._holds_suit(self.ask.order_party):
                stale_handle = self.ask.handle
                self.ask = None
                # Try to rest the buy instead
                rest_result = self._try_rest_bid(handle)
                rest_result["stale_handle"] = stale_handle
                rest_result["stale_reason"] = RejectReason.NO_CARD
                return rest_result

            # Check if buyer can pay
            if not self._can_pay(party, self.ask.order_price):
                return {"outcome": "rejected", "reason": RejectReason.NO_CASH}

            # Execute trade
            trade = self._execute_trade(
                buyer=party,
                seller=self.ask.order_party,
                price=self.ask.order_price,
                order_id=handle.order_id,
            )
            resting_handle = self.ask.handle
            self.bid = None
            self.ask = None
            return {"outcome": "trade", "trade": trade, "resting_handle": resting_handle}

        # Try to rest the bid
        return self._try_rest_bid(handle)

    def _process_sell(self, handle: OrderHandle) -> dict:
        price = handle.price
        party = handle.party

        # Check if we can cross with the bid
        if self.bid is not None and price <= self.bid.order_price:
            # Self-trade check
            if party is self.bid.order_party:
                return {"outcome": "rejected", "reason": RejectReason.SELF_TRADE}

            # Check if resting buyer can still pay
            if not self._can_pay(self.bid.order_party, self.bid.order_price):
                stale_handle = self.bid.handle
                self.bid = None
                # Try to rest the sell instead
                rest_result = self._try_rest_ask(handle)
                rest_result["stale_handle"] = stale_handle
                rest_result["stale_reason"] = RejectReason.NO_CASH
                return rest_result

            # Check if seller has a card
            if not self._holds_suit(party):
                return {"outcome": "rejected", "reason": RejectReason.NO_CARD}

            # Execute trade
            trade = self._execute_trade(
                buyer=self.bid.order_party,
                seller=party,
                price=self.bid.order_price,
                order_id=handle.order_id,
            )
            resting_handle = self.bid.handle
            self.bid = None
            self.ask = None
            return {"outcome": "trade", "trade": trade, "resting_handle": resting_handle}

        # Try to rest the ask
        return self._try_rest_ask(handle)

    def _try_rest_bid(self, handle: OrderHandle) -> dict:
        price = handle.price
        party = handle.party

        # Check if buyer can pay
        if not self._can_pay(party, price):
            return {"outcome": "rejected", "reason": RejectReason.NO_CASH}

        # Check if price is strictly better than current bid
        if self.bid is not None and price <= self.bid.order_price:
            return {"outcome": "rejected", "reason": RejectReason.WORSE_PRICE}

        # Rest the bid
        self.bid = Order("BUY", price, party, handle.order_id, handle)
        return {"outcome": "rested"}

    def _try_rest_ask(self, handle: OrderHandle) -> dict:
        price = handle.price
        party = handle.party

        # Check if seller has a card
        if not self._holds_suit(party):
            return {"outcome": "rejected", "reason": RejectReason.NO_CARD}

        # Check if price is strictly better than current ask
        if self.ask is not None and price >= self.ask.order_price:
            return {"outcome": "rejected", "reason": RejectReason.WORSE_PRICE}

        # Rest the ask
        self.ask = Order("SELL", price, party, handle.order_id, handle)
        return {"outcome": "rested"}

    def _execute_trade(self, buyer: Player, seller: Player, price: float, order_id: str) -> dict:
        """Execute a trade, moving card and cash."""
        # Import here to avoid circular import
        from card import Card

        card = self._find_card(seller)
        seller.remove_card(card)
        buyer.add_card(card)
        buyer.cash -= price
        seller.cash += price

        return {
            "id": order_id,
            "buyer": buyer,
            "seller": seller,
            "price": price,
            "asset": self.asset,
        }

    def _find_card(self, player: Player) -> Optional[Card]:
        for card in player.cards_lst:
            if card.suit == self.asset:
                return card
        return None

    def _holds_suit(self, player: Player) -> bool:
        return self._find_card(player) is not None

    def _can_pay(self, player: Player, price: float) -> bool:
        return player.cash >= price


class OrderBookManager:
    def __init__(self, assets: list[str]):
        self.books: Dict[str, OrderBook] = {asset: OrderBook(asset) for asset in assets}
        self.order_queue: asyncio.Queue = asyncio.Queue()
        self.order_executed: list[dict] = []
        self._worker: Optional[asyncio.Task] = None
        self._order_id_counter = itertools.count(1)

    async def submit_order(self, asset: str, side: str, price: float, party: Player) -> OrderHandle:
        """Submit an order and return a handle to track its outcome."""
        order_id = f"order-{next(self._order_id_counter)}"
        handle = OrderHandle(
            order_id=order_id,
            asset=asset,
            side=side,
            price=price,
            party=party,
        )
        await self.order_queue.put(handle)
        return handle

    def _cancel_all_quotes(self) -> None:
        """Cancel all resting quotes in all books."""
        for book in self.books.values():
            handles = book.clear()
            for h in handles:
                if not h.is_terminal():
                    h._resolve(OrderStatus.CANCELLED)

    async def _matching_engine_worker(self) -> None:
        while True:
            item = await self.order_queue.get()
            try:
                if item is None:
                    return

                handle: OrderHandle = item
                asset = handle.asset
                book = self.books[asset]

                result = book.process_order(handle)

                # Handle stale resting orders
                if "stale_handle" in result:
                    stale = result["stale_handle"]
                    stale._resolve(OrderStatus.REJECTED, result["stale_reason"])

                outcome = result["outcome"]

                if outcome == "trade":
                    trade = result["trade"]
                    resting_handle = result.get("resting_handle")

                    # Mark both handles as FILLED
                    handle._resolve(OrderStatus.FILLED, trade=trade)
                    if resting_handle:
                        resting_handle._resolve(OrderStatus.FILLED, trade=trade)

                    # Cancel all other resting quotes
                    self._cancel_all_quotes()

                    # Log the trade
                    self.order_executed.append({
                        "ID": trade["id"],
                        "BUYER": trade["buyer"].id,
                        "SELLER": trade["seller"].id,
                        "PRICE": trade["price"],
                        "ASSET": asset,
                    })

                elif outcome == "rested":
                    handle._resolve(OrderStatus.RESTING)

                elif outcome == "rejected":
                    handle._resolve(OrderStatus.REJECTED, result["reason"])

            finally:
                self.order_queue.task_done()

    async def start(self) -> None:
        await self.stop()
        self._worker = asyncio.create_task(self._matching_engine_worker())

    async def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        await self.order_queue.put(None)
        await self.order_queue.join()
        await worker
        self._worker = None
