from __future__ import annotations
import asyncio
import time
from typing import TYPE_CHECKING
from utils import *
from player import Player, GreedyPlayer
import random
from card import Card
from order_book import OrderBookManager

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

CLASS_REGISTRY = {"greedy": GreedyPlayer}



class GameEngine:
    def __init__(self,player1="greedy",player2="greedy",player3="greedy",player4="greedy"):
        self.rng = random.Random(SEED)
        self.players_list: List[Player] = self._initialize_players(player1,player2,player3,player4)
        self.num_rounds = NUM_ROUNDS
        self.current_round = 0
        self.pot = 0

    def initialize_game(self):
        self.rng = random.Random(SEED * self.current_round)
        for player in self.players_list:
            player.cards_lst.clear()
        self.rng.shuffle(self.players_list)
        self.cards_set, self.gold_suite = self._initialize_cards_set()
        self.golden_suite_count = self.card_to_count_dict[self.gold_suite]
        self.rng.shuffle(self.cards_set)
        for i, card in enumerate(self.cards_set):
            self.players_list[i % len(self.players_list)].add_card(card)
        self.order_book = self._initialize_order_book()

    def _initialize_players(self,player1: str,player2: str,player3: str,player4: str) -> List[Player]:
        return [self._create_instance(player1),self._create_instance(player2),self._create_instance(player3),self._create_instance(player4)]

    def _create_instance(self,class_key: str, *args, **kwargs) -> Player:
        cls = CLASS_REGISTRY.get(class_key.lower())
        if cls is None:
            raise ValueError(
                f"Unknown key '{class_key}'. Available: {list(CLASS_REGISTRY.keys())}"
            )
        return cls(*args, **kwargs)

    def _initialize_cards_set(self) -> tuple[list[Card], str]:
        shuffled_counts = NO_CARDS[:]
        self.rng.shuffle(shuffled_counts)
        self.card_to_count_dict = {suit: count for suit, count in zip(SUITS, shuffled_counts)}
        twelve_suit = next(suit for suit, count in self.card_to_count_dict.items() if count == 12)
        twelve_color = suit_to_color[twelve_suit]
        golden_suite = next(
            suit for suit in SUITS
            if suit != twelve_suit and suit_to_color[suit] == twelve_color
        )
        cards = [
            Card(suit, suit_to_color[suit])
            for suit in SUITS
            for _ in range(self.card_to_count_dict[suit])
        ]
        return cards, golden_suite

    def _initialize_order_book(self) -> OrderBookManager:
        return OrderBookManager(assets=SUITS)

    async def _play_round(self):
        self.pot = 0
        for player in self.players_list:
            player.give_boot(BOOT_AMOUNT)
            self.pot += BOOT_AMOUNT
        await self.order_book.start()
        tasks = [
            asyncio.create_task(player.play(self.order_book))
            for player in self.players_list
        ]
        await asyncio.sleep(ROUND_TIME)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.order_book.order_queue.join()
        await self.order_book.stop()
    
    def _check_game_over(self):
        for player in self.players_list:
            if player.cash < BOOT_AMOUNT:
                return True
        return False

    async def play_game(self):
        while self.current_round < self.num_rounds:
            if self._check_game_over():
                break
            self.current_round += 1
            self.initialize_game()
            await self._play_round()
            self._post_round_processing()

    def _post_round_processing(self):
        current_boot = self.pot
        winners = []
        max_count = -1
        for player in self.players_list:
            golden_count = player.get_cards_by_suit(self.gold_suite)
            bonus = golden_count * CASH_GAIN_PER_GOLDEN_CARD
            player.get_cash(bonus)
            current_boot -= bonus
            if golden_count > max_count:
                max_count = golden_count
                winners = [player]
            elif golden_count == max_count:
                winners.append(player)
        share = current_boot / len(winners)
        for winner in winners:
            winner.get_cash(share)
