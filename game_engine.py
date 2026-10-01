from __future__ import annotations
import asyncio
import logging
import time
from typing import TYPE_CHECKING
from utils import *
from player import Player, GreedyPlayer
import random
from card import Card
from order_book import OrderBookManager

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

CLASS_REGISTRY = {"greedy": GreedyPlayer}



class GameEngine:
    def __init__(self,player1="greedy",player2="greedy",player3="greedy",player4="greedy",logging_file: str='Sample_logging_Greedy.txt', log_file: str = "figgie.log"):
        setup_logging(log_file=log_file)
        self.rng = random.Random(SEED)
        self.players_list: List[Player] = self._initialize_players(player1,player2,player3,player4)
        self.num_rounds = NUM_ROUNDS
        self.current_round = 0
        self.pot = 0
        self.logging_file = logging_file
        logger.info(
            "Engine ready | classes=%s | rounds=%s | seed=%s | trade_report=%s | debug_log=%s",
            [player1, player2, player3, player4],
            self.num_rounds,
            SEED,
            logging_file,
            log_file,
        )

    def initialize_game(self):
        self.rng = random.Random(SEED * self.current_round)
        for player in self.players_list:
            player.cards_lst.clear()
        logger.debug("Cleared hands before deal")
        self.rng.shuffle(self.players_list)
        self.cards_set, self.gold_suite = self._initialize_cards_set()
        self.golden_suite_count = self.card_to_count_dict[self.gold_suite]
        self.rng.shuffle(self.cards_set)
        for i, card in enumerate(self.cards_set):
            self.players_list[i % len(self.players_list)].add_card(card)
        self.order_book = self._initialize_order_book()
        logger.info(
            "Round %s dealt | counts=%s | golden_suit=%s (%s cards) | seats=%s",
            self.current_round,
            self.card_to_count_dict,
            self.gold_suite,
            self.golden_suite_count,
            [player.id for player in self.players_list],
        )
        for player in self.players_list:
            logger.info("Round %s hand | %s", self.current_round, player)

    def _initialize_players(self,player1: str,player2: str,player3: str,player4: str) -> List[Player]:
        return [self._create_instance(player1),self._create_instance(player2),self._create_instance(player3),self._create_instance(player4)]

    def _create_instance(self,class_key: str, *args, **kwargs) -> Player:
        cls = CLASS_REGISTRY.get(class_key.lower())
        if cls is None:
            logger.error("Unknown player class '%s' | available=%s", class_key, list(CLASS_REGISTRY.keys()))
            raise ValueError(
                f"Unknown key '{class_key}'. Available: {list(CLASS_REGISTRY.keys())}"
            )
        player = cls(*args, **kwargs)
        logger.debug("Created %s as player %s", class_key, player.id)
        return player

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
        logger.debug(
            "Suit layout | counts=%s | twelve_suit=%s | golden_suit=%s",
            self.card_to_count_dict,
            twelve_suit,
            golden_suite,
        )
        cards = [
            Card(suit, suit_to_color[suit])
            for suit in SUITS
            for _ in range(self.card_to_count_dict[suit])
        ]
        return cards, golden_suite

    def _initialize_order_book(self) -> OrderBookManager:
        return OrderBookManager(assets=SUITS,logging_file=self.logging_file)

    async def _play_round(self,round_number: int):
        self.pot = 0
        for player in self.players_list:
            player.give_boot(BOOT_AMOUNT)
            self.pot += BOOT_AMOUNT
        logger.info("Round %s ante collected | boot=%s | pot=%s", round_number, BOOT_AMOUNT, self.pot)
        await self.order_book.start(round_number)
        tasks = [
            asyncio.create_task(player.play(self.order_book))
            for player in self.players_list
        ]
        logger.info(
            "Round %s market open | players=%s | duration=%ss",
            round_number,
            [player.id for player in self.players_list],
            ROUND_TIME,
        )
        started = time.monotonic()
        await asyncio.sleep(ROUND_TIME)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for player, result in zip(self.players_list, results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.error("Player %s failed | %s", player.id, result, exc_info=result)
        logger.info("Round %s market closed after %.2fs", round_number, time.monotonic() - started)
        logger.debug("Round %s waiting for the order queue to drain", round_number)
        await self.order_book.order_queue.join()
        logger.debug("Round %s order queue drained", round_number)
        await self.order_book.stop()
    
    def _check_game_over(self):
        for player in self.players_list:
            if player.cash < BOOT_AMOUNT:
                logger.warning(
                    "Player %s cannot pay boot | cash=%.2f | boot=%s",
                    player.id,
                    player.cash,
                    BOOT_AMOUNT,
                )
                return True
        return False

    async def play_game(self):
        logger.info("Game started | rounds=%s | starting_cash=%s", self.num_rounds, STARTING_CASH)
        while self.current_round < self.num_rounds:
            if self._check_game_over():
                break
            self.current_round += 1
            logger.info("Starting round %s of %s", self.current_round, self.num_rounds)
            self.initialize_game()
            await self._play_round(self.current_round)
            self._post_round_processing()
        logger.info("Game finished after %s round(s) | standings %s", self.current_round, self._standings())

    def _standings(self) -> str:
        ranked = sorted(self.players_list, key=lambda player: player.cash, reverse=True)
        return " ".join(f"p{player.id}={player.cash:.2f}" for player in ranked)

    def _post_round_processing(self):
        current_boot = self.pot
        winners = []
        max_count = -1
        logger.info(
            "Round %s settlement | pot=%s | golden_suit=%s",
            self.current_round,
            self.pot,
            self.gold_suite,
        )
        for player in self.players_list:
            golden_count = player.get_cards_by_suit(self.gold_suite)
            bonus = golden_count * CASH_GAIN_PER_GOLDEN_CARD
            player.get_cash(bonus)
            current_boot -= bonus
            logger.info(
                "Round %s bonus | player=%s | golden_cards=%s | bonus=%s | cash=%.2f",
                self.current_round,
                player.id,
                golden_count,
                bonus,
                player.cash,
            )
            if golden_count > max_count:
                max_count = golden_count
                winners = [player]
            elif golden_count == max_count:
                winners.append(player)
        share = current_boot / len(winners)
        for winner in winners:
            winner.get_cash(share)
        logger.info(
            "Round %s pot awarded | winners=%s | most_golden=%s | share=%.2f | remainder_pot=%.2f",
            self.current_round,
            [winner.id for winner in winners],
            max_count,
            share,
            current_boot,
        )
        logger.info("Round %s standings | %s", self.current_round, self._standings())


if __name__ == "__main__":
    game_engine = GameEngine()
    asyncio.run(game_engine.play_game())
