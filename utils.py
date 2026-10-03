import logging
import os
from pathlib import Path

STARTING_CASH = 350
BOOT_AMOUNT = 50
SEED = 42
CASH_GAIN_PER_GOLDEN_CARD = 10
NUM_ROUNDS = 4
SUITS = ["heart", "diamond", "club", "spade"]
NO_CARDS = [8, 12, 10, 10]
COLORS = ["red", "black"]

suit_to_color = {
    "heart": "red",
    "diamond": "red",
    "club": "black",
    "spade": "black"
}
color_to_suit = {
    "red": ["heart", "diamond"],
    "black": ["club", "spade"]
}

# Round Time in Seconds
ROUND_TIME = 2 * 60

# Random player constants
RANDOM_TICK = 0.1
RANDOM_MAX_BID = 50
RANDOM_MIN_OFFER = 1

# Greedy player constants
MAX_BID = 9
MIN_OFFER = 1
GREEDY_START_BID = 1
GREEDY_START_OFFER = 10
GREEDY_TICK = 0.1

# Chartist player constants
CHARTIST_TICK = 0.1
# Trades looked back when averaging a suit's return. Quoting needs this many
# plus the current trade and the trade just before it.
CHARTIST_TAU = 5

# Bottom-feeder player constants
BOTTOM_FEEDER_TICK = 0.1
# Buy orders and sell orders required from an agent before it counts as prey.
BOTTOM_FEEDER_K = 4

# Fundamentalist player constants
FUNDAMENTALIST_TICK = 0.1
# Growth of a card's majority value as the player nears a majority.
FUNDAMENTALIST_R = 1.5


def setup_logging(log_file: str = "figgie.log", console_level: int | None = None, file_level: int = logging.DEBUG) -> None:
    """Send INFO (or FIGGIE_LOG_LEVEL) to the console and DEBUG to log_file."""
    if console_level is None:
        level_name = os.environ.get("FIGGIE_LOG_LEVEL", "INFO").upper()
        console_level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(getattr(handler, "_figgie_console", False) for handler in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(formatter)
        console._figgie_console = True
        root.addHandler(console)

    if not log_file:
        return

    log_path = str(Path(log_file).resolve())
    already_writing = any(
        getattr(handler, "_figgie_file", False) and getattr(handler, "baseFilename", None) == log_path
        for handler in root.handlers
    )
    if already_writing:
        return

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    file_handler._figgie_file = True
    root.addHandler(file_handler)

