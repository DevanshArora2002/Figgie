from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class Card:
    def __init__(self,suit: str,color: str):
        self.suit = suit
        self.color = color

