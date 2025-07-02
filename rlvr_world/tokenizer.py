"""Tokenisation for the toy text world.

A world state is a short text string such as ``"r3 c5 door open lamp off gem yes"``.
The tokeniser is a greedy longest-match scheme over a small set of keyword pieces
plus single characters.  Because keywords can always be spelled out as characters,
the *same* string has several tokenisations, e.g.::

    encode("door open")  -> [<door>, ' ', <open>]
    encode_charwise(...) -> ['d', 'o', 'o', 'r', ' ', 'o', 'p', 'e', 'n']

Both decode to the same string.  That is deliberate: it makes the property the
paper cares about -- the reward is a function of the *decoded* prediction, not of
the token ids -- directly testable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

PAD = "<pad>"
BOS = "<bos>"
SEP = "<sep>"
EOS = "<eos>"
UNK = "<unk>"

SPECIALS: tuple[str, ...] = (PAD, BOS, SEP, EOS, UNK)

#: Multi-character pieces the greedy tokeniser prefers.
KEYWORDS: tuple[str, ...] = (
    "north",
    "south",
    "east",
    "west",
    "door",
    "lamp",
    "gem",
    "open",
    "shut",
    "on",
    "off",
    "yes",
    "no",
    "act",
)

#: Characters that can appear in a state or question string.
ALPHABET: str = "abcdefghijklmnopqrstuvwxyz0123456789 "


class Tokenizer:
    """Greedy longest-match tokeniser over ``SPECIALS + ALPHABET + KEYWORDS``."""

    def __init__(
        self,
        keywords: Sequence[str] = KEYWORDS,
        alphabet: str = ALPHABET,
    ) -> None:
        # Longest first so that "north" wins over "no" and so on.
        self.keywords: tuple[str, ...] = tuple(sorted(keywords, key=len, reverse=True))

        pieces: list[str] = list(SPECIALS) + sorted(set(alphabet)) + list(self.keywords)
        seen: set[str] = set()
        self.itos: list[str] = []
        for piece in pieces:
            if piece not in seen:
                seen.add(piece)
                self.itos.append(piece)
        self.stoi: dict[str, int] = {piece: i for i, piece in enumerate(self.itos)}

        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.sep_id = self.stoi[SEP]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    # ------------------------------------------------------------------ encode
    def encode(self, text: str, use_keywords: bool = True) -> list[int]:
        """Tokenise ``text``; with ``use_keywords=False`` fall back to characters."""
        ids: list[int] = []
        i = 0
        while i < len(text):
            piece = None
            if use_keywords:
                for candidate in self.keywords:
                    if text.startswith(candidate, i):
                        piece = candidate
                        break
            if piece is None:
                char = text[i]
                if char not in self.stoi:
                    raise ValueError(f"character {char!r} is not in the vocabulary")
                piece = char
            ids.append(self.stoi[piece])
            i += len(piece)
        return ids

    def encode_charwise(self, text: str) -> list[int]:
        """The alternative tokenisation of the same string, one id per character."""
        return self.encode(text, use_keywords=False)

    # ------------------------------------------------------------------ decode
    def decode(self, ids: Iterable[int], keep_specials: bool = False) -> str:
        """Invert :meth:`encode`.  Special pieces are dropped by default."""
        out: list[str] = []
        for idx in ids:
            piece = self.itos[int(idx)]
            if piece in SPECIALS:
                if keep_specials:
                    out.append(piece)
                continue
            out.append(piece)
        return "".join(out)

    # ------------------------------------------------------------- conventions
    def encode_prompt(self, text: str) -> list[int]:
        """``<bos> text <sep>`` -- the question part of a transition example."""
        return [self.bos_id] + self.encode(text) + [self.sep_id]

    def encode_response(self, text: str) -> list[int]:
        """``text <eos>`` -- the response part of a transition example."""
        return self.encode(text) + [self.eos_id]

    def text_piece_ids(self) -> list[int]:
        """Ids that are neither special nor a single space (free-text characters)."""
        return [
            i
            for i, piece in enumerate(self.itos)
            if piece not in SPECIALS and piece != " "
        ]
