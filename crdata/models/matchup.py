"""Neural matchup model for Clash Royale battles.

`MatchupModel` predicts the log-odds that side A beats side B as the sum of four
terms, each of which is antisymmetric on its own:

    logit P(A wins) = skill + investment + transitive deck strength + counters

Antisymmetry is structural rather than learned. Swapping the two sides negates
every term, so the model cannot assign both players a winning probability. The
random side-swapping used with concatenated encodings is therefore unnecessary.

Two pieces go beyond an additive card-strength model:

`DeckEncoder` sum-pools card embeddings and then applies an MLP, which lets the
model represent synergy. A model that sums a per-card strength cannot express
"these two cards are worth more together than apart"; applying a nonlinearity
after pooling can.

`MatchupModel.counter_term` scores a pair of decks with an antisymmetric bilinear
form, which lets the model represent counter relationships. A single strength
number per deck forces a total order and cannot represent a cycle such as
A beating B, B beating C, and C beating A.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LogitComponents:
    """Each additive contribution to the predicted log-odds, kept separate."""

    skill: Tensor
    investment: Tensor
    deck: Tensor
    counters: Tensor

    def total(self) -> Tensor:
        return self.skill + self.investment + self.deck + self.counters


class OddFunction(nn.Module):
    """Wraps an MLP so the result is guaranteed odd: f(-x) equals -f(x).

    Used for the investment term so card level can act nonlinearly while the
    model stays exactly antisymmetric.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) - self.net(-x)


class DeckEncoder(nn.Module):
    """Map an unordered set of card ids to a deck vector.

    Sum-pooling before the MLP makes the encoder permutation invariant, which is
    the correct inductive bias because a deck is a set and slot order is
    meaningless.
    """

    def __init__(self, n_cards: int, embed_dim: int = 48, hidden: int = 96) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(n_cards, embed_dim)
        self.synergy = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        nn.init.normal_(self.card_embedding.weight, std=0.05)

    def forward(self, deck: Tensor) -> Tensor:
        pooled = self.card_embedding(deck).sum(dim=1)
        return self.synergy(pooled)


class MatchupModel(nn.Module):
    def __init__(self, n_cards: int, n_players: int,
                 embed_dim: int = 48, hidden: int = 96, counter_rank: int = 8) -> None:
        super().__init__()
        self.encoder = DeckEncoder(n_cards, embed_dim, hidden)
        self.strength_head = nn.Linear(hidden, 1, bias=False)
        self.blade_head = nn.Linear(hidden, counter_rank, bias=False)
        self.chest_head = nn.Linear(hidden, counter_rank, bias=False)
        self.player_skill = nn.Embedding(n_players, 1)
        self.investment = OddFunction()
        nn.init.zeros_(self.player_skill.weight)

    def counter_term(self, deck_a: Tensor, deck_b: Tensor) -> Tensor:
        """Antisymmetric pair score. Swapping the arguments negates the result.

        Written as the blade-chest inner product of Chen and Joachims (2016):
        whatever advantage deck A holds over deck B, deck B holds the exact
        negative against deck A, and no deck counters itself.
        """
        blade_a, chest_a = self.blade_head(deck_a), self.chest_head(deck_a)
        blade_b, chest_b = self.blade_head(deck_b), self.chest_head(deck_b)
        return ((blade_a * chest_b).sum(-1) - (blade_b * chest_a).sum(-1)).unsqueeze(-1)

    def components(self, cards_a: Tensor, cards_b: Tensor,
                   player_a: Tensor, player_b: Tensor,
                   level_a: Tensor, level_b: Tensor) -> LogitComponents:
        deck_a, deck_b = self.encoder(cards_a), self.encoder(cards_b)
        return LogitComponents(
            skill=self.player_skill(player_a) - self.player_skill(player_b),
            investment=self.investment((level_a - level_b).unsqueeze(-1)),
            deck=self.strength_head(deck_a) - self.strength_head(deck_b),
            counters=self.counter_term(deck_a, deck_b))

    def forward(self, cards_a: Tensor, cards_b: Tensor,
                player_a: Tensor, player_b: Tensor,
                level_a: Tensor, level_b: Tensor) -> Tensor:
        return self.components(
            cards_a, cards_b, player_a, player_b, level_a, level_b).total().squeeze(-1)
