from typing import Dict, Tuple
import math
import numpy as np
from Deck import Deck

MATTA = 40
FIGURES = [10, 20, 30, 40, 8, 18, 28, 38, 9, 19, 29, 39]

#  1  2  3  4  5  6  7  8  9 10
#  1  2  3  4  5  6  7 .5 .5 .5
class SetteEMezzo:
    def __init__(self):
        self.actions = [0, 1]
        self.deck = Deck()
        self.win = 0
        self.matta_drawn = False
        self.policy = (-1, -1)

    def observation(self) -> Tuple[float, float]:
        return (self.player, self.bust_prob(self.player))

    def start_new_episode(self) -> Tuple[float, float]:
        if len(self.deck.cards) < 15 or self.matta_drawn:
            self.matta_drawn = False
            self.deck.shuffle_deck()

        self.player = self.card_value(self.deck.draw_card())
        self.dealer = self.card_value(self.deck.draw_card())
        self.masked_card = self.player

        self.cards_count = 1
        self.cards_count_dealer = 1

        return self.observation()

    def is_bust(self, cards_value: int) -> bool:
        return cards_value > 7.5

    def bust_prob(self, actual_value):
        max_card_value = math.floor(7.5 - actual_value)
        pos = sum(self.deck.card_count[i] for i in range(1, max_card_value + 1))
        pos += sum(self.deck.card_count[i] for i in range(8, 11))
        tot = sum(self.deck.card_count[i] for i in range(1, 11))

        prob = (tot - pos) / tot
        return round(prob * 100 / 5) * 5

    def card_value(self, card: int, actual_value: float = 0) -> float:
        if card == MATTA:
            self.matta_drawn = True
            return 7 - actual_value if actual_value < 7 else 0.5
        elif card in FIGURES:
            return 0.5
        else:
            return card % 10

    def get_action(self, cards_value_bust_prob: Tuple[Tuple[float, float]]) -> str:
        if cards_value_bust_prob[0] == 0.5:
            return 0  # HIT

        if cards_value_bust_prob[0] == 7.5:
            return 1  # STAY

        if self.policy == (-1, -1):
            return np.random.choice([0, 1])

        return (
            0  # "HIT"
            if (cards_value_bust_prob[0] <= self.policy[0] or self.policy[0] == -1)
            and (cards_value_bust_prob[1] <= self.policy[1] or self.policy[1] == -1)
            else 1  # "STICK"
        )

    def guessing_card(self):
        guessing_list = list(range(math.ceil(7.5 - self.player + self.masked_card)))
        guessing_list.append(0.5)
        guessing_list.remove(0)
        return np.random.choice(guessing_list)

    def step(
        self, action: str, player_bust_reward=-1
    ) -> Tuple[Tuple[float, float], float, bool]:
        if action == 0:
            self.player += self.card_value(self.deck.draw_card())
            self.cards_count += 1
            if self.is_bust(self.player):
                done = True
                reward = player_bust_reward
            else:
                done = False
                reward = 0
        else:
            done = True
            reward = 0
            hidden_card_value = (
                self.guessing_card() if self.player != 7.5 else self.masked_card
            )
            # print(
            #     "Dealer guess: "
            #     + str(self.masked_card)
            #     + " with: "
            #     + str(hidden_card_value)
            # )
            while self.dealer < self.player - self.masked_card + hidden_card_value:
                self.dealer += self.card_value(self.deck.draw_card())
                self.cards_count_dealer += 1

            # player_real_seven_and_half = self.player == 7.5 and self.cards_count == 2
            # dealer_real_seven_and_half = (
            #     self.dealer == 7.5 and self.cards_count_dealer == 2
            # )

            if self.is_bust(self.dealer):
                reward += 1
                self.win += 1

            else:
                if self.dealer >= self.player:
                    reward -= 1
                    # if dealer_real_seven_and_half:
                    #     reward -= 1
                else:
                    reward = 1
                    self.win += 1

            # if player_real_seven_and_half:
            #     reward += 1

        return self.observation(), reward, done

    def run_episode(self):
        states, actions, rewards = [], [], []
        obs = self.start_new_episode()
        while True:
            states.append(obs)

            action = self.get_action(obs)
            actions.append(action)

            obs, reward, is_finished = self.step(action=action)
            rewards.append(reward)

            if is_finished:
                break
        return states, actions, rewards
