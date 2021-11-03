import random
from numpy.random.mtrand import rand


class Deck:
    def __init__(self):
        self.seed = 0
        self.shuffle_deck()

    def draw_card(self, card_to_drawn=0) -> int:
        if card_to_drawn != 0:
            self.cards.remove(card_to_drawn)
            card = card_to_drawn
        else:
            r = random.randint(0, len(self.cards) - 1)
            card = self.cards.pop(r)

        if card % 10 == 0:
            self.card_count[10] -= 1
        else:
            self.card_count[card % 10] -= 1

        return card

    def shuffle_deck(self):
        # self.seed += 1
        # random.seed(self.seed)
        self.cards = [card for card in range(1, 41)]
        random.shuffle(self.cards)
        self.card_count = dict.fromkeys([i for i in range(1, 11)], 4)
