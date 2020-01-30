from typing import List

START_SEQUENCE_TOKEN = "__start"
END_SEQUENCE_TOKEN = "__end"
UNKNOWN_CHARACTER_TOKEN = '__unknown'


class LanguageDatabase:

    def __init__(self, records: List[str] = None):
        self._items = {
            START_SEQUENCE_TOKEN: 0,
            END_SEQUENCE_TOKEN: 1,
            UNKNOWN_CHARACTER_TOKEN: 2,
        }

        self._inverse_items = {v: k for k, v in self._items.items()}

        for item in records:
            for character in item:
                if character not in self._items:
                    self._items[character] = self.number_of_items
                    self._inverse_items[len(self._inverse_items)] = character

    @property
    def number_of_items(self) -> int:
        return len(self._items)

    @property
    def start_token_index(self) -> int:
        return self.get_index(START_SEQUENCE_TOKEN)

    @property
    def end_token_index(self) -> int:
        return self.get_index(END_SEQUENCE_TOKEN)

    @property
    def unknown_token_index(self) -> int:
        return self.get_index(UNKNOWN_CHARACTER_TOKEN)

    def get_index(self, item: str) -> int:
        return self._items.get(item, self._items[UNKNOWN_CHARACTER_TOKEN])

    def get_item(self, index: int) -> str:
        return self._inverse_items.get(index, UNKNOWN_CHARACTER_TOKEN)
