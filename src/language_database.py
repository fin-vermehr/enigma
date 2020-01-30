from typing import List, Type

START_SEQUENCE_TOKEN = "0"
START_SEQUENCE_INDEX = 0

END_SEQUENCE_TOKEN = "1"
END_SEQUENCE_INDEX = 1

PAD_TOKEN = '_'
PAD_INDEX = 2


class LanguageDatabase:

    def __init__(self, name: str, records: List[str] = None):
        self.name = name
        self._items = {
            START_SEQUENCE_TOKEN: START_SEQUENCE_INDEX,
            END_SEQUENCE_TOKEN: END_SEQUENCE_INDEX,
            PAD_TOKEN: PAD_INDEX,
        }

        self._inverse_items = {v: k for k, v in self._items.items()}

        if records:
            for text_snippet in records:
                for character in text_snippet:
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
    def pad_token_index(self) -> int:
        return self.get_index(PAD_TOKEN)

    def get_item(self, index: int) -> str:
        return self._inverse_items.get(index)

    def get_index(self, item: str) -> int:
        return self._items.get(item)

    def add_record(self, record: str) -> None:
        for character in record:
            if character not in self._items:
                self._items[character] = self.number_of_items
                self._inverse_items[len(self._inverse_items)] = character
