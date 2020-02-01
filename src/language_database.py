from typing import List

from dynaconf import settings


class LanguageDatabase:

    def __init__(self, name: str, records: List[str] = None):
        """
        A database to contain the mapping between characters and their corresponding indices.

        @param name: the name of the database.
        @param records: items to initialize the database with. Can be None.
        """
        self.name = name
        self._items = {
            settings.START_SEQUENCE_TOKEN: settings.START_SEQUENCE_INDEX,
            settings.END_SEQUENCE_TOKEN: settings.END_SEQUENCE_INDEX,
            settings.PAD_TOKEN: settings.PADDING_INDEX,
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
        return self.get_index(settings.START_SEQUENCE_TOKEN)

    @property
    def end_token_index(self) -> int:
        return self.get_index(settings.END_SEQUENCE_TOKEN)

    @property
    def pad_token_index(self) -> int:
        return self.get_index(settings.PAD_TOKEN)

    def get_item(self, index: int) -> str:
        """
        Given an index, return its corresponding character / token
        """
        return self._inverse_items.get(index)

    def get_index(self, item: str) -> int:
        """
        Given a character / token, return its corresponding index
        """
        return self._items.get(item)

    def add_record(self, record: str) -> None:
        """
        Add a single record to the database
        """
        for character in record:
            if character not in self._items:
                self._items[character] = self.number_of_items
                self._inverse_items[len(self._inverse_items)] = character
