import itertools
from typing import Tuple, List

import torch
from dynaconf import settings
from nlp_takehome.src.language_database import LanguageDatabase

import logging
logger = logging.getLogger(__name__)

PAIR_CIPHER_INDEX = 0
PAIR_PLAIN_INDEX = 1


class LanguageLoader:

    def __init__(self):
        lines = open('data/enc-eng.txt', encoding='utf-8').read().strip().split('\n')
        self.pairs = [[text_snippet for text_snippet in line.split('\t')] for line in lines]

        self.cipher_database = LanguageDatabase('cipher', [pair[0] for pair in self.pairs])
        self.plain_database = LanguageDatabase('plain', [pair[1] for pair in self.pairs])
        self.data_pointer = 0

        logger.info(f'Created the databases. The cipher database'
                    f'contains {self.cipher_database.number_of_items} records,'
                    f'the plain database contains {self.plain_database.number_of_items} records.')

    def _get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cipher_batch = []
        plain_batch = []

        for _ in range(batch_size):
            self.data_pointer += 1
            cipher_sentence, plain_sentence = self.pairs[self.data_pointer]
            cipher_batch.append(self.get_embedding(cipher_sentence, self.cipher_database))
            plain_batch.append(self.get_embedding(plain_sentence, self.plain_database))

        # Return (Max_Len x Batch Size)
        return torch.stack(cipher_batch).squeeze(2).permute(1, 0), torch.stack(plain_batch).squeeze(2).permute(1, 0)

    def get_batches(self, number_of_batches: int, batch_size: int) -> Tuple[List, List]:
        cipher_batches = []
        plain_batches = []

        for _ in range(number_of_batches):
            cipher_batch, plain_batch = self._get_batch(batch_size)
            cipher_batches.append(cipher_batch)
            plain_batches.append(plain_batch)

        return cipher_batches, plain_batches

    @staticmethod
    def get_embedding(sentence: str, database: LanguageDatabase) -> torch.Tensor:
        # subtract one for the END_SEQUENCE_INDEX
        padding = [settings.PADDING_INDEX] * (42 - 1 - len(sentence))

        index_list = [database.get_index(character) for character in sentence]
        padded_index_list = index_list + [settings.END_SEQUENCE_INDEX] + padding
        return torch.tensor(padded_index_list, dtype=torch.long).view(-1, 1)

