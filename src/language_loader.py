import logging
from typing import Tuple, List

import torch
from dynaconf import settings

from paths import data_directory_path
from language_database import LanguageDatabase

logger = logging.getLogger(__name__)

class LanguageLoader:

    def __init__(self):
        """
        LanguageLoader is the data pre-processing module. It cleans up the data, splits it into two datasets and
        allows other modules to request data batches where all the characters are indexed.
        """
        lines = open(data_directory_path / 'enc-eng.txt', encoding='utf-8').read().strip().split('\n')

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
        """
        Acts as an endpoint for the engine. Returns indexed cipher batch with its corresponding
        target plain batch.
        """
        cipher_batches = []
        plain_batches = []

        for _ in range(number_of_batches):
            cipher_batch, plain_batch = self._get_batch(batch_size)
            cipher_batches.append(cipher_batch)
            plain_batches.append(plain_batch)

        logger.info(f'Requested {number_of_batches} batches, got {len(cipher_batches)} batches.')

        return cipher_batches, plain_batches

    @staticmethod
    def get_embedding(sentence: str, database: LanguageDatabase) -> torch.Tensor:
        """
        Given a sentence, map each character to its corresponding index and put the indices into a tensor.
        @return: (max_sequence_length x 1)
        """
        # subtract one for the END_SEQUENCE_INDEX
        padding = [settings.PADDING_INDEX] * (settings.MAX_SEQUENCE_LENGTH - 1 - len(sentence))

        index_list = [database.get_index(character) for character in sentence]
        padded_index_list = index_list + [settings.END_SEQUENCE_INDEX] + padding
        return torch.tensor(padded_index_list, dtype=torch.long).view(-1, 1)
