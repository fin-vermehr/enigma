from typing import Tuple, List

import torch
from dynaconf import settings
from nlp_takehome.src.language_database import LanguageDatabase, END_SEQUENCE_INDEX

import logging
logger = logging.getLogger(__name__)

PAIR_CIPHER_INDEX = 0
PAIR_PLAIN_INDEX = 1
PADDING_INDEX = 2

class LanguageLoader:

    def __init__(self):
        self.cipher_database, self.plain_database, self.pairs = self._get_language_databases()

    @staticmethod
    def _get_language_databases() -> Tuple[LanguageDatabase, LanguageDatabase, List]:

        # TODO: Change this
        lines = open('data/enc-eng.txt', encoding='utf-8').read().strip().split('\n')
        pairs = [[text_snippet for text_snippet in line.split('\t')] for line in lines]

        cipher_database = LanguageDatabase('cipher', [pair[0] for pair in pairs])
        plain_database = LanguageDatabase('plain', [pair[1] for pair in pairs])

        logger.info(f'Created the databases. The cipher database'
                    f'contains {cipher_database.number_of_items} records,'
                    f'the plain database contains {plain_database.number_of_items} records.')

        return cipher_database, plain_database, pairs

    # #TODO: Rename
    def get_embed_pairs(self, number_of_iterations, device):
        input_output_pair = []

        for index in range(number_of_iterations):
            pair = self.pairs[index]

            input_tensor = self.get_embedding(pair[0], self.cipher_database)
            target_tensor = self.get_embedding(pair[1], self.plain_database)

            input_output_pair.append((input_tensor, target_tensor))

        return input_output_pair

    def get_embedding(self, sentence, database, device='cuda'):
        # subtract one for the END_SEQUENCE_INDEX
        padding = [database.pad_token_index] * (settings.MAX_SEQUENCE_LENGTH - len(sentence) - 1)

        index_list = [database.get_index(character) for character in sentence]
        padded_index_list = index_list + [END_SEQUENCE_INDEX] + padding
        return torch.tensor(padded_index_list, dtype=torch.long, device=device).view(-1, 1)

