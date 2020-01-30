from typing import Tuple, List

import torch

from nlp_takehome.src.version_two.language_database import LanguageDatabase, END_SEQUENCE_INDEX

import logging
logger = logging.getLogger(__name__)

PAIR_CIPHER_INDEX = 0
PAIR_PLAIN_INDEX = 1


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

            input = [self.cipher_database.get_index(character) for character in pair[0]] + [END_SEQUENCE_INDEX]
            target = [self.plain_database.get_index(character) for character in pair[1]] + [END_SEQUENCE_INDEX]

            input_tensor = torch.tensor(input, dtype=torch.long, device=device).view(-1, 1)
            target_tensor = torch.tensor(target, dtype=torch.long, device=device).view(-1, 1)

            input_output_pair.append((input_tensor, target_tensor))

        return input_output_pair

    def get_embedding(self, sentence, database, device):
        index_list = [database.get_index(character) for character in sentence] + [END_SEQUENCE_INDEX]
        return torch.tensor(index_list, dtype=torch.long, device=device).view(-1, 1)

