import logging

import torch

from nlp_takehome.src.language_database import LanguageDatabase

logger = logging.getLogger(__name__)

PAIR_CIPHER_INDEX = 0
PAIR_PLAIN_INDEX = 1


class LanguageLoader:

    def __init__(self, plain_text, cipher_text):
        logger.info('Initializing LanguageLoader')

        self.plain_database = LanguageDatabase(plain_text)
        self.cipher_database = LanguageDatabase(cipher_text)
        logger.info('Populated Databases')

        self.sequence_pairs = [(cipher_text[index], plain_text[index]) for index in range(len(plain_text))]
        logger.info(f'Paired {len(cipher_text)} cipher sequences with their corresponding plain sequences')

    def get_indices(self, sequence: str, database: LanguageDatabase):
        return [database.get_index(character) for character in sequence]

    def get_embedding_tensor(self, sequence: str, database: LanguageDatabase, device: str):
        indices = self.get_indices(sequence, database)
        indices.append(database.end_token_index)

        # TODO: what does view do?
        return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

    def get_tensor_from_pair(self, pair, device: str):
        input_tensor = self.get_embedding_tensor(pair[PAIR_CIPHER_INDEX], self.cipher_database, device)
        target_tensor = self.get_embedding_tensor(pair[PAIR_PLAIN_INDEX], self.plain_database, device)
        return input_tensor, target_tensor
