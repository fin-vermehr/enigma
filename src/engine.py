import logging
from datetime import datetime

import numpy as np
import torch
from dynaconf import settings

from nlp_takehome.src.enigma_decryption_model import EnigmaDecryptionModel
from nlp_takehome.src.language_loader import LanguageLoader
from nlp_takehome.src.model_parameters import ModelParameters

logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)


class Engine:
    def __init__(self, num_iterations, device='cuda'):

        self.loader = LanguageLoader()
        self.device = device
        self.num_iterations = num_iterations

        self.model_parameters = ModelParameters(embedding_size=self.loader.cipher_database.number_of_items)

        self.model = EnigmaDecryptionModel(self.loader.cipher_database,
                                           self.loader.plain_database,
                                           ModelParameters(),
                                           self.device)

        self.cipher_batches, self.plain_batches = self.loader.get_batches(8000, 16)

    def early_stopping(self):
        # TODO: make early stopping or rename
        losses = []
        logger.info('Starting Training')
        for iteration in range(self.num_iterations):

            input_tensor = self.cipher_batches[iteration].to(self.device)
            target_tensor = self.plain_batches[iteration].to(self.device)

            loss = self.model.train(input_tensor, target_tensor)

            losses.append(loss)

            if iteration % 500 == 0:

                logger.info(f"{datetime.now().time()}"
                            f"Iteration: {iteration} out of {self.num_iterations},"
                            f"Loss: {np.round(np.mean(losses), 4)}")
                losses = []

    def GreedySearchDecoder(self, input_sequence):
        input_length = torch.tensor([sum(input_sequence != 0)]).to(self.device)

        encoder_outputs, encoder_hidden = self.model.encoder(input_sequence, input_length)
        decoder_hidden = encoder_hidden[:self.model.decoder.number_of_layers]

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * settings.START_SEQUENCE_INDEX
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(settings.MAX_SEQUENCE_LENGTH):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            if decoder_input.item() in [settings.END_SEQUENCE_INDEX, settings.PADDING_INDEX]:
            # if decoder_input.item() == settings.END_SEQUENCE_INDEX:
                break
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

    def evaluate(self, sentence, max_length=settings.MAX_SEQUENCE_LENGTH):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = self.loader.get_embedding(sentence, self.loader.cipher_database)

        input_batch = indexes_batch.to(self.device)
        # Decode sentence with searcher
        # print(input_batch)
        tokens, scores = self.GreedySearchDecoder(input_batch,)
        # indexes -> words
        decoded_words = [self.loader.plain_database.get_item(token.item()) for token in tokens]
        return decoded_words
