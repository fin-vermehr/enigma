import logging
import pickle

import torch
from dynaconf import settings
from torch import Tensor

from nlp_takehome.paths import data_directory_path

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


class EvaluationEngine:
    def __init__(self):
        """
        The evaluation module of the deciphering mechanism. Loads the data_loader and the pre-trained model.
        """

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        logger.info('Loading pre-trained model...')
        self.model = torch.load(data_directory_path / 'serialized_decipher_model.pth.tar')
        logger.info('Model loaded')
        self.loader = pickle.load(open(data_directory_path / 'serialized_loader.p', 'rb'))
        logger.info('Data Loader loaded')

    def evaluate(self, sentence: str) -> str:
        """
        Decipher this sentence to plain text
        @param sentence: the cipher sentence to be deciphered
        @return: plain text
        """
        embedded_batch = self.loader.get_embedding(sentence, self.loader.cipher_database).to(self.device)
        tokens = self.decode_embedded_cipher(embedded_batch)

        # turn the index tensor representation of the plain word to its string version
        decoded_words = [self.loader.plain_database.get_item(token.item()) for token in tokens]
        return ''.join(decoded_words)

    def decode_embedded_cipher(self, input_sequence: Tensor) -> Tensor:
        """
        A greedy decoder. Given a cipher that is already in its indexed form, decipher it to plain text.
        @param input_sequence: the cipher to be deciphered
        @return: The indexed version of the plain text
        """
        input_length = torch.tensor([sum(input_sequence != 0)]).to(self.device)

        encoder_outputs, encoder_hidden = self.model.encoder(input_sequence, input_length)
        decoder_hidden = encoder_hidden[:self.model.decoder.number_of_layers]

        # Initialize decoder input with start of sequence index and
        decoder_input = torch.tensor([settings.START_SEQUENCE_INDEX], device=self.device, dtype=torch.long).unsqueeze(0)
        tokens = torch.zeros([0], dtype=torch.long).to(self.device)
        scores = torch.zeros([0]).to(self.device)

        # Iteratively decode one word token at a time
        for _ in range(settings.MAX_SEQUENCE_LENGTH):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Argmax the softmax output to get the most likely token and highest score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            # We don't want the End Sequence Token or the padding tokens in the final output
            if decoder_input.item() == settings.END_SEQUENCE_INDEX:
                break

            # Record token and score
            scores = torch.cat((scores, decoder_scores), dim=0)
            tokens = torch.cat((tokens, decoder_input), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = decoder_input.unsqueeze(0)

        # Return all the tokens as a tensor
        return tokens
