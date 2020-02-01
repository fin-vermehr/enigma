import pickle

import torch
from dynaconf import settings
from torch import Tensor

from nlp_takehome.paths import data_directory_path


class EvaluationEngine:
    def __init__(self):
        """
        The evaluation module of the deciphering mechanism. Loads the data_loader and the pre-trained model.
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = torch.load(data_directory_path / 'serialized_model.pth.tar')
        self.loader = pickle.load(open(data_directory_path / 'serialized_loader.p', 'rb'))

    # TODO: Clean up
    def evaluate(self, sentence: str) -> str:
        """
        Decipher this sentence to plain text
        @param sentence: the cipher sentence to be deciphered
        @return: plain text
        """
        embedded_batch = self.loader.get_embedding(sentence, self.loader.cipher_database).to(self.device)
        tokens = self.greedy_decode_embedded_cipher(embedded_batch)
        decoded_words = [self.loader.plain_database.get_item(token.item()) for token in tokens]
        return ''.join(decoded_words)

    def greedy_decode_embedded_cipher(self, input_sequence: Tensor) -> Tensor:
        """
        A greedy decoder. Given a cipher that is already in its indexed form, decode it to plain text.
        @param input_sequence: the cipher to be deciphered
        @return: The indexed version of the plain text
        """

        # TODO:

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
        return all_tokens
