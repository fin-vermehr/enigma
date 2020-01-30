import logging
import random
from datetime import time

import torch
import torch.nn as nn
from dynaconf import settings

from nlp_takehome.src.decoder import Decoder
from nlp_takehome.src.encoder import Encoder
from nlp_takehome.src.language_database import LanguageDatabase
from nlp_takehome.src.model_parameters import ModelParameters

logger = logging.getLogger(__name__)

MAX_LOSS_HISTORY = 250


class EnigmaDecryptionModel:
    def __init__(
            self,
            cipher_database: LanguageDatabase,
            plain_database: LanguageDatabase,
            parameters: ModelParameters,
            device: str,):
        logger.info(f"Creating decryption model on {device}")

        self.device = device

        self._model_parameters = parameters

        self.cipher_database = cipher_database
        self.plain_database = plain_database

        self.encoder = Encoder(self.cipher_database.number_of_items,
                               self._model_parameters.hidden_size).to(self.device)

        self.decoder = Decoder(self._model_parameters.hidden_size,
                               self.plain_database.number_of_items,
                               dropout_p=0.1).to(self.device)

        self.iteration = 0

        self.encoder_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()),
            lr=self._model_parameters.learning_rate,
        )

        self.decoder_optimizer = torch.optim.Adam(
            list(self.decoder.parameters()),
            lr=self._model_parameters.learning_rate,
        )

        self.loss = nn.NLLLoss()

        self._max_loss_history = MAX_LOSS_HISTORY

        self.loss_history = []

        logger.info('Decryption model created')

    def train(self, input_tensor, target_tensor):
        #TODO: Change
        teacher_forcing_ratio = 0.5
        encoder_hidden = self.encoder.initialize_hidden_state().to(self.device)

        input_tensor.to(self.device)
        target_tensor.to(self.device)

        # Reset the gradients after each training observation
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_sequence_length = input_tensor.shape[0]
        target_sequence_length = target_tensor.shape[0]

        encoder_outputs = torch.zeros(settings.MAX_SEQUENCE_LENGTH, self.encoder.hidden_size, device=self.device)

        loss = 0

        for index in range(input_sequence_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[index], encoder_hidden
            )
            encoder_outputs[index] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.cipher_database.start_token_index]], device=self.device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_sequence_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                loss += self.loss(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_sequence_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.loss(decoder_output, target_tensor[di])
                if decoder_input.item() == self.plain_database.end_token_index:
                    break

        loss.backward()

        # Adjust the encoder and decoder weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # Reset the gradients after each optimiser step
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        normalized_loss = loss.item() / target_sequence_length

        # Update logs
        self.loss_history.append(normalized_loss)
        # If loss_history is too large then reduce size
        self.loss_history = self.loss_history[-self._max_loss_history:]

        logger.debug(f'Normalized Loss {normalized_loss}')

        return normalized_loss
