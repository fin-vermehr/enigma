import logging
import random

import torch
import torch.nn as nn
from dynaconf import settings

from decoder import Decoder
from encoder import Encoder
from language_database import LanguageDatabase
from masked_nllloss import MaskedNLLLoss
from model_parameters import ModelParameters

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(
            self,
            cipher_database: LanguageDatabase,
            plain_database: LanguageDatabase,
            parameters: ModelParameters,
            device: str, ):
        super(Model, self).__init__()

        """
        Integrates the different model components into one.
        """
        logger.info(f"Creating decryption model on {device}")

        self.device = device

        self._model_parameters = parameters
        self.teacher_forced_probability = 0.5

        self.cipher_database = cipher_database
        self.plain_database = plain_database

        self.encoder = Encoder(self._model_parameters.embedding_size,
                               self._model_parameters.hidden_size,
                               self._model_parameters.number_of_encoder_layers,
                               self._model_parameters.drop_out).to(self.device)

        self.decoder = Decoder(
            self._model_parameters.embedding_size,
            self._model_parameters.hidden_size,
            self.plain_database.number_of_items,
            self._model_parameters.number_of_decoder_layers,
            self._model_parameters.drop_out,
        ).to(self.device)

        self.iteration = 0

        self.encoder_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()),
            lr=self._model_parameters.learning_rate,
        )

        self.decoder_optimizer = torch.optim.Adam(
            list(self.decoder.parameters()),
            lr=self._model_parameters.learning_rate,
        )

        self.loss = MaskedNLLLoss().to(self.device)

        logger.info('Decryption model created')

    def train(self, cipher_tensor: torch.Tensor, plain_tensor: torch.Tensor):
        """
        Train the model for a single iteration with a pre-specified probability of using teacher forced learning.

        @param cipher_tensor: the indexed input tensor to be deciphered
        @param plain_tensor: the indexed target tensor
        """
        cipher_tensor.to(self.device)
        plain_tensor.to(self.device)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        lengths = self.get_lengths(cipher_tensor)
        mask = self.get_mask(cipher_tensor)

        total_loss = 0
        losses = []
        count = 0  # the total number of un-masked items.

        encoder_outputs, encoder_hidden = self.encoder(cipher_tensor, lengths)

        decoder_input = torch.tensor([self.cipher_database.start_token_index] * self._model_parameters.batch_size,
                                     device=self.device).unsqueeze(0)

        # set the final encoder hidden state, to the initial decoder hidden state.
        decoder_hidden = encoder_hidden[:self.decoder.number_of_layers]

        # use teacher forced learning according to the set probability
        if random.random() > self.teacher_forced_probability:
            # Teacher forced
            for step in range(settings.MAX_SEQUENCE_LENGTH):
                # In the case that every item in the batch is just padding at this step, skip.
                if mask[step].sum() == 0:
                    break

                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forced learning
                decoder_input = plain_tensor[step].flatten().unsqueeze(0)

                mask_loss, total_items = self.loss(decoder_output, plain_tensor[step].unsqueeze(1), mask[step])
                total_loss += mask_loss
                losses.append(mask_loss.item() * total_items)
                count += total_items
        else:
            # non-teacher forced
            for step in range(settings.MAX_SEQUENCE_LENGTH):
                # In the case that every item in the batch is just padding at this step, skip.
                if mask[step].sum() == 0:
                    continue

                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                _, index = decoder_output.max(1)
                decoder_input = torch.tensor(
                    [index[i] for i in range(settings.BATCH_SIZE)]
                ).unsqueeze(0).to(self.device)

                # Calculate and accumulate loss
                mask_loss, total_items = self.loss(decoder_output, plain_tensor[step], mask[step])
                total_loss += mask_loss
                losses.append(mask_loss.item() * total_items)
                count += total_items

        # Perform backpropagation
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.encoder.parameters(),
                                 self._model_parameters.gradient_clipping)

        nn.utils.clip_grad_norm_(self.decoder.parameters(),
                                 self._model_parameters.gradient_clipping)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(losses) / count

    def get_lengths(self, cipher_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the lengths of each sequence in a batch (not including mask).

        @param cipher_tensor: (max_sequence_length x batch_size)
        @return: (batch_size)
        """
        lengths = [
            sum(cipher_tensor[:, i] != self.cipher_database.pad_token_index).item()
            for i in range(cipher_tensor.shape[1])]

        return torch.tensor(lengths).to(self.device)

    def get_mask(self, cipher_tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a cipher_tensor get the padding mask
        @param cipher_tensor: (max_sequence_length x batch_size)
        @return: (max_sequence_length x batch_size)
        """
        return (cipher_tensor != self.cipher_database.pad_token_index).to(self.device)
