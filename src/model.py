import logging
import random

import torch
import torch.nn as nn
from dynaconf import settings

from nlp_takehome.src.decoder import Decoder
from nlp_takehome.src.encoder import Encoder
from nlp_takehome.src.language_database import LanguageDatabase
from nlp_takehome.src.model_parameters import ModelParameters

logger = logging.getLogger(__name__)

# TODO: don't need this anymore
MAX_LOSS_HISTORY = 250


class EnigmaDecryptionModel:
    def __init__(
            self,
            cipher_database: LanguageDatabase,
            plain_database: LanguageDatabase,
            parameters: ModelParameters,
            device: str, ):
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

        self.loss = nn.NLLLoss(ignore_index=settings.PADDING_INDEX)

        logger.info('Decryption model created')

    #TODO: Clean Up
    def train(self, cipher_tensor, plain_tensor):
        cipher_tensor.to(self.device)
        plain_tensor.to(self.device)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        lengths = self.get_lengths(cipher_tensor)
        mask = self.get_mask(cipher_tensor)

        loss = 0
        print_losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = self.encoder(cipher_tensor, lengths)

        decoder_input = torch.tensor([self.cipher_database.start_token_index] * self._model_parameters.batch_size,
                                     device=self.device).unsqueeze(0)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.number_of_layers]

        # Forward batch of sequences through decoder one time step at a time
        if random.random() < self.teacher_forced_probability:
            for t in range(settings.MAX_SEQUENCE_LENGTH):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = plain_tensor[t].view(1, -1)

                if mask[t].sum() == 0:
                    continue

                mask_loss, nTotal = self.maskNLLLoss(decoder_output, plain_tensor[t].unsqueeze(1), mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(settings.MAX_SEQUENCE_LENGTH):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(settings.BATCH_SIZE)]])
                decoder_input = decoder_input.to(self.device)

                if mask[t].sum() == 0:
                    continue

                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, plain_tensor[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        # Perform backpropatation
        loss.backward()

        # TODO: change this somehow
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(),
                                     self._model_parameters.gradient_clipping)

        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(),
                                     self._model_parameters.gradient_clipping)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return sum(print_losses) / n_totals

    # TODO: change this yo
    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean().to(self.device)

        return loss, nTotal.item()

    def get_lengths(self, tensor: torch.Tensor) -> torch.Tensor:
        lengths = [sum(tensor[:, i] != self.cipher_database.pad_token_index).item() for i in range(tensor.shape[1])]

        return torch.tensor(lengths).to(self.device)

    def get_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = (tensor != self.cipher_database.pad_token_index).to(self.device)
        return mask
