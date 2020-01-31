import logging
import random
from datetime import time

import numpy as np
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
        self.teacher_forcing_ratio = 0.5

        self.cipher_database = cipher_database
        self.plain_database = plain_database

        self.encoder = Encoder(self._model_parameters.embedding_size,
                               self._model_parameters.hidden_size,
                               self._model_parameters.number_of_encoder_layers,
                               self._model_parameters.drop_out).to(self.device)

        self.decoder = Decoder(
            'dot',
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



    def train_two(self, input_tensor, target_tensor):
        input_tensor.to(self.device)
        target_tensor.to(self.device)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        lengths = self.get_lengths(input_tensor)
        mask = self.get_mask(input_tensor)

        loss = 0
        print_losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = self.encoder(input_tensor, lengths)

        decoder_input = torch.LongTensor([[self.cipher_database.start_token_index for _ in range(settings.BATCH_SIZE)]])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(settings.MAX_SEQUENCE_LENGTH):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_tensor[t].view(1, -1)

                if mask[t].sum() == 0:
                    continue

                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_tensor[t].unsqueeze(1), mask[t])
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
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_tensor[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        #TODO: configurable
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), 50)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return sum(print_losses) / n_totals

    # def train(self, input_tensor, target_tensor):
    #     #TODO: Change
    #     teacher_forcing_ratio = 0.5
    #     # encoder_hidden = self.encoder.initialize_hidden_state().to(self.device)
    #
    #     input_tensor.to(self.device)
    #     target_tensor.to(self.device)
    #
    #     # Reset the gradients after each training observation
    #     self.encoder_optimizer.zero_grad()
    #     self.decoder_optimizer.zero_grad()
    #
    #     input_sequence_length = input_tensor.shape[0]
    #     target_sequence_length = target_tensor.shape[0]
    #
    #     #TODO: make configurable
    #     encoder_outputs = torch.zeros(settings.MAX_SEQUENCE_LENGTH, self.encoder.hidden_size * 2, device=self.device)
    #
    #     loss = 0
    #
    #     for index in range(input_sequence_length):
    #         encoder_output, encoder_hidden = self.encoder(
    #             input_tensor[index], torch.ones(len(input_tensor[index])) * 42
    #         )
    #         encoder_outputs[index] = encoder_output[0, 0]
    #
    #     decoder_input = torch.tensor([[self.cipher_database.start_token_index]], device=self.device)
    #     decoder_hidden = encoder_hidden
    #     logger.info(f'Encoder Hidden Size: {encoder_hidden.shape}')
    #     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #
    #     if use_teacher_forcing:
    #         # Teacher forcing: Feed the target as the next input
    #         for di in range(target_sequence_length):
    #             decoder_output, decoder_hidden, decoder_attention = self.decoder(
    #                 decoder_input, decoder_hidden, encoder_outputs)
    #
    #             loss += self.loss(decoder_output, target_tensor[di])
    #             decoder_input = target_tensor[di]  # Teacher forcing
    #
    #     else:
    #         # Without teacher forcing: use its own predictions as the next input
    #         for di in range(target_sequence_length):
    #             decoder_output, decoder_hidden, decoder_attention = self.decoder(
    #                 decoder_input, decoder_hidden, encoder_outputs)
    #             topv, topi = decoder_output.topk(1)
    #             decoder_input = topi.squeeze().detach()  # detach from history as input
    #
    #             loss += self.loss(decoder_output, target_tensor[di])
    #             if decoder_input.item() == self.plain_database.end_token_index:
    #                 break
    #
    #     loss.backward()
    #
    #     # Adjust the encoder and decoder weights
    #     self.encoder_optimizer.step()
    #     self.decoder_optimizer.step()
    #
    #     # Reset the gradients after each optimiser step
    #     self.encoder_optimizer.zero_grad()
    #     self.decoder_optimizer.zero_grad()
    #
    #     padding_ammount = int(sum(input_tensor == 2))
    #     normalized_loss = loss.item() / (settings.MAX_SEQUENCE_LENGTH - padding_ammount)
    #
    #     logger.debug(f'Normalized Loss {normalized_loss}')
    #
    #     return normalized_loss

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
