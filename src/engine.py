import logging

import torch
from dynaconf import settings
import numpy as np

logger = logging.getLogger(__name__)

from nlp_takehome.src.enigma_decryption_model import EnigmaDecryptionModel
from nlp_takehome.src.model_parameters import ModelParameters
from nlp_takehome.src.language_loader import LanguageLoader, PAIR_CIPHER_INDEX, PAIR_PLAIN_INDEX
SOS_index = 0
EOS_index = 1

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

        self.cipher_batches, self.plain_batches = self.loader.get_batches(15000, 16)

    def early_stopping(self):
        # TODO: make early stopping or rename
        losses = []
        for iteration in range(self.num_iterations):

            input_tensor = self.cipher_batches[iteration].to(self.device)
            target_tensor = self.plain_batches[iteration].to(self.device)

            loss = self.model.train_two(input_tensor, target_tensor)
            losses.append(loss)
            if iteration % 100 == 0:
                logger.info(f"Iteration: {iteration} out of {self.num_iterations}, Loss: {np.mean(losses)}")
                losses = []

    # def indexesFromSentence(self, lang, sentence):
    #     return [lang.get_index(char) for char in sentence]
    #
    # def tensorFromSentence(self, lang, sentence):
    #     indexes = self.indexesFromSentence(lang, sentence)
    #     indexes.append(EOS_index)
    #     return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = self.loader.get_embedding(sentence, self.loader.cipher_database, self.device)

            input_length = input_tensor.size()[0]

            encoder_outputs = torch.zeros(settings.MAX_SEQUENCE_LENGTH, self.model.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.model.encoder(input_tensor[ei])
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.loader.cipher_database.start_token_index]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(42, 42)

            #TODO
            for di in range(42 - 1):
                decoder_output, decoder_hidden, decoder_attention = self.model.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data

                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.loader.plain_database.end_token_index:
                    break
                else:
                    decoded_words.append(self.loader.plain_database.get_item(topi.item()))

                decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# if __name__ == "__main__":
#     plain, cipher = generate_data(1 << 5)
#     engine = Engine(5000)
#     engine.early_stopping()
#     for i in range(len(plain)):
#         print('>', cipher[i])
#         print('=', plain[i])
#         output_words, attentions = engine.evaluate(cipher[i])
#         output_sentence = ''.join(output_words)
#         print(f'< {output_sentence} \n')
