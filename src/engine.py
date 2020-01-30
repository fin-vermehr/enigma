import torch
from dynaconf import settings

from nlp_takehome.src.cipher_take_home import generate_data
from nlp_takehome.src.enigma_decryption_model import EnigmaDecryptionModel
from nlp_takehome.src.model_parameters import ModelParameters
from nlp_takehome.src.language_loader import LanguageLoader, PAIR_CIPHER_INDEX, PAIR_PLAIN_INDEX
SOS_index = 0
EOS_index = 1


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

        self.training_pairs = self.loader.get_embed_pairs(num_iterations, device)

    def early_stopping(self):
        # TODO: make early stopping or rename
        for iteration in range(self.num_iterations):

            training_pair = self.training_pairs[iteration]

            input_tensor = training_pair[PAIR_CIPHER_INDEX]
            target_tensor = training_pair[PAIR_PLAIN_INDEX]

            loss = self.model.train(input_tensor, target_tensor)

            if iteration % 500 == 0:

                print(f"Iteration: {iteration} out of {self.num_iterations}, Loss: {loss}")

    def indexesFromSentence(self, lang, sentence):
        return [lang.get_index(char) for char in sentence]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_index)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = self.loader.get_embedding(sentence, self.loader.cipher_database, self.device)

            input_length = input_tensor.size()[0]
            encoder_hidden = self.model.encoder.initialize_hidden_state().to(self.device)
            encoder_outputs = torch.zeros(settings.MAX_SEQUENCE_LENGTH, self.model.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.model.encoder(input_tensor[ei], encoder_hidden)
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


if __name__ == "__main__":
    plain, cipher = generate_data(1 << 5)
    engine = Engine(5000)
    engine.early_stopping()
    for i in range(len(plain)):
        print('>', cipher[i])
        print('=', plain[i])
        output_words, attentions = engine.evaluate(cipher[i])
        output_sentence = ''.join(output_words)
        print(f'< {output_sentence} \n')
