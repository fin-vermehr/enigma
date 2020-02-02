import logging
import pickle
from datetime import datetime

import numpy as np
import torch
from dynaconf import settings

from nlp_takehome.src.language_loader import LanguageLoader, data_directory_path
from nlp_takehome.src.model import Model
from nlp_takehome.src.model_parameters import ModelParameters

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


class TrainingEngine:
    """
    Integrates the different modules into one and trains the model.
    """

    def __init__(self):

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        logger.info(f'Initializing the TrainingEngine on {self.device}')

        self.loader = LanguageLoader()

        self.model_parameters = ModelParameters(
            embedding_size=self.loader.cipher_database.number_of_items,
            batch_size=settings.BATCH_SIZE,
            max_sequence_length=settings.MAX_SEQUENCE_LENGTH
        )

        self.model = Model(self.loader.cipher_database,
                           self.loader.plain_database,
                           self.model_parameters,
                           self.device)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        parameter_count = sum([np.prod(p.size()) for p in model_parameters])

        logger.info(f'Initialized TrainingEngine with {parameter_count} trainable parameters')

    def train_model(self, number_of_iterations: int):
        """
        Train the model for the given number of iterations.
        """
        losses = []
        logger.info('Starting Training')
        cipher_batches, plain_batches = self.loader.get_batches(number_of_batches=number_of_iterations,
                                                                batch_size=settings.BATCH_SIZE)

        for iteration in range(number_of_iterations):

            input_tensor = cipher_batches[iteration].to(self.device)
            target_tensor = plain_batches[iteration].to(self.device)

            loss = self.model.train(input_tensor, target_tensor)

            losses.append(loss)

            if iteration % 500 == 0:
                logger.info(f"{datetime.now().time()} "
                            f"Iteration: {iteration} out of {number_of_iterations}, "
                            f"Loss: {np.round(np.mean(losses), 4)}")
                losses = []

        logger.info('Training Complete. Saving components.')

        self.serialize_components()

    def serialize_components(self):
        """
        Serialize the model and the data loader
        """
        torch.save(self.model, data_directory_path / 'serialized_decipher_model.pth.tar')

        self.loader.pairs = []

        with open(data_directory_path / 'serialized_loader.p', 'wb') as loader:
            pickle.dump(self.loader, loader)

        logger.info('Saving components complete.')


if __name__ == '__main__':
    engine = TrainingEngine()
    engine.train_model(170000)
