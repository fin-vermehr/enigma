import logging
import pickle
from datetime import datetime

import numpy as np
import torch
from dynaconf import settings

from paths import data_directory_path
from language_loader import LanguageLoader
from model import Model
from model_parameters import ModelParameters

logger = logging.getLogger(__name__)

logging.basicConfig(filename='training_history.log', level=logging.INFO)


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

        self.model_parameters = ModelParameters(embedding_size=self.loader.cipher_database.number_of_items)

        self.model = Model(self.loader.cipher_database,
                           self.loader.plain_database,
                           self.model_parameters,
                           self.device)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        logger.info(f'Initialized TrainingEngine with {params} trainable parameters')

    def train_model(self, num_iterations):
        """
        Train the model for the given number of iterations.
        """
        losses = []
        logger.info('Starting Training')
        cipher_batches, plain_batches = self.loader.get_batches(number_of_batches=num_iterations,
                                                                batch_size=settings.BATCH_SIZE)

        for iteration in range(num_iterations):

            input_tensor = cipher_batches[iteration].to(self.device)
            target_tensor = plain_batches[iteration].to(self.device)

            loss = self.model.train(input_tensor, target_tensor)

            losses.append(loss)

            if iteration % 500 == 0:
                logger.info(f"{datetime.now().time()} "
                            f"Iteration: {iteration} out of {num_iterations}, "
                            f"Loss: {np.round(np.mean(losses), 4)}")
                losses = []
            print(loss)
        logger.info('Training Complete. Saving components.')

        self.serialize_components()

    def serialize_components(self):
        """
        Serialize the model and the data loader
        """
        torch.save(self.model, data_directory_path / 'serialized_model.pth.tar')

        with open(data_directory_path / 'serialized_loader.p', 'wb') as loader:
            pickle.dump(self.loader, loader)

        logger.info('Saving components complete.')


if __name__ == '__main__':
    engine = TrainingEngine()
    engine.train_model(90000)
