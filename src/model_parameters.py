class ModelParameters:

    def __init__(self,
                 embedding_size: int = 35,
                 hidden_size: int = 256,
                 output_length: int = 42,
                 weight_decay: float = 0.1,
                 learning_rate: float = 0.0005,
                 batch_size: int = 16,
                 ):

        self.weight_decay = weight_decay
        self.output_length = output_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
