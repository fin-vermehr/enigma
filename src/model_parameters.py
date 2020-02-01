class ModelParameters:
    """
    Simple class to contain all the model parameters
    """

    def __init__(self,
                 embedding_size: int = 28,
                 hidden_size: int = 53,
                 max_sequence_length: int = 42,
                 weight_decay: float = 0.1,
                 learning_rate: float = 0.0005,
                 batch_size: int = 16,
                 number_of_decoder_layers: int = 1,
                 number_of_encoder_layers: int = 1,
                 drop_out: float = 0,
                 gradient_clipping: float = 50.0,
                 ):

        self.gradient_clipping = gradient_clipping
        self.weight_decay = weight_decay
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.number_of_decoder_layers = number_of_decoder_layers
        self.drop_out = drop_out
        self.number_of_encoder_layers = number_of_encoder_layers
