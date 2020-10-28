# GRU and Attention Based Engima Decipher Algorithm 

## Layout

The `language_loader.py` and `language_database.py` organize and manage the cipher and english plain representation.
It allows the model to easily load and process all the data.

The `model.py` file manages the training and organizes the sub-components (`decoder`, `encoder`, `loss`) into a single model.

The `training_engine.py` manages the interaction between the model, and the training data. It is responsible for the training process.

The `evaluation_engine.py` manages the interaction between the test data. It is responsible for the inference. This has to be seperate from the `TrainingEngine` as the decoding mechanism is greedy rather than teacher forced. This returns the most likely decoding of the cipher, rather than one which is more conducive to training.

The trained model, associated language loader and data is all stored in the `data` directory.

All the hyper-parameters are stored in `settings.yaml`. 

## Design Choices

### Why use a GRU rather than an LSTM
The GRU is a stripped down LSTM unit, it doesn't have the memory unit. This means we have fewer trainable parameters,
and thus require less training data, and is faster to train. In addition, there's some research
indicating that GRUs perform better on relatively short sequence lengths.

### Why use teacher forced learning
Teacher forced learning enables RNN to train quicker. Instead of decoding an entire sequence, by using the previous
prediction to predict the current output, it uses the ground-truth.

### Why use attention

In a vanilla Seq2Seq model, a single context vector is created by the encoder that contains all the information of the input sequence.
The decoder has to decode it. The problem with this is that a fixed length context vector can only encode so much information,
and tends to be more biased towards recent tokens the encoder encoded.

Attention alleviates this issue by creating a new context vector for each token in the source sequence. Then for the
decoding mechanism, attention calculates how relevent each of the context vectors are for the token that's just being decoded.

### Evaluation Metric
I used Levenshtein distance to measure the difference between the plain english, and the ml decoded engima cipher of this plain
english sequence. It represents the 'edit distance', that is, how many changes have to be made to the output sequence to
transform it back to the original english.


## Steps to Run

1. Generate the training and testing data using:
`python src/generate_data.py`

2. Train the Seq2Seq model and serialize it:
`python src/training_engine.py`

3. Evaluate the performance of the model using Levenshtein Distance:
`python src/cipher_main.py`
