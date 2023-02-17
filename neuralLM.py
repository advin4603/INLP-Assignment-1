from torch import nn
from torch.nn.functional import log_softmax
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import *

START_TOKEN = "*start*"
END_TOKEN = "*end*"
UNKNOWN_TOKEN = "*unknown*"


class LSTMLanguageModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size):
        super(LSTMLanguageModel, self).__init__()
        self.vocabulary_size = vocab_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.batch_size = batch_size
        self.hidden_to_word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeddings = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeddings)
        word_space = self.hidden_to_word(lstm_out)
        word_probabilities = log_softmax(word_space, dim=1)
        return word_probabilities


class SentencesDataset(Dataset):
    def __init__(self, tokenized_sentences: Sequence[Sequence[str]], vocabulary: List[str]):
        self.tokenized_sentences = tokenized_sentences
        self.vocabulary_index_mapping = {word: i for i, word in enumerate(vocabulary)}

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        tokenized_sentence = [START_TOKEN] + list(self.tokenized_sentences[idx]) + [END_TOKEN]
        encoded_sentence = [self.vocabulary_index_mapping.get(word, self.vocabulary_index_mapping[UNKNOWN_TOKEN]) for
                            word in tokenized_sentence]

        return torch.LongTensor(encoded_sentence[:-1]), torch.LongTensor(encoded_sentence[1:])


def create_collate(vocabulary_mapping: Dict[str, int]):
    def custom_collate(data):
        x_list = []
        y_list = []
        for x, y in data:
            x_list.append(x)
            y_list.append(y)

        return pad_sequence(x_list, batch_first=True, padding_value=vocabulary_mapping[END_TOKEN]), pad_sequence(y_list,
                                                                                                                 batch_first=True,
                                                                                                                 padding_value=
                                                                                                                 vocabulary_mapping[
                                                                                                                     END_TOKEN])

    return custom_collate
