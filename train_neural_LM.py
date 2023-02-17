import torch
from torch.utils.data import DataLoader
from neuralLM import LSTMLanguageModel, SentencesDataset, create_collate
from tokenizer import tokenize_english
from neuralLM import END_TOKEN, UNKNOWN_TOKEN, START_TOKEN
from alive_progress import alive_bar
import math
import random

learning_rate = 2e-3
batch_size = 64
loss_function = torch.nn.CrossEntropyLoss()
optimizer_function = torch.optim.Adam
epochs = 10
hidden_dim = 128
embedding_dim = 100


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    with open("data/Pride and Prejudice - Jane Austen.txt") as f:
        tokenized_sentences = tokenize_english(f.read())

    random.shuffle(tokenized_sentences)

    vocabulary = {END_TOKEN, UNKNOWN_TOKEN, START_TOKEN}

    train_validate_test_split = 75, 15, 10
    split_index_train_validate = int(train_validate_test_split[0] / 100 * len(tokenized_sentences))
    split_index_validate_test = -int(train_validate_test_split[2] / 100 * len(tokenized_sentences))
    training_sentences, validation_sentences, test_sentences = tokenized_sentences[:split_index_train_validate], \
        tokenized_sentences[split_index_train_validate:split_index_validate_test], tokenized_sentences[
                                                                                   split_index_validate_test:]
    for tokenized_sentence in training_sentences:
        for word in tokenized_sentence:
            vocabulary.add(word)
    vocabulary = list(vocabulary)
    model = LSTMLanguageModel(embedding_dim, hidden_dim, len(vocabulary), batch_size)
    vocabulary_mapping = {word: i for i, word in enumerate(vocabulary)}
    collate_function = create_collate(vocabulary_mapping)
    training_data = SentencesDataset(training_sentences, vocabulary)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_function)
    validation_data = SentencesDataset(validation_sentences, vocabulary)
    validate_dataloader = DataLoader(validation_data, batch_size=1, collate_fn=collate_function)
    testing_data = SentencesDataset(test_sentences, vocabulary)
    test_dataloader = DataLoader(testing_data, batch_size=1, collate_fn=collate_function)
    optimizer = optimizer_function(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        train_loop(train_dataloader, model, loss_function, optimizer, t + 1)
        validation_loop(validate_dataloader, model, loss_function, t + 1)
    test_loop(test_dataloader, model, loss_function)
    print("Done!")


def train_loop(dataloader, model, loss_fn, optimizer, epoch_number):
    size = len(dataloader.dataset)
    with alive_bar(len(dataloader), title=f"Training (Epoch: {epoch_number})") as bar:
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X).permute(0, 2, 1)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss, current = loss.item(), (batch + 1) * len(X)
            bar.text(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            bar()


def validation_loop(dataloader, model, loss_fn, epoch_number):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, perplexity = 0, 0

    with torch.no_grad(), alive_bar(len(dataloader), title=f"Validating (Epoch: {epoch_number})") as bar:
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred.permute(0, 2, 1), y).item()
            test_loss += loss_fn(pred.permute(0, 2, 1), y).item()
            perplexity += math.exp(loss)
            bar()

    test_loss /= num_batches
    perplexity /= size
    print(f"Validation Error: \n Avg Perplexity: {perplexity:>0.1f}, Avg loss: {test_loss:>8f} \n")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, perplexity = 0, 0

    with torch.no_grad(), alive_bar(len(dataloader), title=f"Testing") as bar:
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred.permute(0, 2, 1), y).item()
            test_loss += loss_fn(pred.permute(0, 2, 1), y).item()
            perplexity += math.exp(loss)
            bar()

    test_loss /= num_batches
    perplexity /= size
    print(f"Test Error: \n Avg Perplexity: {perplexity:>0.1f}, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    main()
