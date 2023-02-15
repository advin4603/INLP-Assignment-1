import click
from ngram import WittenBellSmoothedNGram, KneserNeySmoothedNGram
from tokenizer import tokenize_english
from alive_progress import alive_bar
from random import choices

@click.command()
@click.argument("smoothing", nargs=1)
@click.argument("corpus_path", type=click.Path(exists=True))
def main(smoothing, corpus_path):
    """Generates a sentence using the passed smoothing mechanism and corpus."""
    n = 4
    if smoothing == "k":
        ngram = KneserNeySmoothedNGram(n)
    elif smoothing == "w":
        ngram = WittenBellSmoothedNGram(n)
    else:
        click.echo("Invalid smoothing.")
        exit(1)

    with open(corpus_path) as f:
        text = f.read()

    tokenized_text = tokenize_english(text)

    with alive_bar(len(tokenized_text), title="Training") as bar:
        for sentence in tokenized_text:
            ngram.train_sentence(tuple(sentence))
            bar()

    ngram.compute_cache()
    history = tuple()
    sentence_end = {".", "!", "?"}
    token = None
    vocabulary = list(ngram.vocabulary)

    while token not in sentence_end:
        weights = (ngram.probability(w, history) for w in vocabulary)
        token = choices(vocabulary, weights=tuple(weights), k=1)[0]
        history += (token,)
        print(token, end=" ", flush=True)
    print()


if __name__ == "__main__":
    main()
