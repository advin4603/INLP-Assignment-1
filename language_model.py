import click
from ngram import WittenBellSmoothedNGram, KneserNeySmoothedNGram
from tokenizer import tokenize_english
from alive_progress import alive_bar


@click.command()
@click.argument("smoothing", nargs=1)
@click.argument("corpus_path", type=click.Path(exists=True))
@click.option("--input_sentence", prompt=True)
def main(smoothing, corpus_path, input_sentence):
    """Asks for a sentence and prints the probability of that sentence using the passed smoothing mechanism."""
    if smoothing == "k":
        ngram = KneserNeySmoothedNGram(4)
    elif smoothing == "w":
        ngram = WittenBellSmoothedNGram(4)
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
    print(ngram.probability_sentence(
        tuple(tokenize_english(input_sentence)[0])
    ))


if __name__ == "__main__":
    main()
