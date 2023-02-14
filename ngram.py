from typing import *
import math
from alive_progress import alive_bar


class NGram:
    def __init__(self, n: int):
        self.n = n
        self.counts: Dict[Tuple[Union[str, None], ...], Dict[str, int]] = {}
        self.unique_ngram_count: int = 0
        self.cache_invalidated: bool = False
        self.word_histories = {}

    def train_sentence(self, sentence_tokens: Tuple[str, ...]):
        tokens = (None,) * (self.n - 1) + sentence_tokens
        self.cache_invalidated = True
        for i in range(self.n - 1, len(tokens)):
            history = tokens[i - self.n + 1:i]
            token = tokens[i]
            self.word_histories.setdefault(token, set()).add(history)
            self.counts.setdefault(history, {}).setdefault(token, 0)
            self.counts[history][token] += 1

    def compute_cache(self):
        if not self.cache_invalidated:
            return
        self.unique_ngram_count = sum(len(words) for words in self.counts.values())
        self.cache_invalidated = False


class BackoffSmoothedNgram:
    def __init__(self, n: int):
        self.n = n
        self.ngrams: List[NGram] = [NGram(i) for i in range(1, n + 1)]
        self.vocabulary: Set[str] = set()
        self.continuation_counts: List[Dict[str, int]] = [{} for i in range(1, n + 1)]

    def cache_invalidated(self):
        return any(ngram.cache_invalidated for ngram in self.ngrams)

    def train_sentence(self, sentence_tokens: Tuple[str, ...]):
        for ngram in self.ngrams:
            ngram.train_sentence(sentence_tokens)
        self.vocabulary = set(self.ngrams[0].counts[tuple()].keys())

    def compute_cache(self):
        with alive_bar(len(self.ngrams), title="Computing Cache") as bar:
            for ngram, continuation_count in zip(self.ngrams, self.continuation_counts):
                ngram.compute_cache()
                for word in self.vocabulary:
                    continuation_count[word] = len(ngram.word_histories[word])
                bar()

    def entropy(self, sentences_tokens: Sequence[Sequence[str]], compute_cache: bool = False) -> float:
        if compute_cache:
            self.compute_cache()
        entropy = 0
        with alive_bar(len(sentences_tokens), title="Computing Entropy") as bar:
            for sentence_tokens in sentences_tokens:
                probability = self.probability_sentence(tuple(sentence_tokens))
                if probability == 0:
                    continue
                entropy += probability * math.log(probability, 2)
                bar()
        return -entropy

    def perplexity(self, sentences_tokens: Sequence[Sequence[str]]) -> float:
        entropy = self.entropy(sentences_tokens)
        return 2 ** entropy

    def probability(self, word: str, history: Tuple[str, ...]) -> float:
        raise NotImplementedError

    def probability_sentence(self, tokens: Tuple[str, ...]) -> float:
        tokens = (None,) * (self.n - 1) + tokens
        probability = 1
        for i in range(self.n - 1, len(tokens)):
            history = tokens[i - self.n + 1:i]
            token = tokens[i]
            probability *= self.probability(token, history)

        return probability


class WittenBellSmoothedNGram(BackoffSmoothedNgram):
    def probability(self, word: str, history: Tuple[str, ...]) -> float:
        history = (None,) * (self.n - 1) + history
        return self._probability(word, history[-(self.n - 1):])

    def _probability(self, word: str, history: Tuple[Union[str, None], ...]) -> float:
        ngram_order = len(history) + 1
        if ngram_order == 1:
            lower_order_probability = 1 / len(self.vocabulary)
        else:
            lower_order_probability = self._probability(word, history[1:])

        ngram = self.ngrams[ngram_order - 1]

        if history not in ngram.counts:
            return lower_order_probability

        if word not in ngram.counts[history]:
            count_word_with_history = 0
        else:
            count_word_with_history = ngram.counts[history][word]
        count_unique_words_with_history = len(ngram.counts[history])
        count_history = sum(ngram.counts[history].values())
        return (count_word_with_history + count_unique_words_with_history * lower_order_probability) / \
            (count_history + count_unique_words_with_history)


class KneserNeySmoothedNGram(BackoffSmoothedNgram):
    def __init__(self, n: int, d: float = .75):
        super().__init__(n)
        self.d = d

    def probability(self, word: str, history: Tuple[str, ...]) -> float:
        history = (None,) * (self.n - 1) + history
        return self._probability(word, history[-(self.n - 1):])

    def entropy(self, sentences_tokens: Sequence[Sequence[str]], compute_cache=True):
        return super().entropy(sentences_tokens, compute_cache)

    def _probability(self, word: str, history: Tuple[Union[str, None], ...]) -> float:
        ngram_order = len(history) + 1
        if ngram_order == 1:
            lower_order_probability = 1 / len(self.vocabulary)
        else:
            lower_order_probability = self._probability(word, history[1:])

        ngram = self.ngrams[ngram_order - 1]

        if history not in ngram.counts:
            return lower_order_probability
        count_history = sum(ngram.counts[history].values())

        if ngram_order == self.n:
            count_kn_word = ngram.counts[history].get(word, 0)
            count_kn_history = count_history
        else:
            count_kn_word = self.continuation_counts[ngram_order - 1].get(word, 0) \
                if not ngram.cache_invalidated \
                else sum(1 for ngram_history in ngram.counts if word in ngram.counts[ngram_history])
            count_kn_history = ngram.unique_ngram_count if not ngram.cache_invalidated else sum(
                len(words) for words in ngram.counts.values())

        count_unique_words_with_history = len(ngram.counts[history])
        ngram_probability = max(count_kn_word - self.d, 0) / count_kn_history
        backoff_lambda = self.d * count_unique_words_with_history / count_history

        return ngram_probability + backoff_lambda * lower_order_probability


if __name__ == "__main__":
    from tokenizer import tokenize_english

    fourgram = WittenBellSmoothedNGram(4)
    with open("data/Ulysses - James Joyce.txt") as f:
        text = f.read()
    # text = "I am eating chocolate. He is consuming ice-cream."
    tokenized_text = tokenize_english(text)
    with alive_bar(len(tokenized_text), title="Training") as bar:
        for sentence in tokenize_english(text):
            fourgram.train_sentence(tuple(sentence))
            bar()
    print(fourgram.perplexity(tokenized_text))
