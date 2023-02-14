from typing import *
import re
from pprint import pprint

split_sentences_broad_regex = re.compile(r"(?:\n(?:[\r\t\f\v ]*)\n)+")
split_tokens_broad_regex = re.compile(r"\s+")
sentence_end_broad_regex = re.compile(r"((?:\?|!)+(?:[\"\']*)$)")

sentence_end_period_regex = re.compile(r"(\.(?:[\"\']*)$)")

abbreviation_regex = re.compile(r"(?:(?:[a-zA-Z]+\.){2,})|(?:[A-Z]\.)")

punctuation_split_regex = re.compile(r"([,\"(){}\[\]_;:*\/—]+)|(-(?:-+))")

url_regex = re.compile(
    r"(http[s]?:\/\/(www\.)?|ftp:\/\/(www\.)?|www\.){1}([0-9A-Za-z-\.@:%_\+~#=]+)+((\.[a-zA-Z]{2,3})+)(\/(.)*)?(\?(.)*)?")
hashtag_regex = re.compile(r"#(?:[a-zA-Z0-9]+)")
mention_regex = re.compile(r"@(?:[a-zA-Z0-9]+)")
num_regex = re.compile(r"([0-9]+,)*[0-9]+(\.([0-9]*))?(k|m|b|t)", re.IGNORECASE)

remove_regex = re.compile(r"(?:[*\"_{}\[\]—:]+)")

token_substitute = {
    url_regex: "<URL>",
    hashtag_regex: "<HASHTAG>",
    mention_regex: "<MENTION>"
}


def is_abbreviation(token: str) -> bool:
    if abbreviation_regex.search(token):
        return True

    if token.lower().replace("\"", "").replace("\'", "") in {"dr.", "mr.", "mrs."}:
        return True

    return False


def is_num(token: str) -> bool:
    return bool(num_regex.search(token))


def tokenize_english(text: str) -> list[list[str]]:
    sentences: list[str] = split_sentences_broad_regex.split(text)
    sentences = [sentence for sentence in sentences if sentence]
    sentences_tokens: list[list[str]] = [split_tokens_broad_regex.split(sentence) for sentence in sentences]
    parsed_sentence_tokens = []
    for sentence in sentences_tokens:
        tokens = []
        for token in sentence:
            for regex, sub in token_substitute.items():
                if regex.search(token):
                    token = regex.sub(token, sub)
                    break
            if punctuation_split_regex.search(token):
                tokens.extend(punctuation_split_regex.split(token))
                continue

            tokens.append(token)
        tokens = [token for token in tokens if token]
        if tokens:
            parsed_sentence_tokens.append(tokens)

    sentences_tokens = parsed_sentence_tokens
    parsed_sentence_tokens = []
    for sentence in sentences_tokens:
        tokens = []
        for token in sentence:

            if sentence_end_broad_regex.search(token):
                tokens.extend(sentence_end_broad_regex.split(token))
                tokens = [token for token in tokens if token]
                parsed_sentence_tokens.append(tokens)
                tokens = []
                continue

            if sentence_end_period_regex.search(token) and not is_abbreviation(token):
                tokens.extend(sentence_end_period_regex.split(token))
                tokens = [token for token in tokens if token]
                parsed_sentence_tokens.append(tokens)
                tokens = []
                continue

            tokens.append(token)
        tokens = [token for token in tokens if token]
        if tokens:
            parsed_sentence_tokens.append(tokens)
    parsed_sentence_tokens = [
        [remove_regex.sub(token, "") if remove_regex.search(token) else token for token in sentence] for sentence in
        parsed_sentence_tokens]

    parsed_sentence_tokens = [[token.lower() for token in sentence if token] for sentence in parsed_sentence_tokens]
    parsed_sentence_tokens = [sentence for sentence in parsed_sentence_tokens if sentence]
    return parsed_sentence_tokens


if __name__ == "__main__":
    with open("data/Ulysses - James Joyce.txt") as f:
        sentences_tokens = tokenize_english(f.read())
        print("\n".join([str(i) for i in sentences_tokens]))
