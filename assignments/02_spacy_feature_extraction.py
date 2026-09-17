"""Count three simple features for every verse in the New Testament."""

import json
from pathlib import Path

import requests
import spacy

SOURCE_URL = (
    "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/new-testament.json"
)
CACHE_PATH = Path("new-testament.json")
OUTPUT_PATH = Path("verse_features.json")


def load_scriptures() -> dict:
    """Download the scripture JSON once, then read it from disk forever after."""
    if not CACHE_PATH.exists():
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        CACHE_PATH.write_text(response.text, encoding="utf-8")
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def iter_verses(scriptures: dict):
    """Yield (book, reference, text) for every verse, in reading order."""
    for book in scriptures["books"]:
        for chapter in book["chapters"]:
            for verse in chapter["verses"]:
                yield book["book"], verse["reference"], verse["text"]


def main() -> None:
    verses = list(iter_verses(load_scriptures()))
    print(f"Read {len(verses)} verses.")

    nlp = spacy.load("en_core_web_sm")
    texts = [text for _book, _reference, text in verses]

    rows = []
    # nlp.pipe() runs the model in batches. Calling nlp(text) in a loop does the
    # same work several times slower; at 8,000 verses you notice.
    for (book, reference, _text), doc in zip(verses, nlp.pipe(texts, batch_size=200)):
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        n_tokens = len(tokens)
        # "Verb" here means pos_ == VERB, which EXCLUDES auxiliaries (be, have,
        # will, shall). Change this line to include AUX and the numbers move.
        n_verbs = sum(1 for token in tokens if token.pos_ == "VERB")
        n_plural_nouns = sum(1 for token in tokens if token.tag_ in {"NNS", "NNPS"})

        rows.append(
            {
                "id": reference,
                "book": book,
                "n_tokens": n_tokens,
                "verbs_per_token": n_verbs / n_tokens if n_tokens else 0.0,
                "plural_nouns_per_token": n_plural_nouns / n_tokens if n_tokens else 0.0,
            }
        )

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
