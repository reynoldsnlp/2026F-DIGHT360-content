"""Redo Assignment 4 with a contextual model, then compare the two rankings.

    uv run assignments/05_sentence_transformer_embeddings.py

Same question as last week -- which verse of Matthew is most like Matthew 5:14
-- and a different way of answering it. Last week a verse was the average of
its word vectors, and every occurrence of `light` had the same vector no matter
what sentence it was sitting in. This week the vector for a word is computed
from the sentence it appears in, so `light` in "let there be light" and `light`
in "a light bag" are different vectors.

The interesting part of this assignment is not that the new model is better.
It is WHERE the two rankings disagree, and what that tells you about what each
one is measuring.

Read 05_sentence_transformer_embeddings.md first.
"""

import json
import math
from pathlib import Path

import requests
import spacy
from sentence_transformers import SentenceTransformer

SOURCE_URL = (
    "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/new-testament.json"
)
CACHE_PATH = Path("new-testament.json")

BOOK = "Matthew"
TARGET = "Matthew 5:14"

SPACY_MODEL = "en_core_web_lg"
# 384 numbers per sentence, about 90 MB of weights. Downloads itself the first
# time you run this and lives in ~/.cache/huggingface afterward, where it takes
# up closer to 180 MB once it has pulled the alternate formats it ships with.
TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


def load_scriptures() -> dict:
    """Download the scripture JSON once, then read it from disk forever after."""
    if not CACHE_PATH.exists():
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        CACHE_PATH.write_text(response.text, encoding="utf-8")
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def iter_verses(scriptures: dict, book_name: str):
    """Yield (reference, text) for every verse of one book, in reading order."""
    for book in scriptures["books"]:
        if book["book"] != book_name:
            continue
        for chapter in book["chapters"]:
            for verse in chapter["verses"]:
                yield verse["reference"], verse["text"]


def cosine(a, b) -> float:
    """Copy this in from your Assignment 4 script. It has not changed.

    It works on the transformer's vectors too -- they are 384 numbers instead
    of 300, but cosine similarity does not care how many numbers there are.
    """
    raise NotImplementedError("Copy cosine() from your Assignment 4 script.")


def spacy_vectors(docs: list) -> list:
    """Assignment 4's answer: average the word vectors, skipping stop words.

    Stop-word-filtered version -- the one that put Matthew 4:16 on
    top and left Matthew 5:13 down at 198th. If you preferred plain doc.vector
    last week, replace the body with

        return [doc.vector for doc in docs]

    and every number in the spaCy column moves. Try it once; it is one line and
    it changes the argument you are going to make.
    """
    vectors = []
    for doc in docs:
        tokens = [
            token
            for token in doc
            if token.has_vector
            and not token.is_stop
            and not token.is_punct
            and not token.is_space
        ]
        if not tokens:
            vectors.append(doc.vector)  # nothing left to average; fall back
        else:
            vectors.append(sum(token.vector for token in tokens) / len(tokens))
    return vectors


def transformer_vectors(texts: list) -> list:
    """This week's answer: let a transformer read each verse and encode it.

    YOUR TURN. This is three lines, and the markdown walks through all of them.

    Load the model with SentenceTransformer(TRANSFORMER_MODEL), then call its
    .encode() method on the whole list of texts at once. Pass
    normalize_embeddings=True and the vectors come back already scaled to
    length 1, which costs nothing and makes your cosine() a little faster.

    Encode the whole list in ONE call. Calling .encode() once per verse works
    and is roughly forty times slower -- the same lesson as nlp.pipe() in
    Assignments 2 through 4.
    """
    raise NotImplementedError("Write transformer_vectors(). See Part 3 of the markdown.")


def rank(target_index: int, vectors: list, verses: list) -> list:
    """Score every verse against the target. Returns [(score, reference, text)].

    Unchanged from Assignment 4 except that it now takes the vectors as an
    argument, so you can call it once per model.
    """
    scored = []
    for index, (reference, text) in enumerate(verses):
        if index == target_index:
            continue  # a verse is always a perfect match for itself
        scored.append((cosine(vectors[target_index], vectors[index]), reference, text))
    scored.sort(reverse=True)
    return scored


def report(label: str, scored: list, verses: list) -> None:
    """Print a top-10 and the spread, for one model."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    for position, (score, reference, text) in enumerate(scored[:10], start=1):
        print(f"{position:>3}. {score:.4f}  {reference:<14} {text[:70]}")

    middle = scored[len(scored) // 2][0]
    print(f"\n    best {scored[0][0]:.4f}   median {middle:.4f}   worst {scored[-1][0]:.4f}")

    # Where did Matthew 5:13 land? Last week this was the whole argument.
    for position, (score, reference, _text) in enumerate(scored, start=1):
        if reference == "Matthew 5:13":
            print(f"    Matthew 5:13 ranked {position} ({score:.4f})")
            break


def main() -> None:
    nlp = spacy.load(SPACY_MODEL)
    verses = list(iter_verses(load_scriptures(), BOOK))
    texts = [text for _reference, text in verses]
    spacy_docs = list(nlp.pipe(texts, disable=nlp.pipe_names))
    target_index = [reference for reference, _text in verses].index(TARGET)
    print(f"Read {len(verses)} verses from {BOOK}.")
    print(f"Target -- {TARGET}: {verses[target_index][1]}")

    spacy_ranked = rank(target_index, spacy_vectors(spacy_docs), verses)
    transformer_ranked = rank(target_index, transformer_vectors(texts), verses)

    report("spaCy: average of static word vectors", spacy_ranked, verses)
    report("Transformer: contextual sentence embedding", transformer_ranked, verses)

    # YOUR TURN, part 2: how much do the two rankings actually agree?
    #
    # Compare the top 10 of each. How many references appear in both? That one
    # number is the start of your write-up, and whichever way it comes out --
    # high or low -- the interesting question is which verses are NOT shared.
    #
    # print(set(r for _s, r, _t in spacy_ranked[:10]) & set(...))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# YOUR TURN, THE LONGER VERSION
#
# 1. DO NOT ASSUME THE NEW MODEL WINS.
#
#    It is newer, bigger, and slower, and none of those are arguments. Read
#    both top-10 lists as a person who knows Matthew. For each list, find the
#    one verse you would throw out, and say what put it there.
#
# 2. THE SCORES ARE NOT COMPARABLE ACROSS MODELS.
#
#    spaCy's best match scores around 0.83 and the transformer's around 0.41.
#    That does NOT mean spaCy is twice as confident. Different models spread
#    their cosine scores over different ranges, and the only honest way to
#    compare is WITHIN a model: how far is the best match above the median?
#    report() prints both numbers for exactly this reason. Use the ratio, not
#    the raw score, and say so in your write-up.
#
# 3. WHERE THEY DISAGREE IS THE ASSIGNMENT.
#
#    Find a verse that one model ranks in its top 20 and the other ranks below
#    200. There are several. For each one you find, decide which model you
#    think is right, and explain what the loser was looking at instead.
#
# 4. THINGS WORTH TRYING ONCE THE BASE VERSION WORKS.
#
#    - Swap all-MiniLM-L6-v2 for all-mpnet-base-v2 (420 MB, slower, generally
#      better). Does the top 10 change? Does your answer change?
#    - Run it on a verse of your own choosing instead of 5:14. Pick one whose
#      "obvious" partner verse you already know, and see whether either model
#      finds it.
#    - The model truncates input at 256 word pieces. Find the longest verses in
#      the New Testament and work out whether any of them are being cut off.
# ---------------------------------------------------------------------------
