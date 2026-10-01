"""Find the verse in Matthew most similar to Matthew 5:14.

    uv run assignments/04_spacy_embeddings.py

This is a STUB. It does the boring half for you -- download the scriptures,
pull out the 1,071 verses of Matthew, run them through spaCy -- and leaves the
interesting half, which is turning each verse into a vector and measuring the
angle between vectors, for you to write. Look for "YOUR TURN" below.

Assignments 2 and 3 described a text with features you chose by hand: count the
plural nouns, divide by the tokens. Every number in that pipeline meant
something you could say out loud. This week the description comes from a model
instead -- 300 numbers per verse, not one of which has a name -- and the thing
you get in exchange is that you no longer have to guess in advance which
properties of a verse matter.

Read 04_spacy_embeddings.md first. It builds up every piece this script needs.
"""

import json
import math
from pathlib import Path

import requests
import spacy

SOURCE_URL = (
    "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/new-testament.json"
)
CACHE_PATH = Path("new-testament.json")

BOOK = "Matthew"
TARGET = "Matthew 5:14"

# The large model, not the small one. en_core_web_sm has NO word vectors -- it
# will still answer similarity questions, cheerfully and wrongly, using the
# tagger's internal tensors. Part 0 of the markdown shows you what that looks
# like and why the large model is not optional here.
MODEL = "en_core_web_lg"


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
    """How closely do two vectors point in the same direction? 1.0 = identical.

    YOUR TURN (1 of 2).

    Part 2 of the markdown walks through this. In words: multiply the two
    vectors together number by number and add up the results (that is the "dot
    product"), then divide by the length of each vector. Dividing by the
    lengths is what makes this a measure of DIRECTION rather than size.

    The length of a vector is the square root of the sum of its squared
    numbers -- the Pythagorean theorem, with 300 sides instead of 2.

    Return 0.0 if either vector has length 0, or you will divide by zero. That
    happens for words spaCy has never seen, which come back as 300 zeros.
    """
    raise NotImplementedError("Write cosine(). See Part 2 of the markdown.")


def verse_vector(doc):
    """Turn one spaCy Doc into one 300-number vector.

    YOUR TURN (2 of 2).

    The one-line version is `return doc.vector`, and you should try that first
    so you have something to compare against. spaCy computes it by averaging
    the vectors of all the tokens in the doc.

    Then read Part 4 of the markdown and come back, because averaging *every*
    token means averaging `the`, `of`, `and` and `that`, which make up half of
    most verses and are the same half in every verse. Your second version
    should skip those. The attributes you need are `token.is_stop`,
    `token.is_punct`, `token.is_space` and `token.has_vector`.

    Whichever you choose, say which one in a comment, because the two give
    different answers and the assignment asks you to report both.
    """
    raise NotImplementedError("Write verse_vector(). See Part 4 of the markdown.")


def main() -> None:
    verses = list(iter_verses(load_scriptures(), BOOK))
    print(f"Read {len(verses)} verses from {BOOK}.")

    nlp = spacy.load(MODEL)
    # nlp.pipe() batches, same as in Assignments 2 and 3. All of Matthew runs
    # in about three seconds, so there is no excuse for nlp(text) in a loop.
    docs = list(nlp.pipe([text for _reference, text in verses], batch_size=200))

    vectors = [verse_vector(doc) for doc in docs]

    target_index = [reference for reference, _text in verses].index(TARGET)
    target_vector = vectors[target_index]

    # Score every verse against the target. One cosine per verse: 1,071 of
    # them, which takes well under a second.
    scored = []
    for index, (reference, text) in enumerate(verses):
        if index == target_index:
            continue  # a verse is always a perfect match for itself
        scored.append((cosine(target_vector, vectors[index]), reference, text))

    scored.sort(reverse=True)  # highest similarity first

    print(f"\nTarget -- {TARGET}: {verses[target_index][1]}\n")
    print(f"Most similar verses in {BOOK}:")
    for rank, (score, reference, text) in enumerate(scored[:10], start=1):
        print(f"{rank:>3}. {score:.4f}  {reference:<14} {text}")

    # The spread matters as much as the order -- see Part 6 of the markdown.
    middle = scored[len(scored) // 2][0]
    print(f"\nbest {scored[0][0]:.4f}   median {middle:.4f}   worst {scored[-1][0]:.4f}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# YOUR TURN, THE LONGER VERSION
#
# Getting the script to run is most of the work and none of the point. The
# point is what the ranking tells you, and whether you believe it.
#
# 1. RUN IT BOTH WAYS.
#
#    Once with `return doc.vector` and once with stop words filtered out. Save
#    both top-10 lists. They will not agree, and they will not disagree in a
#    small way -- one verse that places 11th under one definition places 198th
#    under the other. Neither list is a bug. You changed what "this verse is
#    about" means and the answer changed with it.
#
#    Look at the SPREAD of the scores too, not just the order. main() already
#    prints the best, median and worst for you. If the median verse scores
#    almost as high as the best one, the measure is barely discriminating and
#    the top of your list is mostly noise.
#
# 2. ARGUE WITH THE RESULT.
#
#    You know what Matthew 5:14 says. Read the top ten and ask, for each one,
#    whether a person would have put it there. Some will be obviously right.
#    At least one will be obviously wrong, and the interesting question is
#    always the same: what does it SHARE with the target that you were not
#    thinking of? A bag of averaged word vectors has no access to word order,
#    grammar, metaphor, or rhetorical structure. It sees vocabulary in a bowl.
#    Everything it gets right and everything it gets wrong follows from that.
#
#    Pay particular attention to Matthew 5:13, the verse immediately before the
#    target. Decide for yourself whether it belongs near the top, then find out
#    where your code put it under each definition, and explain the gap.
#
# 3. THINGS WORTH TRYING ONCE THE BASE VERSION WORKS.
#
#    - Weight the average by how rare each word is instead of filtering stop
#      words outright. Rare words carry more about a verse than common ones,
#      and a hard is_stop cutoff is a crude version of that idea.
#    - Lemmatize first (token.lemma_), so `saith` and `said` stop being
#      different points in space. Does it help? Say how you decided.
#    - Run the same code over all 27 books and ask which book's verses are
#      closest to Matthew 5:14 on average. You have just built a crude topic
#      detector.
#    - Compare against a sentence-transformer embedding (the `transformers`
#      package is already in pyproject.toml). Those models read the whole verse
#      in order instead of averaging it. Where do the two rankings differ, and
#      does the difference look like word order mattering?
# ---------------------------------------------------------------------------
