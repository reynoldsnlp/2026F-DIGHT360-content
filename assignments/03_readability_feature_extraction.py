"""Extract readability features for every article in the OneStopEnglish corpus.

    uv run assignments/03_readability_feature_extraction.py

Assignment 2 one size up. There the label was a lookup table -- five books are
narrative, the rest are not -- and there were two answers. Here the label is an
ordinal scale with three rungs (elementary, intermediate, advanced), written by
hand by teachers at *onestopenglish.com*, who rewrote the same Guardian article
three times for three levels of English learner.

That last detail is what makes this corpus worth using. The same 189 articles
appear at all three levels, so the topic, the facts and the proper nouns are
held constant across the classes. "Amazon" shows up in an elementary text as
often as in an advanced one. A classifier cannot cheat by learning subject
matter; the only signal left is *how the thing is written*. That is exactly
what a readability feature is supposed to measure, and it is why your features
have to do real work here.

Writes readability_features.json, which content/machine_learning/multiclass_classifier.py reads.
"""

import json
import re
import zipfile
from pathlib import Path

import requests
import spacy

SOURCE_URL = (
    "https://codeload.github.com/nishkalavallabhi/OneStopEnglishCorpus/zip/refs/heads/master"
)
CACHE_PATH = Path("OneStopEnglishCorpus.zip")
OUTPUT_PATH = Path("readability_features.json")

# The corpus stores one directory per level, and encodes the level again in
# every filename ("Amazon-ele.txt"). We read the suffix, not the directory.
LEVELS = {"ele": "elementary", "int": "intermediate", "adv": "advanced"}

# Three landmines in this corpus, all of which silently corrupt your features if
# you ignore them. None of them are linguistic. All of them are artifacts of how
# somebody saved the files, and all three leak the label.
#
# First: every single -int.txt file opens with a bare line reading
# "Intermediate". The -ele and -adv files do not. Leave it in and that one word
# becomes a perfect giveaway for exactly one class -- your classifier scores
# beautifully and has learned nothing but a stray header in the corpus.
#
# Second: some files carry a UTF-8 byte-order mark and some do not. Decoding
# with plain "utf-8" leaves a stray ﻿ glued to the first word of about half
# the corpus, so "When" and "﻿When" become two different types and your
# type/token ratio drifts. "utf-8-sig" strips the mark when present and is
# harmless when it is not.
#
# Third, and much the worst: the intermediate files have been run through
# something that stripped every non-ASCII character. Elementary and advanced
# texts contain curly apostrophes, curly quotes, en dashes and pound signs;
# intermediate texts contain none, and their contractions have simply lost the
# apostrophe -- "what's" is stored as "whats". The result is that a single
# feature asking "does this document contain any non-ASCII character at all?"
# identifies the intermediate class with 100% accuracy, and scores 0.67 on the
# three-way problem, which is better than the three real linguistic features in
# this file manage. A model trained on unnormalized text is mostly a very
# expensive character-encoding detector.
HEADER_LINES = {level.lower() for level in LEVELS.values()}


def normalize(text: str) -> str:
    """Level the character-set playing field described above.

    The only way to compare these three classes fairly is to do to every
    document what the corpus already did to the intermediate ones: fold curly
    punctuation to ASCII, drop what will not fold, and delete apostrophes so
    that "what's" and "whats" are the same string everywhere.

    This is lossy on purpose, and it costs you something real. "whats" is not a
    word, spaCy will tag it as one token where "what's" would have been two, and
    any feature you might have built on contractions or quotation is now off the
    table.

    Measured honestly: the three features below barely notice this cleanup,
    because lengths and ratios are not where the artifact lives -- turn it off
    and the scores move by well under a point. The reason to do it anyway is
    that it disarms the trap BEFORE you go looking for features that would
    spring it, and a fair number of obvious readability features are exactly the
    kind that would: anything counting punctuation, contractions, quotation,
    character classes, or unusual word shapes. Those would not be measuring
    reading difficulty here. They would be measuring which directory the file
    came out of.
    """
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[–—]", " ", text)  # en/em dash, deleted in the int files
    text = text.encode("ascii", "ignore").decode()  # curly quotes, £, €, °, ...
    return text.replace("'", "")


def load_corpus() -> zipfile.ZipFile:
    """Download the corpus zip once, then read it from disk forever after."""
    if not CACHE_PATH.exists():
        print(f"Downloading corpus to {CACHE_PATH} (~25 MB, one time only)...")
        response = requests.get(SOURCE_URL, timeout=300)
        response.raise_for_status()
        CACHE_PATH.write_bytes(response.content)
    return zipfile.ZipFile(CACHE_PATH)


def iter_documents(archive: zipfile.ZipFile):
    """Yield (slug, level, text) for all 567 articles, in a stable order.

    The slug is the article name shared by all three levels -- "Amazon" for
    Amazon-ele.txt, Amazon-int.txt and Amazon-adv.txt. Carry it into the output:
    the classifier uses it to keep all three versions of an article on the same
    side of the train/test split.

    The usual argument for that is leakage -- siblings on both sides let a model
    memorize the test set's topics and post a flattering score. What actually
    happens here is stranger, and worth seeing for yourself. Because every
    article exists at all three levels, topic is balanced across the classes by
    construction, so memorizing it cannot help. For a held-out sibling it is
    worse than useless: the model has already seen those exact words carrying
    the other two labels, so it learns to rule out the correct answer. Measured
    on tf-idf features over these texts, a group-aware split scores 0.77 and a
    random split scores 0.28 -- below chance, on the same data and the same
    model.

    With the three structural features below, which carry no topic information
    at all, the two splits land within a point of each other. Group splitting is
    cheap insurance you will start needing the moment your features touch
    vocabulary, which several of the ten you are about to write will.
    """
    for name in sorted(archive.namelist()):
        # Int-Txt contains a nested duplicate copy of itself (Int-Txt/Int-Txt).
        # Counting the slashes keeps the intermediate level from being read
        # twice, which would quietly double its weight in the training data.
        if not name.endswith(".txt") or name.count("/") != 3:
            continue
        if "/Texts-SeparatedByReadingLevel/" not in name:
            continue

        stem = Path(name).stem  # "Amazon-ele"
        slug, _, suffix = stem.rpartition("-")
        if suffix not in LEVELS:
            continue

        text = archive.read(name).decode("utf-8-sig")
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in HEADER_LINES:
            lines = lines[1:]  # drop the giveaway header described above
        yield slug, LEVELS[suffix], normalize("\n".join(lines)).strip()


def tokens(doc) -> list:
    """The words of a doc: everything that is not punctuation or whitespace.

    Every feature below counts *something per something*, and this is the
    denominator for most of them. Defining it in one place means that when you
    decide numbers are not words, you change your mind once instead of ten times.
    """
    return [token for token in doc if not token.is_punct and not token.is_space]


def letters_per_word(doc) -> float:
    """Mean letters per word.

    The word-length half of nearly every readability formula ever published.
    Flesch-Kincaid and friends prefer syllables, but syllables need a
    pronouncing dictionary and letters do not, and the two correlate strongly.
    Counting only .isalpha() characters keeps "2014" and "$30m" from being
    scored as long words.
    """
    words = tokens(doc)
    if not words:
        return 0.0
    letters = sum(sum(char.isalpha() for char in token.text) for token in words)
    return letters / len(words)


def words_per_sentence(doc) -> float:
    """Mean sentence length in words -- the other half of the classic formulas.

    doc.sents needs a parser, so this only works with a model that has one
    (en_core_web_sm does). Sentences that are nothing but punctuation are
    dropped so a stray bullet or dash cannot count as a sentence and drag the
    average down.
    """
    sentences = [sent for sent in doc.sents if tokens(sent)]
    if not sentences:
        return 0.0
    return len(tokens(doc)) / len(sentences)


def type_token_ratio(doc) -> float:
    """Distinct word forms divided by total words -- a measure of vocabulary variety.

    Case-folded, so "The" and "the" are one type rather than two.

    Fair warning, and a good thing to think about before you trust this number:
    TTR falls as documents get longer, because every text eventually runs out of
    new words while it never runs out of room for old ones. So on a corpus where
    the advanced articles are also the longest, TTR partly measures length --
    and you will not be able to tell which of the two the classifier is using.
    Look up MTLD, MATTR and the moving-average family for fixes.
    """
    words = tokens(doc)
    if not words:
        return 0.0
    return len({token.lower_ for token in words}) / len(words)


# Register every feature here and main() picks it up automatically: one entry
# per column in readability_features.json, and the classifier reads whatever it
# finds. Adding a feature means writing a function and adding one line.
FEATURES = {
    "letters_per_word": letters_per_word,
    "words_per_sentence": words_per_sentence,
    "type_token_ratio": type_token_ratio,
}


def main() -> None:
    documents = list(iter_documents(load_corpus()))
    levels = sorted({level for _slug, level, _text in documents})
    print(f"Read {len(documents)} documents across {len(levels)} levels: {', '.join(levels)}")

    nlp = spacy.load("en_core_web_sm")
    texts = [text for _slug, _level, text in documents]

    rows = []
    # These documents are full articles, not verses -- a few thousand words each
    # rather than a few dozen. Smaller batches, and expect this to take a minute.
    for (slug, level, _text), doc in zip(documents, nlp.pipe(texts, batch_size=20)):
        row = {"id": f"{slug}-{level}", "group": slug, "level": level}
        row.update({name: feature(doc) for name, feature in FEATURES.items()})
        rows.append(row)

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows with {len(FEATURES)} features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# YOUR TURN
#
# Three features is a warm-up, not an experiment. Two of them are word length
# and sentence length, which is to say this script currently reimplements a
# formula from 1948. You can do better, and the point of the exercise is to find
# out *how much* better, and which ideas actually pay.
#
# 1. RESEARCH FIRST, CODE SECOND.
#
#    Use an AI assistant (Claude, ChatGPT, Copilot -- whatever you have) as a
#    research tool here. This is a genuinely good use for one: readability has a
#    century of literature behind it, spread across education, psycholinguistics
#    and NLP, and no textbook chapter collects it all. Questions worth asking:
#
#      - "What text features predict reading difficulty? Organize the answer by
#         lexical, syntactic, and discourse-level features."
#      - "What do modern NLP approaches to automatic readability assessment use
#         that traditional formulas like Flesch-Kincaid do not?"
#      - "Which of these can I compute from a spaCy Doc with pos_, tag_, dep_,
#         morph, lemma_ and the dependency tree? Which need outside data?"
#      - "What is the difference between MTLD, MATTR and plain TTR, and why do
#         corpus linguists distrust TTR?"
#
#    Then push back on what it tells you. Ask for the actual papers, and check
#    that they exist -- assistants invent citations, and a confidently named
#    study that does not exist is worse than no answer. Ask which features are
#    known to be *redundant* with each other, because you will meet that problem
#    again in the feature-evaluation step of the classifier. And ask what a
#    feature will do on a 500-word document versus a 50-word one: several
#    popular measures are length-dependent, as the TTR docstring above warns.
#
#    Notice also what the AI cannot tell you: whether a feature helps *on this
#    corpus*. That is what the classifier is for. Research narrows the search;
#    the experiment decides.
#
# 2. IMPLEMENT TEN NEW FEATURE FUNCTIONS.
#
#    Write ten new functions below, each following the pattern above exactly:
#    takes a spaCy Doc, returns a single float, guards against division by zero,
#    and carries a docstring saying what it measures and why you expect it to
#    track reading difficulty. Then add all ten to the FEATURES dict.
#
#    Some constraints, so that the results mean something:
#
#      - Make them RATES, not raw counts, wherever you can. These articles vary
#        a lot in length, and a raw count of subordinate clauses mostly measures
#        how long the document is. "Per sentence" or "per token" is comparable
#        across documents; a bare total is not.
#      - Cover more than one level of description. Ten flavors of word length
#        are not ten features -- they are one feature measured ten ways, and the
#        classifier's feature evaluation will tell you so in a way you will not
#        enjoy. Reach for the syntax spaCy already gives you: dependency
#        distance, tree depth, subordination, passives, clause types,
#        pronoun and connective use, named entity density, verb tense and mood.
#      - If a feature scores suspiciously well, suspect the corpus before you
#        congratulate yourself. Read normalize() above: this data had three
#        separate ways of leaking the label through nothing but file formatting,
#        and one throwaway feature built on the worst of them identified a whole
#        class perfectly while knowing nothing about language. Anything you
#        build on punctuation, capitalization, whitespace or character classes
#        deserves that second look.
#      - At least one should require something spaCy alone does not give you --
#        a word frequency list, an age-of-acquisition norm, an academic word
#        list. Feature engineering usually means bringing outside knowledge to
#        the text, and these are the features that tend to earn their keep.
#      - Write down your prediction before you run anything: which of your ten
#        do you think will win, and why? Being wrong about that in an
#        interesting way is a perfectly good result to write up.
#
# 3. THEN RUN THE CLASSIFIER.
#
#        uv run assignments/03_readability_feature_extraction.py
#        uv run content/machine_learning/multiclass_classifier.py readability_features.json
#
#    and read its last section carefully. It reports each feature three
#    different ways, and when those three disagree, the disagreement is the
#    finding worth writing about.
# ---------------------------------------------------------------------------
