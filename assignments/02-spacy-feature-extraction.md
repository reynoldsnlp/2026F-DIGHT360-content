# Assignment 2: Feature Extraction with spaCy

**Goal:** stop describing text with patterns you wrote by hand and start
describing it with features a model computed for you -- then find out where the
model is wrong, the same way you found out where your regex was wrong.

In [Assignment 1](01-regex.md) you tried to find plural nouns with a regular
expression and discovered you could not do it cleanly. This week a
part-of-speech tagger does it in one line. That is the good news. The bad news
is that the tagger is also wrong sometimes, and now the errors are somebody
else's and harder to see.

## Part 0: Install the model

`spacy` is already in this repository's `pyproject.toml`, but the library ships
without any language model -- you download those separately.

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

That pulls about 12 MB. `en_core_web_sm` is the *small* English pipeline:
tokenizer, tagger, parser, and named-entity recognizer, trained on modern web
text -- blogs, news, forums. Keep that last detail in mind all week: you are
about to point a model trained on modern English at 1611 English, and it will
cost you something.

Check that it loaded:

```bash
uv run python -c "import spacy; spacy.load('en_core_web_sm'); print('ok')"
```

## Part 1: Look before you count

Never count something you have not looked at first. Make a scratch file and run
one verse through the pipeline:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("And Jesus said unto them, Follow me, and I will make you fishers of men.")

for token in doc:
    print(f"{token.text:<10} {token.pos_:<8} {token.tag_}")
```

You get one row per token:

```
And        CCONJ    CC
Jesus      PROPN    NNP
said       VERB     VBD
unto       ADP      IN
them       PRON     PRP
,          PUNCT    ,
Follow     VERB     VB
me         PRON     PRP
...
fishers    NOUN     NNS
of         ADP      IN
men        NOUN     NNS
.          PUNCT    .
```

Three things to understand before you write anything else:

- **`doc` is iterable, and its items are `Token` objects.** `len(doc)` is your
  token count. Punctuation is a token too, which is a decision you have to make
  rather than inherit.
- **`token.pos_` is the coarse tag** (Universal POS: `NOUN`, `VERB`, `ADP`,
  `PROPN`, ...). There are about 17 of them.
- **`token.tag_` is the fine-grained tag** (Penn Treebank: `NN` singular noun,
  `NNS` plural noun, `NNP` singular proper noun, `NNPS` plural proper noun,
  `VBD` past tense, `VBZ` third-person singular, ...). There are about 50.

So `token.tag_ in {"NNS", "NNPS"}` is the entire plural-noun detector you spent
last week failing to write. Try it on the list of irregulars from Assignment 1
(`children`, `women`, `criteria`, `corpora`, `feet`) and on the traps
(`analysis`, `always`, `statistics`, `he runs`). Then try it on `ye shall
receive`, where spaCy tags `ye` as a plural noun, and on `Abraham begat Isaac`,
where it tags `begat` as a noun and finds no verb at all. The tagger is much
better than your regex. It is not right.

`token.pos_ == "VERB"` does **not**
include auxiliaries. In `they were astonished`, spaCy tags `were` as `AUX` and
`astonished` as `VERB`. In `I will make`, `will` is `AUX` and `make` is `VERB`.
Whether "verb" means `VERB` or `VERB or AUX` is your call -- but you have to
make the call out loud, in a comment, because the two definitions give
noticeably different numbers.

## Part 2: Extract the features

Write a script called `verse_features.py` that reads the whole New Testament,
computes three features for every verse, and writes them to JSON.

### The data

The same source as the [El/Yahweh
example](../content/regex/el_yhwh_merge.py) from class:

```
https://raw.githubusercontent.com/bcbooks/scriptures-json/master/new-testament.json
```

The repository at [bcbooks/scriptures-json](https://github.com/bcbooks/scriptures-json)
also has `old-testament.json`, `book-of-mormon.json`,
`doctrine-and-covenants.json`, and `pearl-of-great-price.json`. Use the New
Testament for the graded part so we are all looking at the same numbers.

The structure is books inside a top-level list, chapters inside books, verses
inside chapters:

```json
{
  "books": [
    {
      "book": "Matthew",
      "chapters": [
        {
          "chapter": 1,
          "verses": [
            {"reference": "Matthew 1:1", "text": "The book of the generation ...", "verse": 1}
          ]
        }
      ]
    }
  ]
}
```

**Download it once and cache it.** Do not re-fetch 6 MB on every run.

### The three features

For each verse, compute:

| Feature | What it is |
| --- | --- |
| `n_tokens` | how many tokens the verse has |
| `verbs_per_token` | verb count divided by token count |
| `plural_nouns_per_token` | plural-noun count divided by token count |

Two of the three are **rates**, not counts, and that is the point. Revelation
has longer verses than Romans; if you record raw verb counts you have mostly
recorded verse length three times over. Dividing by `n_tokens` is what makes the
last two features say something length does not already say.

Guard the division. A few verses tokenize to nothing once you drop punctuation,
and `ZeroDivisionError` will find them for you at verse 6,000.

### The script

```python
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
```

Run it:

```bash
uv run verse_features.py
```

It reads 7,957 verses and takes about 20 seconds on a laptop. The output starts
like this:

```json
[
  {
    "id": "Matthew 1:1",
    "book": "Matthew",
    "n_tokens": 16,
    "verbs_per_token": 0.0,
    "plural_nouns_per_token": 0.0
  },
  {
    "id": "Matthew 1:2",
    "book": "Matthew",
    "n_tokens": 14,
    "verbs_per_token": 0.0,
    "plural_nouns_per_token": 0.07142857142857142
  }
]
```

You may use AI to help write the script, but you must understand and be able to
explain every part of it.


## Part 3: What else could you count?

This is the part that matters, and it needs no code -- not yet. In Part 4 you
build three of whatever you come up with here, so propose things you would
actually want to try.

Imagine the real task: given one verse, with no reference attached, **predict
which book of the New Testament it came from.** Your three features are a start
-- they can already tell a gospel from an epistle -- but they will never tell
Matthew from Mark.

**Write down at least eight more features you could extract.** For each one,
give me three things in a sentence or two:

- **What it is.** Say it precisely enough that two people would compute the same
  number. "Lots of names" is not a feature; "proper nouns per token" is.
- **How you would get it.** Which spaCy attribute, or which other method?
  (`token.ent_type_`, `token.lemma_`, `token.dep_`, `token.morph`,
  `doc.sents`, `token.is_stop`, or a plain regex, or a lookup table you build
  by hand.)
- **Why you think it varies by book.** What do you actually believe about these
  texts that makes you expect the number to move? This is the part people skip
  and it is the only part that is scholarship.

Then sort your eight into two piles and label them:

- **Topic features** -- they work because different books are *about* different
  things (place names, `ent_type_ == "PERSON"`, mentions of the sea).
- **Style features** -- they work because different books are *written*
  differently (sentence length, subordination depth, pronoun rate, function-word
  frequencies).

Finish with a paragraph on this question: if a classifier using your features
hits 80% accuracy, what have you actually learned about the New Testament? Is a
model that spots Revelation by the word `beast` telling you something about
authorship, about genre, about subject matter, or about nothing at all? Would
your answer change if the feature were `and`-per-token instead?

Keep your list. We build the classifier in a few weeks, and you will use it.

## Part 4: Write the functions

Now build three of them.

Your Part 2 script computes features inline, in the middle of the loop. That is
fine for three features and unworkable for twelve: every new feature means
editing the loop, the counters, and the dictionary, and you cannot test any of
them in isolation. Refactor before you add anything.

### One feature, one function

Give every feature the same signature -- takes a `Doc`, returns a `float` -- and
they become interchangeable:

```python
import math

from spacy.tokens import Doc


def content_tokens(doc: Doc) -> list:
    """Every feature starts here, so the decision lives in exactly one place."""
    return [token for token in doc if not token.is_punct and not token.is_space]


def rate(count: int, total: int) -> float:
    """Divide, but survive the empty verse."""
    return count / total if total else 0.0


def n_tokens(doc: Doc) -> float:
    return float(len(content_tokens(doc)))


def verbs_per_token(doc: Doc) -> float:
    """VERB excludes auxiliaries: 'will' and 'were' are AUX, not VERB."""
    tokens = content_tokens(doc)
    return rate(sum(1 for token in tokens if token.pos_ == "VERB"), len(tokens))


def plural_nouns_per_token(doc: Doc) -> float:
    tokens = content_tokens(doc)
    return rate(sum(1 for token in tokens if token.tag_ in {"NNS", "NNPS"}), len(tokens))


def proper_nouns_per_token(doc: Doc) -> float:
    """Names and places per token. Expect narrative to outscore epistle."""
    tokens = content_tokens(doc)
    return rate(sum(1 for token in tokens if token.tag_ in {"NNP", "NNPS"}), len(tokens))
```

Then name them once, in a registry, and the loop stops caring how many there
are:

```python
FEATURES = {
    "n_tokens": n_tokens,
    "verbs_per_token": verbs_per_token,
    "plural_nouns_per_token": plural_nouns_per_token,
    "proper_nouns_per_token": proper_nouns_per_token,
}


def features_for(doc: Doc) -> dict:
    return {name: function(doc) for name, function in FEATURES.items()}
```

Which turns the body of your loop into one line:

```python
for (book, reference, _text), doc in zip(verses, nlp.pipe(texts, batch_size=200)):
    rows.append({"id": reference, "book": book} | features_for(doc))
```

Adding a feature is now one function and one line in `FEATURES`. That is the
whole point of the refactor: the cost of trying an idea drops to almost nothing,
so you will try more ideas.

`proper_nouns_per_token` above is a worked example of a *new* feature, written in
the new shape. Read it, run it, then write your own.

### Check them against your own hands

A feature function that returns a plausible number for 7,957 verses and the
wrong number for all of them looks exactly like a correct one. So count one
verse yourself and make the code agree:

```python
VERSE = "And Jesus said unto them, Follow me, and I will make you fishers of men."


def check(nlp) -> None:
    """Hand-count one verse, then make the code agree with you.

    Counting by hand: 15 tokens once punctuation is dropped; the verbs are
    said / Follow / make (will is AUX); the plural nouns are fishers / men;
    the only proper noun is Jesus.
    """
    doc = nlp(VERSE)
    assert n_tokens(doc) == 15
    assert math.isclose(verbs_per_token(doc), 3 / 15)
    assert math.isclose(plural_nouns_per_token(doc), 2 / 15)
    assert math.isclose(proper_nouns_per_token(doc), 1 / 15)
    print("all feature checks passed")
```

Call `check(nlp)` at the top of `main()`. It costs one verse of runtime and it
will catch you the day you write `NNS` where you meant `NNP`.

### Your three

Pick **three** of the features you proposed in Part 3 and write them as
functions in this shape. To keep you from writing the same feature three times,
take one from each row:

| Take one that uses | For example |
| --- | --- |
| `token.pos_` or `token.tag_` | pronoun rate, adjective rate, past-tense rate |
| `token.lemma_`, `token.is_stop`, or `token.ent_type_` | type/token ratio, stopword rate, `PERSON` entities per token |
| `token.dep_`, `doc.sents`, or `token.morph` | mean sentence length, subordinate clauses per token, first-person verbs per token |

For each of your three:

1. **Write the function**, with a docstring saying what it counts and what you
   expect it to do across books. Write the expectation *before* you run it.
2. **Add a hand-checked assertion** to `check()`, using a verse you counted
   yourself. Not the one above -- pick your own.
3. **Add it to `FEATURES`** and re-run over the whole New Testament.

You may use AI to help, but you must be able to explain every line, and the
hand-count in step 2 has to be yours.

### Did it work?

Average all seven features by book. Here is what the four above look like, so
you can check your pipeline against mine:

```
book              n_tokens  verbs/tok  propn/tok
Matthew             22.162      0.149      0.060
Mark                22.410      0.158      0.047
Luke                22.586      0.155      0.053
John                21.776      0.151      0.080
Acts                24.120      0.152      0.079
Romans              21.797      0.111      0.070
1 Corinthians       21.728      0.119      0.052
1 Timothy           19.894      0.125      0.046
Titus               19.565      0.116      0.053
Hebrews             22.789      0.128      0.044
James               21.361      0.126      0.036
Revelation          29.710      0.117      0.041
```

The gospels and Acts run near 0.15 verbs per token; Romans and Hebrews near
0.12. Stories are made of events, letters are made of claims, and a verb rate
tells them apart without knowing a word of the content. None of that required
anyone to read the New Testament, which should make you a little uneasy as well
as a little pleased.

Then run one more check on each of your three, and this is the part that
separates a feature from a number:

```python
# Does this feature say anything that n_tokens did not already say?
correlation = statistics.correlation(
    [row["n_tokens"] for row in rows],
    [row["your_feature"] for row in rows],
)
```

Here is why you should care. Two of the features suggested above are traps:

| Feature | Correlation with `n_tokens` |
| --- | --- |
| `verbs_per_token` | -0.100 |
| `proper_nouns_per_token` | -0.081 |
| `pronouns_per_token` | -0.029 |
| `type_token_ratio` | **-0.536** |
| `mean_sentence_length` | **+0.839** |

`mean_sentence_length` correlates 0.84 with verse length because most verses are
one sentence -- you have rewritten `n_tokens` in a costlier font. And
`type_token_ratio` sounds like a measure of vocabulary richness, but at -0.54 it
is largely reporting that long verses repeat words. Revelation scores lowest on
it (0.744) for the same reason it scores highest on verse length. A classifier
would happily use it and you would happily misread the result.

Report the correlation for each of your three, and say in a sentence whether the
feature is telling you something new or telling you about length again.

## Part 5: What to turn in

Submit in Learning Suite:

1. `verse_features.py`, refactored as in Part 4, with all seven feature
   functions and a `check()` that passes.
2. The first 20 entries of your `verse_features.json` (not the whole file), plus
   your per-book average table for all 27 books and all seven features.
3. Your feature list from Part 3 -- eight or more, each with its definition, its
   method, and its rationale; sorted into topic and style; ending with the
   80%-accuracy paragraph.
4. For each of the three features you implemented: the expectation you wrote
   down *before* running it, the per-book numbers you got, its correlation with
   `n_tokens`, and a sentence on whether it earned its place.

And a short note (3-5 sentences):

- Which of your three features behaved least like you expected, and what do you
  now think is going on in the text?
- Which of the eight you proposed do you still most expect to fail, and why?

## Reference

- [spaCy 101](https://spacy.io/usage/spacy-101) -- read "Linguistic annotations"
  and "Architecture". Twenty minutes, and the rest of the docs stop being
  cryptic.
- [`Token` API](https://spacy.io/api/token#attributes) -- the full attribute
  list. Skim it before Part 3; you cannot propose features using attributes you
  do not know exist.
- [POS tag scheme](https://spacy.io/usage/linguistic-features#pos-tagging) --
  what every `pos_` and `tag_` value means.
- [Processing pipelines](https://spacy.io/usage/processing-pipelines) -- what
  `nlp.pipe()` does and how to switch off components you are not using.
- [bcbooks/scriptures-json](https://github.com/bcbooks/scriptures-json) -- the
  data. Open the JSON and look at it before you write the loop; same rule as
  last week.
- [el_yhwh_merge.py](../content/regex/el_yhwh_merge.py) -- the class example
  that loads and caches this same data, and a reminder of what the hand-written
  version of feature extraction looks like.
