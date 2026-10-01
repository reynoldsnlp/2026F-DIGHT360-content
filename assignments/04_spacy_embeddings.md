# Assignment 4: Word and Document Embeddings with spaCy

**Goal:** stop describing a text with features you named and start describing it
with features nobody named -- then find out what you gave up in the trade.

In [Assignment 2](02-spacy-feature-extraction.md) and
[Assignment 3](03_readability_feature_extraction.py) every number in your
pipeline had a name. `plural_nouns_per_token` meant plural nouns per token. If a
verse scored high you could point at the plural nouns. That is a real virtue and
you are about to lose it.

This week a verse becomes 300 numbers, not one of which means anything you can
say out loud. What you get in exchange is that you stop having to guess in
advance which properties matter. Nobody decided that dimension 47 would track
royalty. It fell out of counting which words appear near which other words, over
billions of words of text, and it works better than the features you would have
written by hand.

By the end you will have written a script that finds the verse of Matthew most
similar to Matthew 5:14. Everything that script needs gets built up below, one
piece at a time.

## Part 0: Get a model that actually has vectors

This is not the usual "install the thing" step. Read it, because two of the
three obvious choices are broken for this assignment and both fail *silently*.

### `en_core_web_sm` has no vectors at all

The model you have used so far ships with zero word vectors:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
print(nlp.vocab.vectors.shape)   # (0, 0)
```

That `(0, 0)` is the whole table. And yet this works:

```python
a = nlp("Ye are the light of the world.")
b = nlp("Ye are the salt of the earth.")
print(a.similarity(b))   # 0.8203
```

A confident, plausible, entirely meaningless `0.8203`. With no vectors loaded
spaCy falls back on the tagger and parser's internal tensors, which were trained
to predict parts of speech and have no business answering questions about
meaning. It does warn you:

```
[W007] The model you're using has no word vectors loaded, so the result of the
Doc.similarity method will be based on the tagger, parser and NER, which may not
give useful similarity judgements.
```

Warnings scroll past. Numbers get written into reports. This is the exact shape
of mistake that is worth training yourself to catch: **the code ran, the output
was a number in the right range, but it was worthless.**

### `en_core_web_md` has vectors, but they are wrecked

The medium model looks like the obvious fix. It is 54 MB and reports a real
vector table.

```python
nlp = spacy.load("en_core_web_md")
print(nlp.vocab.vectors.shape)   # (20000, 300)
```

But 684,830 words are crammed into those 20,000 rows, and the rows themselves
are duplicated -- there are only 10,112 genuinely distinct vectors in there. The
consequence:

```python
print(nlp.vocab["dog"].similarity(nlp.vocab["cat"]))      # 0.9999999
print(nlp.vocab["dog"].similarity(nlp.vocab["banana"]))   # 0.2334
```

`dog` and `cat` are not merely similar in this model. They are *the same
vector*, number for number -- that `0.9999999` is a perfect 1.0 with a rounding
error on the end. No amount of cleverness downstream can tell them apart. Ask
this model for `king - man + woman` and it answers `hairdresser`.

### Use `en_core_web_lg`

```bash
uv sync
uv run python -m spacy download en_core_web_lg
```

That is 434 MB installed, which is why we did not start the semester with it.
Check it:

```python
nlp = spacy.load("en_core_web_lg")
print(nlp.vocab.vectors.shape)   # (342918, 300)
```

342,918 distinct vectors, 300 numbers each. Now `dog` and `cat` score 0.80 and
`dog` and `banana` score 0.24, which is the behavior everything below depends
on.

> **The lesson:** three models, same API, same method names,
> same plausible-looking floats. One of them answers your question, one answers
> a different question without telling you, and one answers wrongly with total
> confidence. Checking `nlp.vocab.vectors.shape` takes four seconds.

## Part 1: A word is 300 numbers

Every token now carries a vector:

```python
doc = nlp("light")
print(doc[0].vector.shape)    # (300,)
print(doc[0].vector[:6])      # first six numbers of the vector
```

```
(300,)
[ 0.0076215  0.29972    0.33304    0.30317   -0.23069   -0.18922  ]
```

That is it. That is a word embedding: a fixed list of 300 numbers standing in
for `light`. Three things to know and then we can use them.

- **Nobody chose what the dimensions mean.** There is no "royalty dimension" you
  can look up. The numbers came from a model that read a great deal of text and
  learned to put words that appear in similar contexts at similar coordinates.
  Dimension 47 means whatever it means.
- **The vector belongs to the word type, not the occurrence.** `light` in "let
  there be light" and `light` in "a light backpack" get the identical vector.
  These are *static* embeddings. Modern transformer models give each occurrence
  its own vector, which is a real improvement and a later lecture.
- **Not every word has one.** `token.has_vector` tells you. Misspellings, rare
  proper nouns, and anything that was not in the training vocabulary come back
  as all zeros, and a zero vector will quietly break similarity math.

```python
for token in nlp("Jesus wept beside Gergesenes"):
    print(f"{token.text:<12} {token.has_vector}  {token.vector_norm:.3f}")
```

## Part 2: Comparing two vectors with cosine similarity

Two words are similar when their vectors point the same direction. Not when
they are *close together* -- when they point the same way from the origin. The
measure for that is **cosine similarity**: the cosine of the angle between the
two vectors.

- Same direction → 1.0
- Perpendicular, nothing in common → 0.0
- Opposite directions → -1.0

In practice with word vectors you will see roughly 0.2 to 0.9 and almost never
anything negative, which is worth remembering when you read a score: 0.3 is not
"somewhat similar," it is closer to the floor. Here are some examples:

```
king      queen     0.7253
light     darkness  0.5569
king      banana    0.2179
light     banana    0.2891
```

Read those four numbers carefully, because the third one is the whole idea and
the second one is the whole catch.

`king`/`banana` at 0.22 is the model correctly reporting that these have nothing
to do with each other. Good. But `light`/`darkness` at 0.56 is *higher* than
`light`/`banana` -- the model rates antonyms as similar. That is not a bug, it
follows directly from how the vectors were built. Words that appear in similar
contexts get similar vectors, and `light` and `darkness` appear in nearly
identical contexts ("the light came", "the darkness came"). **Word embeddings
measure relatedness, not agreement.** They cannot tell "good" from "bad" or
"always" from "never", and any analysis you build on them inherits that.

spaCy will do this computation for you:

```python
doc = nlp("king queen")
print(doc[0].similarity(doc[1]))   # 0.7253
```

### Write it yourself

You need your own version for the assignment, so here it is in plain Python.
There is no library trick in it -- it is the definition, typed out:

```python
import math


def cosine(a, b):
    """How closely do two vectors point in the same direction?"""
    # Multiply the two vectors together number by number, add up the results.
    # This is the "dot product". Vectors that are big in the same places get a
    # big number; vectors that are big in different places get a small one.
    dot = sum(x * y for x, y in zip(a, b))

    # The length of a vector: square every number, add them up, square root.
    # That is the Pythagorean theorem with 300 sides instead of 2.
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))

    if not length_a or not length_b:
        return 0.0          # an unknown word is 300 zeros; do not divide by 0

    # Dividing by both lengths is what turns "how big and aligned" into "how
    # aligned", which is the whole point: direction, not size.
    return dot / (length_a * length_b)


doc = nlp("king queen")
print(cosine(doc[0].vector, doc[1].vector))   # 0.7253
```

Same `0.7253` spaCy gave you. Four lines of arithmetic, no magic.

One practical note. A `token.vector` is not a plain Python list -- it is an
array object, and it comes with a `.dot()` method that does that first
multiply-and-add step in compiled C instead of in a Python loop. Identical
answer, about seventy times faster, which starts to matter in Part 3 when you
run this against a quarter of a million words:

```python
def cosine(a, b):
    length_a = math.sqrt(float(a.dot(a)))   # a.dot(a) is the sum of squares
    length_b = math.sqrt(float(b.dot(b)))
    if not length_a or not length_b:
        return 0.0
    return float(a.dot(b)) / (length_a * length_b)
```

Note the trick in the first two lines: multiplying a vector by *itself* and
adding up the results gives you the sum of its squares, which is exactly what
goes under the square root. Use whichever version you like -- they return the
same numbers. The `float()` calls are only there to turn the array library's
own number type back into an ordinary Python float so it prints cleanly.

## Part 3: Analogies -- arithmetic on meaning

Here is the result that made these things famous. If the vectors really encode
something about meaning, then the *difference* between two vectors should encode
the relationship between the words. `king - man` ought to be "the royalty part,
minus the maleness part." Add `woman` back and you should land near `queen`.

You cannot just compute that vector, though -- you get 300 numbers, and there is
no word attached. You have to search the vocabulary for the word whose vector is
closest. So first build a search list: every lowercase alphabetic word spaCy has
a vector for, paired with its vector.

```python
import spacy

nlp = spacy.load("en_core_web_lg")

words = []
vectors = []
seen = set()

# nlp.vocab.vectors.key2row maps each known word to the row holding its vector.
# We walk that mapping, look the word back up, and keep the ones we want.
for key, row in nlp.vocab.vectors.key2row.items():
    word = nlp.vocab.strings[int(key)]
    if word.isalpha() and word.islower() and word not in seen:
        seen.add(word)
        words.append(word)
        vectors.append(nlp.vocab.vectors.data[row])

print(len(words), "candidate words")   # 247539
```

The vocabulary stores `queen`, `Queen` and
`QUEEN` as three separate entries pointing at the same vector, so without islower()
your results come out shouting and repeating themselves.

Now the search. The arithmetic is the easy part -- spaCy vectors support `+` and
`-` directly, so `king - man + woman` is written exactly the way you would write
it on paper:

```python
def analogy(a, b, c, n=5):
    """a is to b as ??? is to c.   king - man + woman = queen"""
    target = nlp.vocab[a].vector - nlp.vocab[b].vector + nlp.vocab[c].vector

    scored = []
    for word, vector in zip(words, vectors):
        if word in {a, b, c}:
            continue          # the inputs always score high; skip them
        scored.append((cosine(target, vector), word))

    scored.sort(reverse=True)     # highest similarity first
    return [(word, round(score, 3)) for score, word in scored[:n]]


for triple in [("king", "man", "woman"), ("paris", "france", "italy"),
               ("walking", "walk", "swim"), ("bigger", "big", "small")]:
    print(f"{triple[0]} - {triple[1]} + {triple[2]} =", analogy(*triple))
```

That is the `cosine()` you wrote in Part 2, called once per candidate word, and
an ordinary `sort()`. Nothing new.

This is also where the two versions of `cosine()` stop being equivalent in
practice. Measured on this exact loop: **0.3 seconds** per analogy with the
`.dot()` version, **14 seconds** with the plain one. Same answers either way --
you are asking Python to do 74 million multiplications, and the only question
is whether it does them one at a time in a Python loop or hands them to
compiled code in a batch. Use the plain version first so you can see what it is
doing, then switch.

```
king - man + woman =     [('queen', 0.788), ('prince', 0.64), ('kings', 0.621), ...]
paris - france + italy = [('rome', 0.7), ('milan', 0.674), ('milano', 0.664), ...]
walking - walk + swim =  [('swimming', 0.781), ('bathing', 0.629), ('swimmers', 0.625), ...]
bigger - big + small =   [('smaller', 0.868), ('larger', 0.817), ('sized', 0.659), ...]
```

Four different relationships -- gender, capital-of-country, verb inflection,
comparative adjective -- and the same subtraction recovers all four. Nobody
labeled any of them. They are a side effect of counting co-occurrences.

**Note the line that excludes the three input words.** It is not a cosmetic
touch. The inputs reliably outrank the answer, so without that filter the
"result" of every analogy is one of the words you typed in.

### Now watch it fail

Run these too, and do not skip them:

```
jerusalem - israel + egypt = [('egyptian', 0.647), ('cairo', 0.62), ('luxor', 0.572), ...]
moses - egypt + babylon    = [('larkin', 0.448), ('noah', 0.436), ('amos', 0.427), ...]
doctor - man + woman       = [('nurse', 0.702), ('doctors', 0.676), ('physician', 0.668), ...]
```

The first is half right: `cairo` is the answer, and it places second behind the
adjective `egyptian`. The second is nonsense -- the top hit is the surname of a
twentieth-century English poet. Analogy arithmetic works often enough to be
impressive in a lecture and not nearly often enough to be a tool you trust
unsupervised.

The third one is the one to sit with. `doctor - man + woman = nurse` is not a
glitch, and it is not the model being broken. The model is working exactly as
designed: it read a large amount of human writing and reproduced the
associations in it. The sexism is in the training data, which is to say in us,
and the embedding preserved it faithfully along with everything else. Every
system built on these vectors inherits that, usually without anyone checking.
If you take one thing from this assignment, make it this: **an embedding is a
compressed summary of whoever wrote the training corpus, and it will reproduce
their assumptions as confidently as it reproduces their grammar.**

## Part 4: From words to documents

Now the other half. You have a vector per word; you want a vector per verse.

spaCy gives you one for free:

```python
doc = nlp("Ye are the light of the world.")
print(doc.vector.shape)   # (300,)
```

Before you use it, find out what it actually is. Add up the token vectors and
divide by how many there were -- vectors support `+` and `/` the same way
numbers do, so this is the ordinary way you would average anything:

```python
token_vectors = [token.vector for token in doc]
average = sum(token_vectors) / len(token_vectors)

# Compare the two, number by number. They match.
print(all(abs(x - y) < 0.00001 for x, y in zip(average, doc.vector)))   # True
```

**`doc.vector` is the plain average of the token vectors.** No weighting, no
word order, no syntax. "Dog bites man" and "man bites dog" get byte-identical
document vectors. All the dependency parsing you did in Assignment 3 is thrown
away here.

That it works at all is a little surprising, and it does work:

```python
a = nlp("Ye are the light of the world.")
b = nlp("Ye are the salt of the earth.")
c = nlp("Judas went and hanged himself.")

print(a.similarity(b))   # 0.9441
print(a.similarity(c))   # 0.6224
```

The two Sermon on the Mount verses score far above the unrelated one. For
finding documents that are *about* the same thing, an average of word vectors is
a strong cheap baseline.

### The stop word problem

Look again at that `0.9441`, and at the fact that two unrelated verses still
managed `0.6224`. Here is why both numbers are so high. Count what is actually
going into the average for our target verse:

```python
doc = nlp("Ye are the light of the world. A city that is set on an hill cannot be hid.")
print([t.text for t in doc if not t.is_stop and not t.is_punct])
print([t.text for t in doc if t.is_stop or t.is_punct])
```

```
kept:    ['Ye', 'light', 'world', 'city', 'set', 'hill', 'hid']
dropped: ['are', 'the', 'of', 'the', '.', 'A', 'that', 'is', 'on', 'an', 'can', 'not', 'be', '.']
```

Seven words carry the meaning. Fourteen are `the`, `of`, `is`, `an`, `be` and
punctuation -- and they are the *same* fourteen that pad out every other verse in
the Bible. Averaging all 21 means two thirds of your "meaning" vector is
identical across the entire corpus, which drags every pair of documents toward
each other and compresses your scores into a narrow band near the top.

The fix is to average only the content words:

```python
def content_vector(doc):
    tokens = [t for t in doc
              if t.has_vector and not t.is_stop and not t.is_punct and not t.is_space]
    if not tokens:
        return doc.vector          # nothing left to average; fall back
    return sum(t.vector for t in tokens) / len(tokens)
```

Same averaging as above, just over a shorter list. Three details, all of which
matter. `has_vector` keeps unknown words -- which are 300 zeros -- from dragging
the average toward nothing. The empty-list guard is there because `sum([])` is
`0` and dividing that by `0` crashes; a handful of verses really do lose every
token to the filter. And note that `Ye` survives -- spaCy's stop list is modern
English and does not know that `ye` is a pronoun, which is the Assignment 2
problem again: a model trained on web text, pointed at 1611 English.

You will run the assignment both ways and compare. Do not assume the filtered
version is simply better; it changes what "about" means, and you will have to
look at the output to decide what it changed.

## Part 5: Ranking one verse against all the others

You now have a vector per verse and a way to compare two vectors. The last step
is to compare one verse against every other verse and sort the results.

That sounds expensive and is not. The assignment asks which verse is most like
Matthew 5:14 -- **one** verse against 1,070 others, so you call `cosine()` 1,070
times, not a million. That runs in under a second:

```python
target_vector = vectors[target_index]

scored = []
for index, (reference, text) in enumerate(verses):
    if index == target_index:
        continue          # a verse is always a perfect match for itself
    scored.append((cosine(target_vector, vectors[index]), reference, text))

scored.sort(reverse=True)     # highest similarity first

for score, reference, text in scored[:10]:
    print(f"{score:.4f}  {reference}")
```

Two things worth noticing, because both are easy to get wrong. Putting the
score **first** in each tuple is what makes the plain `sort()` work -- Python
sorts tuples by their first element, so you get the ordering for free without
writing a key function. And `reverse=True` is what gets you most-similar-first;
leave it off and you will be reading the ten verses *least* like your target,
which look wrong in a way that is surprisingly hard to notice.

Skipping the target itself matters too. It scores a perfect 1.0 against itself,
which is correct and useless, and if you forget to drop it then "the most
similar verse to Matthew 5:14" is Matthew 5:14.

You now have every piece the assignment needs.

## Part 6: The assignment

**Find the verse in the book of Matthew most similar to Matthew 5:14.**

> Ye are the light of the world. A city that is set on an hill cannot be hid.

Start from [`04_spacy_embeddings.py`](04_spacy_embeddings.py). It downloads and
caches the scriptures, pulls out all 1,071 verses of Matthew, and runs them
through `nlp.pipe()` -- the parts you have done before. Two functions are left
for you, both covered above:

| Function | Built in |
| --- | --- |
| `cosine(a, b)` | Part 2 |
| `verse_vector(doc)` | Part 4 |

The ranking and sorting in `main()` is already written for you, exactly as in
Part 5.

```bash
uv run assignments/04_spacy_embeddings.py
```

It takes about five seconds once the scriptures are cached.

### Run it both ways

Once with `return doc.vector`, once with stop words filtered out. Save both
top-10 lists. They will not agree, and the disagreement is the assignment.

Here is one anchor so you can tell a working script from a broken one. Under
**plain `doc.vector`**, the top match for Matthew 5:14 is Matthew 18:14 at
0.9537 -- a verse about the Father's will that none of these little ones should
perish, which has nothing to do with light, cities, or hills. Under **content
words only**, the top match is Matthew 4:16 at 0.8322:

> The people which sat in darkness saw great light; and to them which sat in the
> region and shadow of death light is sprung up.

That is a verse a person might well have chosen. The filter earned its keep.

### Look at the spread, not just the order

`main()` already prints the best, median and worst score for you. The median is
the one that tells the story:

| | best match | median verse | fraction scoring > 0.85 |
| --- | --- | --- | --- |
| plain `doc.vector` | 0.954 | 0.887 | 80% |
| content words only | 0.832 | 0.672 | 0% |

With plain `doc.vector`, the winning verse beats the *typical* verse by about
seven hundredths, and four out of five verses in Matthew score above 0.85. Those
scores are not measuring similarity to Matthew 5:14 so much as measuring the
shared mass of `the`, `of`, `and` and `that` sitting in every average. Filtering
stop words drops the median to 0.672 while the top stays at 0.832, which opens
up the gap the ranking depends on.

This generalizes well past this assignment: when a similarity measure returns
high scores for every pair, the first question is not "which is highest" but
"is this measure discriminating at all." A top-10 list always looks like a
result. Check the spread before you believe it.

### Then argue with the answer

You know what Matthew 5:14 says. Read your top ten and decide, verse by verse,
whether a person would have put it there. One in particular is worth chasing:
**Matthew 5:13**, the verse immediately before the target.

> Ye are the salt of the earth: but if the salt have lost his savour, wherewith
> shall it be salted?

To a human this is the obvious partner -- same speaker, same sentence frame
("Ye are the X of the Y"), same rhetorical move, adjacent in the text. Decide
where you think it should rank, then find out where your code put it. In my runs
it placed **11th** with plain `doc.vector` and **198th** with content words
filtered.

Sit with that, because it cuts against the tidy story in Part 4. The filter that
improved the top of the list pushed the human-obvious answer down 187 places.
Both things are true at once, and the reason is the same in both cases: once you
strip the function words, 5:13 and 5:14 share almost no vocabulary -- `salt`,
`earth` and `savour` against `light`, `world`, `city` and `hill`. What they share
is a *grammatical frame*, and a bag of averaged word vectors threw the grammar
away before it started. It cannot see parallelism. It can only see word choice.

## Part 7: What to bring to class

Come to class prepared with the following:

1. Your finished `04_spacy_embeddings.py`, with both versions of
   `verse_vector()` present (one active, one commented out, clearly labeled).
2. **Both top-10 lists** -- plain `doc.vector` and content-words-only -- each
   with its similarity scores, and the lowest similarity in each run.
3. For each of the two lists, pick the **one result you find least defensible**
   and be prepared to explain what that verse shares with Matthew 5:14 that put
   it there.
4. The rank of **Matthew 5:13** under both definitions. Why does stripping stop
   words move it so far?
5. In Assignments 2 and 3 you chose every feature and could explain every number.
   This week you chose none of them and can explain none of them, and the results
   are arguably better. What did you actually give up? Name one research question
   about the New Testament you could answer with `plural_nouns_per_token` that you
   could not answer with a 300-dimensional verse embedding -- and one that runs the
   other way.

## Reference

- [Word vectors and semantic similarity](https://spacy.io/usage/linguistic-features#vectors-similarity)
  -- the official walkthrough of `token.vector`, `doc.similarity`, and the
  caveats spaCy itself puts on them. Short, and read the warnings.
- [`Vectors` API](https://spacy.io/api/vectors) -- `key2row`, `most_similar`,
  and what the model actually stores.
- [spaCy models directory](https://spacy.io/models/en) -- compare `sm`, `md` and
  `lg` and note the vector counts listed for each. Part 0 in one table.
- [Mikolov et al. 2013, "Efficient Estimation of Word Representations in Vector
  Space"](https://arxiv.org/abs/1301.3781) -- the word2vec paper that introduced
  the analogy result. Sections 1 and 4 are readable without the math.
- [Bolukbasi et al. 2016, "Man is to Computer Programmer as Woman is to
  Homemaker?"](https://arxiv.org/abs/1607.06520) -- the paper behind the
  `doctor - man + woman = nurse` result, and what debiasing does and does not
  fix.
- [bcbooks/scriptures-json](https://github.com/bcbooks/scriptures-json) -- the
  same data as Assignment 2.
