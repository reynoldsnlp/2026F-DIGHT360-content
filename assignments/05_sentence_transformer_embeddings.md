# Assignment 5: Contextual Embeddings with Sentence Transformers

**Goal:** ask last week's question again with a model that reads the sentence
instead of averaging it -- then find out which of the two answers you believe.

[Assignment 4](04_spacy_embeddings.md) ended with a complaint. You found the
verse of Matthew most like Matthew 5:14, and the ranking was defensible but
deaf to things a human reader notices instantly. Two failures in particular:

- **Matthew 5:13** -- "Ye are the salt of the earth," the obvious partner verse
  -- came 198th, because it shares a *grammatical frame* with 5:14 and almost
  no vocabulary.
- **"Dog bites man" and "man bites dog"** got byte-identical vectors, because
  averaging throws away word order.

Both failures have the same cause. In spaCy, a word's vector is fixed before
the sentence exists. `light` in "let there be light" and `light` in "a light
bag" are the same 300 numbers, and a verse is just the average of numbers that
were decided in advance.

This week the vector for a word is computed **from the sentence it sits in**.
That is the whole idea, and the name for it is a *contextual* embedding.

## Part 0: Install

```bash
uv sync
```

`sentence-transformers` is now in `pyproject.toml`, and it brings PyTorch with
it, so expect this to take a few minutes the first time. The model downloads
itself the first time you run the script -- about 90 MB, cached in
`~/.cache/huggingface` afterward.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.get_embedding_dimension())   # 384
print(model.max_seq_length)              # 256
```

384 numbers per sentence this time rather than 300. Nothing in your code cares
-- cosine similarity does not know or need to know how many numbers it is
given.

## Part 1: The same word, two different vectors

Start by confirming the problem. In spaCy, `light` is `light`:

```python
import spacy

nlp = spacy.load("en_core_web_lg")
a = nlp("And God said, Let there be light.")
b = nlp("She packed a light bag for the trip.")

light_a = [t for t in a if t.text == "light"][0]
light_b = [t for t in b if t.text == "light"][0]
print(light_a.similarity(light_b))   # 1.0
```

Exactly `1.0`. Not "very similar" -- *the same vector*, looked up from the same
table, with the sentence playing no part at all. Illumination and lack-of-weight
are one word to spaCy.

Now the same experiment with a transformer. This takes a few more lines because
we want the vector for one *token* rather than the whole sentence:

```python
import torch
from transformers import AutoTokenizer, AutoModel

name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)


def token_vector(sentence, word):
    """The vector this model gives `word` *in this particular sentence*."""
    encoded = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():                      # we are not training, just asking
        output = model(**encoded).last_hidden_state[0]
    pieces = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    position = [i for i, piece in enumerate(pieces) if piece == word][0]
    return output[position]


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
```

Try it on a word with two clearly different senses:

```python
animal   = "The bat flew out of the cave at dusk."
baseball = "He swung the bat and hit a home run."
attic    = "A bat roosted in the attic all winter."

print(cosine(token_vector(animal, "bat"), token_vector(baseball, "bat")))  # 0.7836
print(cosine(token_vector(animal, "bat"), token_vector(attic, "bat")))     # 0.9309
```

Two sentences about animals: **0.93**. An animal and a piece of sporting goods:
**0.78**. The model has never been told that `bat` is ambiguous and has no
dictionary of senses. The vector simply came out differently because the
surrounding words were different.

Run it on `bass` (a fish and a singing voice) and `crane` (a bird and a
machine) and you get the same pattern. Note also what you do *not* get: the
different-sense pair still scores 0.78, not 0.2. These are not two separate
dictionary entries. They are the same starting point, pulled in different
directions by their neighbors.

## Part 2: What "contextual" actually means

You do not need the transformer architecture for this assignment. You need one
idea from it, and this is the idea.

The model starts with a static vector for every word -- the same kind of lookup
table spaCy uses. Then, before returning anything, it does this:

> **Every word looks at every other word in the sentence, and rewrites itself
> as a blend of the words it finds relevant.**

That operation is called **attention**. "Relevant" is not hand-coded; the model
learned, from enormous amounts of text, which words should pay attention to
which. When it processes "He swung the bat," the vector sitting at `bat` gets
heavily mixed with `swung`, and the result is no longer the generic `bat`
vector. It is `bat`-as-used-here.

Then it does the whole thing again, five more times. (That is what the `L6` in
`all-MiniLM-L6-v2` means: six layers.) Each pass lets information travel
further through the sentence.

Three consequences worth holding onto, and then we can stop:

1. **Word order matters now.** "Dog bites man" and "man bites dog" contain the
   same words, so each word attends to the same neighbors -- but the model also
   knows *where* each word sits, so `dog`-as-subject and `dog`-as-object come
   out differently. spaCy scores that pair `1.0`; this model scores `0.8812`.
2. **It is still just arithmetic on vectors.** No rules, no parse tree, no
   dictionary. Attention is a weighted average, and the weights are numbers the
   model learned.
3. **The output is one vector per token, not per sentence.** Which leaves one
   last step.

### The step that should surprise you

How does `sentence-transformers` get from 20 token vectors to one sentence
vector? Print the model and it tells you:

```python
print(SentenceTransformer("all-MiniLM-L6-v2"))
```

```
SentenceTransformer(
  (0): Transformer({... 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', ...})
  (2): Normalize(...)
)
```

`pooling_mode: 'mean'`. **It averages the token vectors** -- the identical
operation you wrote by hand in Assignment 4, `sum(vectors) / len(vectors)`.

So the difference between this week and last week is not that one averages and
the other does not. Both average. The difference is **what is being averaged**:
last week, vectors that knew nothing about the sentence they were in; this
week, vectors that have already read it six times over.

There is one more ingredient, and it is why we use `sentence-transformers`
rather than a raw transformer. A plain language model is not trained to make
its averaged output useful for comparison. These models are *additionally*
trained on about a billion sentence pairs with an explicit instruction: make
the vectors of related sentences point the same way. The cosine similarity you
are about to compute means something because somebody trained it to.

## Part 3: Encoding verses

The whole API you need is one method:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = model.encode(
    ["Ye are the light of the world.", "Ye are the salt of the earth."],
    normalize_embeddings=True,
)
print(vectors.shape)   # (2, 384)
```

Two things to get right:

- **Encode the whole list in one call.** Same lesson as `nlp.pipe()` in
  Assignments 2 through 4. All 1,071 verses of Matthew encode in under a second
  as a batch; one call per verse takes closer to forty.
- **`normalize_embeddings=True`** scales every vector to length 1. Free, and it
  makes your `cosine()` from last week do slightly less work.

Your `cosine()` function needs no changes at all. It takes two lists of numbers
and does not care whether there are 300 or 384 of them.

## Part 4: A detour that answers an old question

Remember `Gergesenes` from Assignment 4 -- the word where `has_vector` came back
`False` and spaCy handed you 300 zeros? Ask this model's tokenizer what it does
with the hard words in the King James Bible:

```python
for word in ["light", "savour", "wherewith", "Gergesenes", "thenceforth"]:
    print(word, "->", tokenizer.tokenize(word))
```

```
light       -> ['light']
savour      -> ['sa', '##vo', '##ur']
wherewith   -> ['where', '##with']
Gergesenes  -> ['ge', '##rge', '##sen', '##es']
thenceforth -> ['thence', '##forth']
```

Common words stay whole. Everything else is chopped into **word pieces**, and
`##` means "glued to the previous piece." The vocabulary is only about 30,000
entries, but because any word can be spelled out of pieces, **nothing is ever
out-of-vocabulary.** `Gergesenes` gets a real vector built from four fragments
instead of a row of zeros.

This is also a partial answer to the 1611-English problem that has dogged every
assignment this semester. `wherewith` is not in the training data as a word; it
is handled as `where` + `with`, which is not bad. It is still a modern-English
model reading early modern English, and you should expect that to cost you
something -- but it no longer silently fails.

## Part 5: The assignment

**Redo Assignment 4 with a contextual model, and compare the two rankings.**

Start from
[`05_sentence_transformer_embeddings.py`](05_sentence_transformer_embeddings.py).
It loads Matthew, and `rank()` and `report()` are written for you. Three things
are left:

| Function | Where it comes from |
| --- | --- |
| `cosine(a, b)` | copy from your Assignment 4 script |
| `spacy_vectors(texts)` | copy from your Assignment 4 script |
| `transformer_vectors(texts)` | Part 3 above |

```bash
uv run assignments/05_sentence_transformer_embeddings.py
```

It prints both top-10 lists with their spreads, so you can read them side by
side.

### Calibration

So you can tell a working script from a broken one, the transformer's top match
for Matthew 5:14 is **Matthew 4:16 at 0.4123** -- the same verse spaCy picked,
about people who sat in darkness seeing a great light.

After that the two lists go their separate ways, and that is the assignment.

### The scores are not comparable across models

Look at these two rows before you conclude anything:

| | best match | median verse | best ÷ median |
| --- | --- | --- | --- |
| spaCy, averaged word vectors | 0.8322 | 0.6715 | **1.24** |
| transformer | 0.4123 | 0.1119 | **3.68** |

The transformer's best score is *half* spaCy's. It is not half as confident.
Different models spread their cosine scores across different ranges, and
comparing 0.41 to 0.83 across models is meaningless.

What you can compare is how far the winner sits above the typical verse. By
that measure spaCy's best match is 24% better than its median and the
transformer's is nearly four times its median. The transformer is *separating*
verses; spaCy is mostly reporting that all scripture resembles all scripture.

**Report the ratio, not the raw score.** This is the single most common way to
misread an embedding result, and it is worth one sentence in your write-up.

### What to look at

Your `report()` already prints where Matthew 5:13 landed. Last week it was
198th. Note what it is now and whether you think that settles the argument.

Then look at the disagreements, which are large: **only 2 of the 10 verses
appear in both top-10 lists.** Here are three to get you started, one from each
direction:

| Verse | spaCy rank | transformer rank |
| --- | --- | --- |
| "they be blind leaders of the blind" (15:14) | 613 | **7** |
| "the foolish took their lamps, and took no oil" (25:3) | 490 | **6** |
| "thou art Peter, and upon this rock I will build" (16:18) | **11** | 648 |

The first two share no important vocabulary with "Ye are the light of the
world" and are obviously about it anyway. The third shares a great deal of
vocabulary -- building, rock, city, set -- and is about nothing of the kind.

## Part 6: What to bring to class

1. Your finished `05_sentence_transformer_embeddings.py`.
2. **Both top-10 lists**, with scores and the best/median spread for each, and
   the number of verses the two lists share.
3. **Two disagreements you found yourself** -- not the three in the table above.
   One where the transformer is right, one where spaCy is. For each, say what
   the losing model was looking at instead.
4. Matthew 5:13's rank under both models, and whether you now think the
   transformer "understands" the parallel between 5:13 and 5:14 or has just
   gotten closer by accident. Defend it either way.
5. One sentence on the best ÷ median ratio: what would you conclude if a model
   gave you a best match of 0.95 and a median of 0.94?

And a short note (4-6 sentences):

Across Assignments 2 through 5 you have described a verse four ways: features
you named by hand, averaged static word vectors, and now contextual embeddings.
Each one understands the text better than the last and explains itself less.
For a research question you actually care about, where would you stop on that
scale, and what would you need to be true before you would publish a finding
that rests on 384 numbers nobody can name?

## Reference

- [SentenceTransformers documentation](https://www.sbert.net/) -- start with
  "Quickstart"; you need the first page and nothing else for this assignment.
- [Pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
  -- the size/speed/quality table. `all-MiniLM-L6-v2` is the fast default;
  `all-mpnet-base-v2` is the better one.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  -- Jay Alammar. The pictures are the point. Read as far as "Self-Attention in
  Detail" and stop when the matrices appear; that is already more than this
  assignment needs.
- [BERT 101](https://huggingface.co/blog/bert-101) -- gentler, and explains
  word pieces well.
- [Reimers & Gurevych 2019, "Sentence-BERT"](https://arxiv.org/abs/1908.10084)
  -- the paper behind the library. Read the introduction for why averaging raw
  BERT output is not good enough on its own.
