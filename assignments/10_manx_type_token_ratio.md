# Assignment 10: Type-Token Ratios in Manx

**Goal:** scrape a small corpus in a language most NLP tools ignore, analyze it
with a model built for that language, and find out what a simple measure of
vocabulary richness can and cannot tell you.

## spaCy and Stanza

You have used spaCy all semester. [Stanza](https://stanfordnlp.github.io/stanza/),
from Stanford, does the same basic jobs: tokenizing, lemmatizing, tagging parts
of speech, and parsing. The difference is which languages each one covers.

| | spaCy | Stanza |
| --- | --- | --- |
| Languages with trained models | about 25, mostly widely spoken | more than 70, including many small ones |
| Trained on | each language's own data, often large | [Universal Dependencies](https://universaldependencies.org/) treebanks, which can be very small |
| Speed | fast | slower |
| Part-of-speech tags | UPOS (`token.pos_`) | the same UPOS tags (`word.upos`) |

Use spaCy when your language has a good spaCy model. Use Stanza when it does
not. Keep in mind that a model trained on a tiny treebank makes more mistakes.

## Manx

Manx (*Gaelg*) is the Celtic language of the Isle of Man. Its last native
speaker died in 1974, and it is now spoken again by a small community of
learners and their children. Stanza's Manx model is trained on the
[Cadhan Aonair treebank](https://universaldependencies.org/treebanks/gv_cadhan/index.html).

Like Irish and Welsh, Manx changes the **beginning** of words depending on the
words around them. The name of the island is *Mannin*, but "in the Isle of Man"
is *ayns Ellan **V**annin*. To a computer, *Mannin* and *Vannin* are two
different words.

## The Pages

Scrape these ten articles from the Manx Wikipedia:

1. <https://gv.wikipedia.org/wiki/Mannin>
2. <https://gv.wikipedia.org/wiki/Doolish>
3. <https://gv.wikipedia.org/wiki/Gaelg>
4. <https://gv.wikipedia.org/wiki/Oie_Houney>
5. <https://gv.wikipedia.org/wiki/Yn_Chruinnaght>
6. <https://gv.wikipedia.org/wiki/Edward_Faragher>
7. <https://gv.wikipedia.org/wiki/Sniaul>
8. <https://gv.wikipedia.org/wiki/Ned_Maddrell>
9. <https://gv.wikipedia.org/wiki/Brian_Stowell>
10. <https://gv.wikipedia.org/wiki/Tinvaal>

Cache the pages in a `manx/` directory and follow the
[ethical guidelines from Assignment 8](08_web_crawl_recipes.md#0-scraping-ethically).

## The Task

Write `10_manx_type_token_ratio.py`. For each article it should:

1. **Extract the main text**: the paragraphs of the article itself. Use your
   browser's Inspect tool to find where they are. Not every `<p>` on the page
   is part of the article. Infoboxes and tables can contain paragraphs too, so
   compare your extracted text with what you see in the browser.
2. **Run the text through Stanza's Manx pipeline** to get each word's form,
   lemma, and part of speech.
3. **Compute two type-token ratios (TTR).** TTR is the number of *distinct*
   words (types) divided by the total number of words (tokens). Leave out
   punctuation, symbols, and numbers, and lowercase everything first.
   - **Form TTR** counts each word as it is written.
   - **Lemma TTR** counts each word's lemma instead.
4. **Print a table** with each article's title, number of tokens, form TTR,
   and lemma TTR.

As a check: *Mannin*, the longest article, should come out at about 3,000
tokens.

### Getting Started with Stanza

`stanza` is already in `pyproject.toml`. The first time you run this, it
downloads the Manx model:

```python
import stanza

stanza.download("gv")
nlp = stanza.Pipeline("gv", processors="tokenize,pos,lemma")

doc = nlp("Ta Gaelg loayrit ayns Ellan Vannin.")
for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.lemma, word.upos)
```

```text
Ta bee VERB
Gaelg Gaelg PROPN
loayrit loayrit ADJ
ayns ayns ADP
Ellan ellan NOUN
Vannin Mannin PROPN
. . PUNCT
```

Notice that the lemma of *Vannin* is *Mannin*. Read
[Getting Started](https://stanfordnlp.github.io/stanza/getting_started.html)
for how pipelines work, and the
[Word section of Data Objects](https://stanfordnlp.github.io/stanza/data_objects.html#word)
for everything you can get from a word. The
[UPOS tag list](https://universaldependencies.org/u/pos/) explains the tags.

## What to Bring to Class

1. Your `10_manx_type_token_ratio.py` and the table it prints.
2. **Length.** Sort the articles by number of tokens. What happens to TTR as
   the articles get longer, and why? What does that mean for anyone who
   compares the TTRs of two texts to decide which one has the "richer"
   vocabulary?
3. **Errors.** Find Manx in [Stanza's model performance report](https://stanfordnlp.github.io/stanza/performance.html). Without doing any mathematical computation, how well do you think you can trust these TTR results for Manx?
4. **Future interest.** Which Stanza models would you be interested in trying next, and why?
