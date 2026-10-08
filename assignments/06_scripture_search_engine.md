# Assignment 6: A Semantic Search Engine for Scripture

**Goal:** turn last week's embeddings into a tool someone could actually use --
type a question in plain English, get back the verses that answer it -- then
find out what kinds of questions it cannot answer.

In [Assignment 5](05_sentence_transformer_embeddings.md) you asked one
question: which verse is most like Matthew 5:14? This week the target is not a
verse but whatever the user types, and the search covers the whole New
Testament, not just Matthew. A search for "forgiveness of debts" should find
"forgive us our debts" even though the query and the verse share only one
word. A keyword search cannot do that, and yours will.

This time there is no starter script. You plan the program and you write all of
it. Everything you need has already appeared in Assignments 4 and 5.

## What your script must do

Write `06_scripture_search_engine.py`. When run with

```bash
uv run assignments/06_scripture_search_engine.py
```

it should:

1. **Load every verse of the New Testament**, from the same JSON file you have
   used all semester, keeping each verse's reference with its text.
2. **Embed every verse** with a sentence-transformer model.
   `all-MiniLM-L6-v2` is fine.
3. **Cache the embeddings to disk.** The first run may take a minute or two to
   encode everything. Every run after that should skip the encoding and load
   the saved vectors, starting up in a few seconds. If you change models, your
   program must not load vectors made by the old one.
4. **Prompt the user for a query** using `input()`.
5. **Print the 10 verses most similar to the query**, best first, each with its
   similarity score, its reference, and its text (shortened is fine).
6. **Keep asking** for queries until the user enters an empty line, then exit.

A session should look roughly like this:

```text
Searching 7957 verses. Press Enter on an empty line to quit.

search> forgiveness of debts
0.7053  Matthew 6:12           And forgive us our debts, as we forgive our debtors.
0.5363  Matthew 6:14           For if ye forgive men their trespasses, your heavenly Father will also forgive y
...

search>
```

Your scores should be close to these if you use the same model. The formatting
is up to you.

### Ground rules

- Use only the libraries already in `pyproject.toml` and the Python you have
  learned in this course. You do not need anything new.
- Split the program into functions that each do one job. Loading the verses,
  loading or building the embeddings, and searching should not all live in
  `main()`.
- Load the model **once**, not once per query.
- Before you write any code, write down the steps your program has to go
  through, in order. Bring that plan to class along with the script.
- The user should be able to type "quit" or press Enter on an empty line to
  exit the program. (Most users do not know to type Ctrl+C.)

## Try to break it

Once it works, spend time *using* it. Try at least ten queries of different
kinds, for example:

- a topic ("the resurrection of the dead")
- a phrase you half remember ("something about a camel and a needle")
- a modern-English paraphrase of a King James verse
- a single word, and then a long sentence
- a query with negation ("people who did **not** believe")
- a question ("who betrayed Jesus?")

Keep track of which ones it handles well and which it gets wrong. Be on the
lookout for results that are about the right *topic* but say the opposite of
what you asked for.

## What to bring to class

1. Your plan: the list of steps you wrote before coding.
2. Your finished `06_scripture_search_engine.py`.
3. **Three queries it gets right**, with the top 3 results for each, and one
   sentence on why a keyword search would have done worse.
4. **Three queries it gets wrong**, with the top 3 results for each. For each
   one, say what the model seems to be matching *instead* of what you meant.
5. One sentence on caching: how long does your script take to start up the
   first time, and how long after that?

## Reference

- [SentenceTransformers: Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Assignment 5](05_sentence_transformer_embeddings.md) -- encoding, cosine
  similarity, and ranking are all there.
