# Assignment 13: One Word, Many Vectors

**Goal:** understand a script that pulls BERT's vector for one word out of
many sentences and plots whether those vectors group by meaning -- and what
else they group by.

```bash
uv sync
uv run assignments/13_contextual_word_embeddings.py
```

Read [`13_contextual_word_embeddings.py`](13_contextual_word_embeddings.py)
and open the PNGs it saves. Then work out what every line does:

- **Use the debugger.** Set a breakpoint in `word_vector()` and inspect
  `encoded`, `offsets`, `positions`, and the `.shape` of each array.
- **Change values and rerun.** Try other values of `LAYER`, edit the
  sentences, add your own word to `EXAMPLES`.
- **Print things.** `tokenizer.tokenize(text)`, `vector[:5]`,
  `np.linalg.norm(vector)`.
- **Break things on purpose.** Predict what will change, then run it.

Bring to class one thing the plots showed you that surprised you, and the
experiment that showed it.
