# Assignment 12: Hugging Face Transformers

**Goal:** learn to use pretrained models for NLP tasks through the Hugging Face
`pipeline()`. This is a weekend reading assignment (about 2 hours). You do not
need to learn neural-network math or fine-tuning yet.

## Before you start: running the course's code

The course pages have "Open in Colab" buttons. Ignore them. Everything you need
is already installed in your course environment (`transformers` and `torch`
came in with Assignment 5), so run code the way you always do:

```bash
uv run assignments/12_HF_pipelines.py
```

The course was written for an older version of `transformers`, and a few of its
examples no longer work:

- **`summarization`, `translation`, and `question-answering` pipelines have
  been removed.** If you try them, you will get an error. Skip those examples.
- **`grouped_entities=True` has been removed.** Use
  `aggregation_strategy="first"` instead.
- **Always name a model.** Without one, `pipeline()` picks a default, and some
  defaults are big (1-2 GB). The starter script names small ones.

When a model loads, you may see a table of `UNEXPECTED` weights. It is
harmless.

## 1. Try pretrained pipelines (30 minutes)

Read [Transformers, what can they do?](https://huggingface.co/learn/llm-course/chapter1/3).

Focus on the text examples: classification, named entity recognition, and
choosing a model from the Hub. Try at least two examples with your own text.

Skip the image and audio examples, and skip summarization, translation, and
question answering (see above).

## 2. See what the pipeline does (35 minutes)

Read [Chapter 2: Introduction](https://huggingface.co/learn/llm-course/chapter2/1)
and [Behind the pipeline](https://huggingface.co/learn/llm-course/chapter2/2).

Focus on the roles of the pipeline, tokenizer, and model. Don't worry about
memorizing PyTorch details.

## 3. Look more closely at tokenization (20 minutes)

Read [Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4). Connect
the token IDs and subword pieces to what you already know about tokenization.

## 4. Experiment (30 minutes)

Start from [`12_HF_pipelines.py`](12_HF_pipelines.py). It runs a
text-classification pipeline and an NER pipeline on two sentences. Replace them
with text you choose, then try selecting a different model from the
[Hub](https://huggingface.co/models). Note one result that surprised you.

## 5. Wrap up (5 minutes)

In a few sentences, explain what each of these does: **pipeline**,
**tokenizer**, and **model**.

**Stop at two hours.** Save the remaining Chapter 2 lessons for another time.
