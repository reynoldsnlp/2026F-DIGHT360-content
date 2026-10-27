# Assignment 11: Final Project Proposals

**Goal:** brainstorm at least three possible final projects and describe each
one clearly enough that you (and I) can tell whether it is doable before you
commit to it.

## The Task

Write **at least three** proposals, **one paragraph each**. Each paragraph
should answer four questions:

1. **Data.** What language data will you use, and how will you get it? Name
   the actual source (a website, a corpus, a Hugging Face dataset, a dataset
   you already have) and the method (downloading a file, crawling pages, an
   API). Roughly how much text
   is it?
2. **Processing.** What will you do to the data? Name the tools and steps:
   cleaning, tokenizing, spaCy or Stanza features, embeddings, similarity,
   counting, classifying, and so on.
3. **Deliverable.** What will the result look like? A script that prints a
   table? A chart? A search tool? A JSON dataset someone else could use? Be
   specific enough that someone could picture it.
4. **Evaluation.** How will you know it worked? You do not need a full plan,
   but you need at least a general idea: a comparison against something
   already known, a set of test cases you check by hand, a baseline to beat,
   or a prediction you can test.

The three proposals should be genuinely different from each other, not three
versions of the same idea. It is fine if one of them is ambitious and another
is safe.

## Things to Keep in Mind

- **Scope.** You have the rest of the semester, not the rest of your life.
  A small question answered well is better than a big question answered
  badly.
- **Get the data first.** The most common way a project fails is that the data
  turns out to be unavailable, too small, or much messier than expected. Before
  you write a proposal, spend ten minutes checking that the data really exists
  and that you can get to it.
- **Scrape ethically.** If you plan to crawl a website, check its `robots.txt`
  and terms of use, and follow the
  [guidelines from Assignment 8](08_web_crawl_recipes.md#0-scraping-ethically).
- **Build on what you know.** Every assignment this semester is a possible
  starting point: regular expressions (1), spaCy features (2-3), word and
  sentence embeddings (4-5), semantic search (6), web crawling (7-9), and
  Stanza for other languages (10).
- **Look on Hugging Face.** The [Hugging Face Hub](https://huggingface.co/)
  hosts thousands of ready-made [datasets](https://huggingface.co/datasets)
  and pretrained [models](https://huggingface.co/models). A dataset there can
  save you from scraping anything, and a model there can do jobs we have not
  covered, such as sentiment analysis, named entity recognition, translation,
  summarization, or classifying text into your own categories. Read each
  dataset's or model's card to see where it came from, what it is licensed
  for, and what its known limitations are.
- **Languages other than English.** If your project involves a language other than English, check whether there are existing datasets or models for that language on spacy, stanza, or Hugging Face.


## Two Example Proposals

These show the level of detail I am looking for. Do not use them as your own
proposals.

> **Who wrote the disputed *Federalist Papers*?** Twelve of the 85
> *Federalist Papers* were claimed by both Alexander Hamilton and James
> Madison. I will download the full text of all 85 papers from Project
> Gutenberg as a single plain-text file and split it into individual papers
> with a regular expression, using the header of each paper to record its
> author (Hamilton, Madison, Jay, joint, or disputed). For each paper I will
> use spaCy to compute the relative frequencies of about 50 common function
> words (*upon*, *whilst*, *by*, *to*, and so on), plus a few features from
> Assignment 2 such as mean sentence length. I will then train a simple
> classifier on the papers whose author is known and use it to predict the
> author of each disputed paper. The deliverable is a script that prints a
> table of the twelve disputed papers with the predicted author and how
> confident the classifier is, along with a chart showing which features
> separate the two authors most clearly. To evaluate it, I will first test
> the classifier on known-author papers it did not see during training, to
> make sure it can tell Hamilton from Madison at all. Then I will compare my
> predictions to Mosteller and Wallace's famous 1964 study, which concluded
> that Madison wrote all twelve.

> **A meaning-based search engine for BYU devotionals.** BYU Speeches
> (speeches.byu.edu) has transcripts of hundreds of devotional and forum
> addresses. I will crawl the transcripts from the last ten years (about 300
> talks), caching every page and pausing between requests, and save the
> speaker, title, date, and text of each talk to a JSON file, the way we did
> for General Conference in Assignment 9. I will split each talk into
> paragraphs and embed each paragraph with a sentence transformer model. The
> deliverable is a command-line tool: you type a question like "how do I deal
> with failure?" and it prints the five most similar paragraphs, with the
> speaker, title, and a link to each talk. To evaluate it, I will write 20 test
> questions, each based on a specific talk I have read, and count how often
> that talk appears in the top five results. I will compare that score to a
> plain keyword search that counts shared words, to find out whether the
> embeddings actually do better.

Notice that both examples name a real data source, describe the processing in
concrete steps, describe exactly what the finished product does, and say what
the result will be compared against.

## What to Bring to Class

1. Your three (or more) proposals, one paragraph each. Be prepared to briefly explain each one in class.
2. Which one you are leaning toward, and the biggest risk you see in it.
