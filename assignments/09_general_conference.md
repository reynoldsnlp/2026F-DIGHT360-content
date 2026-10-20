# Assignment 9: Crawling General Conference

**Goal:** build a corpus of every talk from the October 2026 General
Conference, starting from the conference's table of contents:

<https://www.churchofjesuschrist.org/study/general-conference/2026/10?lang=eng>

This time you plan the whole crawler yourself.

## Requirements

Write `09_general_conference.py`. It must:

- **Cache the HTML.** Save every page you download to a `conference/`
  directory. If a page is already saved, read it from disk instead of
  downloading it again. A second run should send no requests at all.
- **Follow the ethical guidelines from [Assignment 8](08_web_crawl_recipes.md#0-scraping-ethically).**
  Check `robots.txt`, send an honest `User-Agent`, pause at least 1 second
  after every request, and request each page at most once.
- **Save `general_conference_2026_10.json`**: a list with one dictionary per
  talk, in the order the talks were given, in this format:

```json
[
  {
    "url": "https://www.churchofjesuschrist.org/study/general-conference/2026/10/12gong?lang=eng",
    "speaker": "Elder Gerrit W. Gong",
    "title": "Come to the Lord in His Holy House",
    "session": "Saturday Morning Session",
    "text": "Welcome to conference!\n\nWhy did the temple workers cross the road? ..."
  }
]
```

`text` is the body of the talk only, with paragraphs separated by a blank line
(`"\n\n"`). Do not include the title, speaker, summary, or footnotes.

## Watch Out

Real websites are messier than the practice site. Look closely at your output
before you submit, because each of these problems can happen here without
causing an error:

- **Mojibake.** Compare the names and punctuation in your JSON with the same
  talk in your browser. If they don't match, the problem started when your
  script turned the downloaded bytes into a string. Read the
  [Response Content](https://requests.readthedocs.io/en/latest/user/quickstart/#response-content)
  section of the `requests` documentation to see how `requests` decides how to
  do that. Fix it before your pages are cached, or your saved copies will be
  wrong too.
- **Duplicates.** A page can link to the same place more than once. Count your
  talks and make sure each one appears exactly once.
- **Inconsistency.** Not every page is built exactly like the first one you
  look at. Check that every field looks right for *every* talk, not just one or
  two.

Do **not** submit or commit `conference/` or the JSON file.
