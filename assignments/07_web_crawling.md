# Web Crawling Lab

## 0. Allowed Imports

Use only the standard library (`pathlib`, `time`, and `urllib.parse` or `re`) plus the
packages installed for this lab (`requests`, `justhtml`).

- `requests` downloads the pages.
- `justhtml` parses the HTML.
- `urllib.parse.urlparse` or `re` for splitting a URL into its parts.

## 1. Create a `scrape` Directory

Use this code at the top of your module to create a `scrape/` directory if it
doesn't already exist:

```python
from pathlib import Path

Path("scrape").mkdir(exist_ok=True)
```

## 2. Write `get_reynoldsnlp(filename)`

Create a function named `get_reynoldsnlp` that takes one argument, `filename`.

- Use `requests` to download `filename` from `https://reynoldsnlp.com/scrape/`.
- Save the downloaded HTML file into the local `scrape/` folder.
- Pause for `1` second after each request, inside this function (use
  `time.sleep()`).
- The function should not return a value.

Example:

- Input: `'aa.html'`
- Request: `https://reynoldsnlp.com/scrape/aa.html`
- Save to: `scrape/aa.html`

Save `response.text`, and write the file with `encoding="utf-8"`.

## 3. Write `parse_hrefs(filename)`

Create a function named `parse_hrefs` that takes one argument, `filename`.

- Open the local file from `scrape/`.
- Parse the HTML using `justhtml`.
- Extract the `href` value of every `<a>` tag.
- Keep only links to pages in `https://reynoldsnlp.com/scrape/`.
- Return a **set** of filenames only (for example, `ab.html`, not the full
  URL).

Real web pages link to many other places, so a crawler always needs a filter
like this. Without one, it would wander off and try to crawl the whole web.

Here is how the pieces you need work, on a small HTML string:

```python
from pathlib import Path
from urllib.parse import urlparse

from justhtml import JustHTML

html_text = """
<p>
  <a href="https://reynoldsnlp.com/scrape/ab.html">a page to crawl</a>
  <a href="https://example.com/other.html">a page on another site</a>
  <a name="top">a link with no href</a>
</p>
"""
doc = JustHTML(html_text, sanitize=False)

for link in doc.query("a[href]"):  # CSS selector: <a> tags with an href
    href = link.attrs["href"]
    # the next two lines demonstrate how to use urlparse, but you can use
    # any method you want to inspect the url (e.g. re or in)
    url = urlparse(href)
    print(url.netloc, url.path, Path(url.path).name)
```

Output:

```
reynoldsnlp.com /scrape/ab.html ab.html
example.com /other.html other.html
```

Notes:

- `sanitize=False` tells `justhtml` to keep the HTML exactly as written. By
  default it removes anything that would be unsafe to *display* in a browser,
  but we only want to *read* the page.
- `link.attrs` is a dictionary of the tag's attributes.
- `urlparse()` splits a URL into parts: `netloc` is the site
  (`reynoldsnlp.com`) and `path` is everything after it (`/scrape/ab.html`).
- `Path(...).name` keeps only the last part of a path.

Your job is to turn this into a function that reads a file from `scrape/`,
uses `netloc` and `path` to skip links that aren't in
`https://reynoldsnlp.com/scrape/`, and returns the filenames as a set.

Example:

- If `scrape/aa.html` links to:
  - `https://reynoldsnlp.com/scrape/ab.html`
  - `https://reynoldsnlp.com/scrape/ac.html`
  - `https://example.com/other.html`
- Then `parse_hrefs("aa.html")` should return:
  - `{'ab.html', 'ac.html'}`

## 4. Crawl With Sets

Use the two functions above to build a crawler.

- Create a `todo` set with one seed filename: `{'aa.html'}`.
- Create an empty `visited` set for already-requested filenames.
- Loop until the `todo` set is empty:
  - Pop one filename from the `todo` set.
  - If it is already in `visited`, skip it.
  - Request and save the page.
  - Add that filename to `visited`.
  - Parse the page to get the filenames it links to.
  - Add only unvisited filenames to the `todo` set.

For the last step, use set difference (`-`), which keeps the items in the left
set that are *not* in the right set:

```python
todo.update(new_filenames - visited)
```

This only works if `new_filenames` is a set. That is why `parse_hrefs` returns
one.

## 5. Expected Result

When complete, you should have **80 files** in the `scrape/` directory. With a
1-second pause per request, the crawl takes about a minute and a half.

- The site contains many duplicate links.
- Your script must request each filename at most once.
- Your crawler should not enter an infinite loop.

## 6. Style and Ethics Requirements

- Follow ethical scraping practices: pause `1` or `2` seconds between
  requests, and never request the same page twice. Import `time` and use `time.sleep(1)`.

## 7. Submission Checklist

- `scrape/` is created automatically.
- `get_reynoldsnlp(filename)` uses `requests` to download pages, saves them
  locally, and returns `None`.
- `parse_hrefs(filename)` uses `justhtml` and returns a set of filenames only.
- The crawler uses sets for `todo` and `visited`.
- The crawler does not request any page more than once.
- Exactly `80` files are saved in `scrape/`.
- Final reflection comments are included.

## 8. Add Two Final Comments to Your Module

On the last lines of your module, add comments answering:

1. What filename pattern do you see in the saved files, and which filename is
   missing from the pattern?
2. Why are sets better than lists for tracking `todo` and `visited`?
