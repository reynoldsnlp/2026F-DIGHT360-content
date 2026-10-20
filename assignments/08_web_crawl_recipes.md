# Assignment 8: Scraping Recipes

**Goal:** point last week's crawler at a real website and come back with
structured data -- not just saved pages, but the specific pieces of each page
you want, in a file you can analyze later.

## 0. Scraping Ethically

Being able to download something does not mean you are allowed to do anything
you want with it. Before you scrape any site, check its `robots.txt` (for
example, <https://preppykitchen.com/robots.txt>) and its terms of use.

**Do:**

- Pause between requests (at least 1 second), and never request the same page
  twice. Save pages to disk and parse your saved copies.
- Scrape only the pages you need.
- Keep the URL of everything you collect, so you can cite it.
- Use what you collect for research, teaching, and analysis: counting,
  measuring, comparing, and quoting short passages with attribution. In the
  U.S. this kind of use is usually *fair use*.

**Don't:**

- Republish what you scraped: no reposting full articles, recipes, or photos,
  and no putting your dataset on GitHub or anywhere public.
- Use scraped content in a way that competes with the original, such as a
  recipe site built from someone else's recipes.
- Scrape anything behind a login, or collect personal information about
  people.
- Ignore `robots.txt` or try to get around a site that is blocking you.

Recipes are a useful example. A list of ingredients is a set of facts and is
generally not protected by copyright, but the writing around it -- the stories,
tips, and instructions -- and the photos are. Analyzing all of it is fine.
Republishing the writing or the photos is not.

## 1. The Task

Write `08_web_crawl_recipes.py`, which collects every Instant Pot recipe from
Preppy Kitchen, starting at this category page:

<https://preppykitchen.com/category/recipes/cooking-method/instant-pot-cooking-method/>

For each recipe, collect:

| key | what it is |
| --- | --- |
| `url` | the recipe's URL |
| `title` | the text of the `.posttitle` element |
| `text` | the full text of `div.postcontent` |
| `ingredients` | a list of strings, one per ingredient |
| `nutrition` | a dictionary of nutrition facts, e.g. `{"Calories": "746 kcal", "Protein": "37 g"}` |

Save all of the recipes as a list of dictionaries in
`instant_pot_recipes.json`.

Use the same tools as [Assignment 7](07_web_crawling.md): `requests` to
download, `justhtml` to parse, and the standard library (`json`, `pathlib`,
`time`, `urllib.parse`).

## 2. Requirements

- **Save every page you download** into a `recipes/` directory, and pause 1
  second after each request. If a page is already saved, read it from disk
  instead of downloading it again. Your second run should send no requests at
  all.
- **Identify your scraper.** By default, `requests` tells every site it is
  `python-requests/2.x`, and this site blocks that name with a "please enable
  cookies" page. Send a `User-Agent` header that says who you are instead:

  ```python
  HEADERS = {"User-Agent": "DIGHT360 class project"}
  response = requests.get(url, headers=HEADERS, timeout=30)
  ```

  Do **not** pretend to be a web browser. Saying honestly who you are lets the
  site decide whether to allow you.
- **Request each recipe once.** Some recipes are linked more than once on the
  category page.
- **Write functions that each do one job**, for example one to get a page, one
  to find the recipe links, and one to parse a single recipe.

## 3. Finding the Selectors

You are given the selectors for the title and the post text. You have to find
the ones for the ingredients and the nutrition facts yourself.

Open a recipe in your browser, right-click an ingredient, and choose
**Inspect**. Look at the `class` attributes of the element and its parents
until you find a selector that matches every ingredient and nothing else. Do
the same for one nutrition fact. Each fact has its own label, value, and unit,
so you will need to put them together into a dictionary entry yourself.

Test each selector on one saved page before running it on all of them.

## 4. Expected Result

You should find **15 recipes**. The whole crawl should take about 20 seconds
the first time and well under a second after that. One entry in your JSON file
should look roughly like this (with `text` shortened here):

```json
{
  "url": "https://preppykitchen.com/instant-pot-black-beans/",
  "title": "Instant Pot Black Beans",
  "text": "This post may contain affiliate links. ... No-soak Instant Pot black beans are a convenient way to cook dried beans in a flash! ...",
  "ingredients": [
    "1 pound dried black beans (450g)",
    "½ yellow onion diced (130g)",
    "6 cloves garlic minced",
    "..."
  ],
  "nutrition": {
    "Calories": "206 kcal",
    "Carbohydrates": "38 g",
    "Protein": "13 g",
    "...": "..."
  }
}
```

Not every recipe lists the same nutrition facts, so your code should not
assume a fixed set of keys.

## 5. Submission Checklist

- `08_web_crawl_recipes.py` runs from start to finish with no errors.
- Pages are saved in `recipes/`, and a second run sends no requests.
- The script pauses at least 1 second after every request.
- `instant_pot_recipes.json` contains 15 recipes, each with all five keys.
- `ingredients` is a list and `nutrition` is a dictionary in every recipe.
- Do **not** submit or commit `recipes/` or `instant_pot_recipes.json`.

## 6. Add Two Final Comments to Your Module

On the last lines of your module, add comments answering:

1. Look at the `text` you collected. Besides the recipe itself, what else ended
   up in it? Name two kinds of text you would want to remove before analyzing
   it, and say how you might remove them.
2. Name one research question you could answer with this dataset that would be
   fair use, and one use of the same data that would not be.
