"""SOLUTION KEY -- Assignment 8. Instructor copy, not distributed (gitignored).

    uv run assignments/08_web_crawl_recipes_solution.py

Download the Preppy Kitchen Instant Pot category page, follow every recipe link
on it exactly once, and save the title, post text, ingredients, and nutrition
facts of each recipe to instant_pot_recipes.json. Every page is saved to
recipes/ the first time it is downloaded and read from disk after that, so
rerunning the script sends no requests. The first run takes about 20 seconds
because of the pause between requests.
"""

import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from justhtml import JustHTML

START_URL = (
    "https://preppykitchen.com/category/recipes/cooking-method/instant-pot-cooking-method/"
)
HTML_DIR = Path("recipes")
OUTPUT_PATH = Path("instant_pot_recipes.json")
HEADERS = {"User-Agent": "DIGHT360 class project"}

HTML_DIR.mkdir(exist_ok=True)


def get_page(url):
    """Return the HTML of url, downloading it only if it is not saved yet.

    The saved file is named after the last part of the URL's path, so
    https://preppykitchen.com/instant-pot-ribs/ becomes recipes/instant-pot-ribs.html.
    """
    path = HTML_DIR / (Path(urlparse(url).path).name + ".html")
    if not path.exists():
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
        time.sleep(1)
    return path.read_text(encoding="utf-8")


def parse_recipe_links(html_text):
    """Return the set of recipe URLs linked from the category page.

    Each recipe is an <article class="griditem">. Some recipes appear twice on
    the page, which the set takes care of.
    """
    doc = JustHTML(html_text, sanitize=False)
    return {link.attrs["href"] for link in doc.query("article.griditem a[href]")}


def parse_nutrition(doc):
    """Return the nutrition facts as a dictionary, e.g. {'Calories': '746 kcal'}."""
    nutrition = {}
    for fact in doc.query(".wprm-nutrition-label-text-nutrition-container"):
        label = fact.query_one(".wprm-nutrition-label-text-nutrition-label")
        value = fact.query_one(".wprm-nutrition-label-text-nutrition-value")
        unit = fact.query_one(".wprm-nutrition-label-text-nutrition-unit")
        name = label.to_text().rstrip(":")
        nutrition[name] = value.to_text() + (" " + unit.to_text() if unit else "")
    return nutrition


def parse_recipe(url, html_text):
    """Return a dictionary of everything we keep from one recipe page."""
    doc = JustHTML(html_text, sanitize=False)
    return {
        "url": url,
        "title": doc.query_one(".posttitle").to_text(),
        "text": doc.query_one("div.postcontent").to_text(
            separator="\n", separator_blocks_only=True
        ),
        "ingredients": [li.to_text() for li in doc.query("li.wprm-recipe-ingredient")],
        "nutrition": parse_nutrition(doc),
    }


def main():
    recipe_urls = parse_recipe_links(get_page(START_URL))
    print(f"Found {len(recipe_urls)} recipes.")

    recipes = []
    for url in sorted(recipe_urls):
        recipe = parse_recipe(url, get_page(url))
        print(f"{recipe['title']:<40} {len(recipe['ingredients']):>2} ingredients, "
              f"{len(recipe['nutrition']):>2} nutrition facts")
        recipes.append(recipe)

    OUTPUT_PATH.write_text(json.dumps(recipes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(recipes)} recipes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
