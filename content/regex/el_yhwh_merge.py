# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.32",
#     "rich>=13.7",
# ]
# ///
"""Map the El/Yahweh seam in Genesis with regular expressions.

THE QUESTION
    Genesis calls God by two different names. Can we *see* the boundaries where
    two independent ancient documents were stitched together, just by counting
    which name appears where?

THE SCHOLARLY ANGLE
    The Documentary Hypothesis holds that Genesis was compiled from older,
    separate sources. One (the Elohist / Priestly strand) calls God 'Elohim';
    another (the Jahwist) calls God by the personal name 'Yahweh'. The KJV
    encodes this distinction typographically and the encoding survives in
    plain text:

        Elohim          -> "God"
        Yahweh (YHWH)   -> "LORD"      (all caps: small-caps in print)
        Adonai          -> "Lord"      (title case)
        Adonai YHWH     -> "Lord GOD"
        YHWH Elohim     -> "LORD God"  (the compound; Genesis 2-3 is full of it)

    So a case-*sensitive* regex over the KJV is a crude but real source-critical
    instrument. Run it and Genesis 1:1-2:3 lights up as pure Elohim, then
    Genesis 2:4 flips to "the LORD God" mid-sentence. That flip is the seam.

RUNNING IT
    uv run --script el_yhwh_merge.py

DEBUGGING IT IN VS CODE
    First create the environment uv would use, and point VS Code at it:

        uv sync --script el_yhwh_merge.py
        uv python find --script el_yhwh_merge.py   # paste into "Python: Select Interpreter"

    Then press F5 ("Python Debugger: Debug Python File"). A suggested itinerary,
    each stop marked in the code below with a BREAKPOINT comment:

    1. tag_text()  -- the heart of it. Set a breakpoint on the `for match in`
       line and step through Genesis 2:4 one match at a time. Watch `match` in
       the VARIABLES pane: expand it and read .lastgroup, .group(), .span().
       This is how you learn what a match object actually *is*.

    2. tag_text()  -- put `DIVINE_NAME_RE.pattern` and `match.lastgroup` in the
       WATCH pane. Then ask the DEBUG CONSOLE questions while paused, e.g.
           DIVINE_NAME_RE.findall("the LORD God said")
           re.findall(r"\\bGod\\b", "the gods of Egypt")
       The debug console evaluates in the paused frame. Use it as a lab bench.

    3. tally_chapter() -- a CONDITIONAL breakpoint (right-click in the gutter,
       "Add Conditional Breakpoint"). Put it on the function's first statement,
       the `tally = ChapterTally(...)` line, with the condition

           chapter["chapter"] == 2

       and you skip chapter 1 to land on the seam chapter.

    4. tag_text() -- the same trick, but aimed at the research question instead
       of at a chapter number. Breakpoint on the `label = match.lastgroup` line,
       with the condition

           match.lastgroup == "compound"

       Press F5 once and you stop on the first "LORD God" in Genesis, which is
       2:4 -- the seam itself. Look at `text` in the VARIABLES pane. You did not
       have to know in advance where the seam was; you described what you were
       looking for and let the debugger find it.

    5. count_switches() -- step INTO it (F11), then step OUT (Shift+F11). The
       way you arrive here is the lesson: nothing calls it directly. The
       `switches` property calls it, verdict() reads that property, and
       render_book() calls verdict(). The CALL STACK pane is how you discover
       that, and it is why an innocuous-looking property access is doing real
       work every time it is read.

"""

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import re

import requests
from rich.console import Console
from rich.table import Table

SOURCE_URL = (
    "https://raw.githubusercontent.com/bcbooks/scriptures-json/master/old-testament.json"
)
CACHE_PATH = Path(__file__).parent / ".cache" / "old-testament.json"

# The instrument. Order of alternatives is load-bearing: Python's regex engine
# takes the FIRST alternative that matches at a position, so the two-word names
# must be tried before the one-word names, or "LORD God" gets torn into two
# separate hits. Named groups (?P<name>...) let us ask the match object which
# alternative won, via match.lastgroup.
DIVINE_NAME_RE = re.compile(
    r"""
      (?P<compound>      \b LORD \s+ God \b )   # YHWH Elohim -- the seam marker
    | (?P<adonai_yahweh> \b Lord \s+ GOD \b )   # Adonai YHWH
    | (?P<yahweh>        \b LORD \b )           # YHWH
    | (?P<elohim>        \b God \b )            # Elohim
    | (?P<adonai>        \b Lord \b )           # Adonai (a title, not a name)
    """,
    re.VERBOSE,
)

# An interpretive decision, spelled out so you can disagree with it and re-run:
# an occurrence of "LORD God" is counted as an attestation of Yahweh (it is the
# divine name, with Elohim appositional), while also being tallied separately as
# a seam marker. Change this set and the whole picture shifts -- which is the
# point. Your regex is an argument, not a measurement.
YAHWEH_GROUPS = frozenset({"yahweh", "compound"})
ELOHIM_GROUPS = frozenset({"elohim"})


@dataclass
class ChapterTally:
    """What one chapter looks like once the regex has had its way with it."""

    chapter: int
    counts: Counter = field(default_factory=Counter)
    verse_labels: list[str] = field(default_factory=list)

    @property
    def elohim(self) -> int:
        return sum(self.counts[group] for group in ELOHIM_GROUPS)

    @property
    def yahweh(self) -> int:
        return sum(self.counts[group] for group in YAHWEH_GROUPS)

    @property
    def compound(self) -> int:
        return self.counts["compound"]

    @property
    def switches(self) -> int:
        return count_switches(self.verse_labels)


def load_old_testament() -> dict:
    """Load the KJV Old Testament JSON from cache when available.

    The cache is kept beside this script so repeated lab runs do not keep
    re-downloading the same 7 MB data file.
    """
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(response.text, encoding="utf-8")
    return response.json()


def find_book(bible: dict, name: str) -> dict:
    """Return one book dict, matched case-insensitively on its title."""
    wanted = name.casefold()
    for book in bible["books"]:
        if book["book"].casefold() == wanted:
            return book
    titles = ", ".join(book["book"] for book in bible["books"])
    raise SystemExit(f"No book named {name!r}. Available books: {titles}")


def tag_text(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str, int, int]]:
    """Find every divine name in `text` and label it.

    Returns (label, matched_text, start, end) for each hit, in reading order.

    BREAKPOINT 1: on the `for match in ...` line. Step (F10) through
    Genesis 2:4 and watch each `match` arrive. `match.lastgroup` is the name of
    the alternative that won -- that single attribute is what turns a pile of
    regex hits into labelled data.

    BREAKPOINT 2: same line. Put `DIVINE_NAME_RE.pattern` and `match.lastgroup`
    in the WATCH pane, then interrogate the paused frame from the DEBUG CONSOLE:
    the console runs code in *this* scope, so `pattern.findall(text)` and
    `text[match.start() - 20:match.end()]` both work. Use it as a lab bench.

    BREAKPOINT 4: a CONDITIONAL breakpoint on the `label = ...` line, condition
    `match.lastgroup == "compound"`, stops on the first "LORD God" in the book
    -- Genesis 2:4. Describe what you are hunting for and the debugger finds it.
    """
    tags: list[tuple[str, str, int, int]] = []
    for match in pattern.finditer(text):
        label = match.lastgroup  # <- BREAKPOINT 4 goes on this line
        assert label is not None, "every alternative in the pattern is named"
        tags.append((label, match.group(), match.start(), match.end()))
    return tags


def label_verse(tags: list[tuple[str, str, int, int]]) -> str:
    """Reduce one verse to a single source signal: 'E', 'J', or '.' for neither.

    Written as an explicit loop rather than a comprehension so that there is
    something to look at in the VARIABLES pane while you step through it.
    """
    elohim = 0
    yahweh = 0
    for label, _matched, _start, _end in tags:
        if label in ELOHIM_GROUPS:
            elohim += 1
        elif label in YAHWEH_GROUPS:
            yahweh += 1

    if elohim == 0 and yahweh == 0:
        return "."
    if yahweh > elohim:
        return "J"
    if elohim > yahweh:
        return "E"
    return "J"  # a tie means the compound "LORD God"; it belongs to the J side


def count_switches(verse_labels: list[str]) -> int:
    """Count how many times a chapter changes divine name, ignoring silent verses.

    BREAKPOINT 5: step INTO this (F11), then read the CALL STACK pane before
    stepping OUT (Shift+F11). Nothing calls this function directly: you get here
    through the `switches` property, which verdict() reads while render_book()
    builds the table. The call stack is how you find that out.
    """
    switches = 0
    previous = None
    for label in verse_labels:
        if label == ".":
            continue
        if previous is not None and label != previous:
            switches += 1
        previous = label
    return switches


def tally_chapter(chapter: dict, pattern: re.Pattern[str]) -> ChapterTally:
    """Run the regex over every verse of one chapter and add up the results."""
    # BREAKPOINT 3: a CONDITIONAL breakpoint on the next line, condition
    #     chapter["chapter"] == 2
    # Fifty chapters go by and the debugger stops on exactly the one you want.
    # Not on the `for verse` line below: that one is reached once per verse, so
    # the condition would hold for all 25 verses of chapter 2 instead of once.
    tally = ChapterTally(chapter=chapter["chapter"])
    for verse in chapter["verses"]:
        tags = tag_text(verse["text"], pattern)
        for label, _matched, _start, _end in tags:
            tally.counts[label] += 1
        tally.verse_labels.append(label_verse(tags))
    return tally


def tally_book(book: dict, pattern: re.Pattern[str]) -> list[ChapterTally]:
    return [tally_chapter(chapter, pattern) for chapter in book["chapters"]]


def verdict(tally: ChapterTally) -> str:
    """Turn counts into the claim a source critic would make about this chapter."""
    if tally.elohim == 0 and tally.yahweh == 0:
        return "[dim]no divine name[/dim]"
    if tally.yahweh == 0:
        return "[cyan]Elohim only (E/P)[/cyan]"
    if tally.elohim == 0:
        return "[magenta]Yahweh only (J)[/magenta]"
    plural = "" if tally.switches == 1 else "es"
    return f"[yellow]mixed ({tally.switches} switch{plural})[/yellow]"


def bar(tally: ChapterTally, width: int = 20) -> str:
    """A proportional Elohim/Yahweh bar, so the seam is visible at a glance.

    Distinguished by glyph as well as by color, so it still reads when piped to
    a file or viewed by someone who does not see cyan and magenta as different.
    """
    total = tally.elohim + tally.yahweh
    if total == 0:
        return "[dim]" + "·" * width + "[/dim]"
    elohim_cells = round(width * tally.elohim / total)
    yahweh_cells = width - elohim_cells
    return f"[cyan]{'█' * elohim_cells}[/cyan][magenta]{'░' * yahweh_cells}[/magenta]"


def cell(count: int) -> str:
    """Zeros are noise in a table like this one, so mute them."""
    return str(count) if count else "[dim]-[/dim]"


def render_book(console: Console, book_name: str, tallies: list[ChapterTally]) -> None:
    table = Table(title=f"{book_name}: Elohim vs. Yahweh, chapter by chapter")
    table.add_column("Ch", justify="right")
    table.add_column("[cyan]Elohim[/cyan]", justify="right")
    table.add_column("[magenta]Yahweh[/magenta]", justify="right")
    table.add_column("LORD God", justify="right")
    table.add_column(
        "[cyan]█[/cyan] El  [magenta]░[/magenta] YHWH", width=20, no_wrap=True
    )
    table.add_column("Reading", no_wrap=True)

    for tally in tallies:
        table.add_row(
            str(tally.chapter),
            cell(tally.elohim),
            cell(tally.yahweh),
            cell(tally.compound),
            bar(tally),
            verdict(tally),
        )
    console.print(table)


def render_summary(console: Console, book_name: str, tallies: list[ChapterTally]) -> None:
    elohim_only = [t.chapter for t in tallies if t.elohim and not t.yahweh]
    yahweh_only = [t.chapter for t in tallies if t.yahweh and not t.elohim]
    compound_chapters = [t.chapter for t in tallies if t.compound]
    seams = sorted(tallies, key=lambda t: t.switches, reverse=True)[:5]

    console.print()
    console.print(f"[bold]{book_name}: what the regex found[/bold]")
    console.print(
        f"  [cyan]Elohim-only chapters[/cyan] ({len(elohim_only)}): "
        f"{format_runs(elohim_only)}"
    )
    console.print(
        f"  [magenta]Yahweh-only chapters[/magenta] ({len(yahweh_only)}): "
        f"{format_runs(yahweh_only)}"
    )
    console.print(f"  'LORD God' compound appears in: {format_runs(compound_chapters)}")
    console.print(
        "  Most-switched chapters: "
        + ", ".join(f"{t.chapter} ({t.switches})" for t in seams if t.switches)
    )
    console.print()
    console.print(
        "[dim]Read the top of the table: the Elohim-only run at the start, then the\n"
        "abrupt turn to 'LORD God'. That is the seam where the source-critical\n"
        "pattern flips from Elohim to Yahweh.[/dim]"
    )


def format_runs(chapters: list[int]) -> str:
    """Collapse [1, 2, 3, 7] into '1-3, 7'. Numeric ranges read better than lists."""
    if not chapters:
        return "[dim]none[/dim]"
    runs: list[tuple[int, int]] = []
    start = previous = chapters[0]
    for chapter in chapters[1:]:
        if chapter == previous + 1:
            previous = chapter
            continue
        runs.append((start, previous))
        start = previous = chapter
    runs.append((start, previous))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def main() -> None:
    console = Console()
    bible = load_old_testament()
    book = find_book(bible, "Genesis")

    tallies = tally_book(book, DIVINE_NAME_RE)
    render_book(console, book["book"], tallies)
    render_summary(console, book["book"], tallies)


if __name__ == "__main__":
    main()
