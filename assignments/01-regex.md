# Assignment 1: Regular Expressions

**Goal:** get fluent enough with regular expressions that you can write one
without looking up the syntax every time -- and find out where regexes stop
working.

## Part 1: Learn the syntax

Work through **all 15 lessons** of the interactive tutorial at
[regexone.com](https://regexone.com/).

Do them in order, and actually type the patterns rather than pasting them --
the point is to build the muscle memory. Each lesson gives you a set of strings
to match and a set to skip; a lesson is done when the site marks every line
correct.

Budget about an hour. If a lesson takes more than ten minutes, move on and come
back to it -- the later lessons often make the earlier ones click.

## Part 2: Write a Python script

Write a script called `extract.py` that runs a text file through two regexes
and prints what it finds.

The text is
[content/regex/data/dh_listserv_digest.txt](../content/regex/data/dh_listserv_digest.txt)
-- about 1,000 words of a fictional department listserv digest. Do **not** paste
it into your script. Open it and read it in:

```python
from pathlib import Path

TEXT = Path("dh_listserv_digest.txt").read_text(encoding="utf-8")
```

Put the file next to `extract.py`, or give the full path to wherever you saved
it. (`encoding="utf-8"` is not decoration -- leave it off and your script will
behave differently on different machines.)

Your script must do two things.

### 1. Extract every email address

Print each address you find. There are **10** in the text. Some things to get
right:

- local parts contain dots, hyphens, and plus signs (`t.brandt+archive@...`)
- domains have subdomains (`gradsec@dh.example.edu`)
- addresses are capitalized inconsistently (`Support@DHConf.example.net`)
- an address at the end of a sentence is followed by a period that is **not**
  part of the address

Getting all 10 with no false positives is achievable with a single pattern.

### 2. Extract every plural noun

Print each plural noun you find, with a count of how often it occurs. This one
is much harder than it looks, and finding out *why* is the real assignment.

The obvious rule -- "a word ending in `-s` is a plural noun" -- is wrong in both
directions, and the text is seeded with counterexamples:

- **ends in -s, not a plural noun:** `analysis`, `campus`, `bus`, `chorus`,
  `Thomas`, `always`, `across`, `perhaps`, `this`, `statistics`, and every
  third-person verb in the text (`he runs`, `she focuses`, `nobody watches`)
- **a plural noun, does not end in -s:** `children`, `women`, `men`, `data`,
  `criteria`, `phenomena`, `corpora`, `feet`

You will not get this perfect. Nobody does with regexes alone -- that is the
lesson. Get as close as you can by stacking simple ideas: a pattern for the
shape of the word, a list of exceptions, a look at the word next door (`a
statistics course` vs. `the criteria`), a lookup table for the irregulars.

You may use AI to help write the script, but you must understand and be able to explain every part of it!

### 3. Report your errors

At the bottom of your output, print a short list of words your script got
wrong -- ones it claimed were plural nouns and are not, and plural nouns in the
text it missed. Read your own output to find them.

This part is not optional and it is not busywork. A method whose error rate you
have measured is a method; one whose error rate you have not measured is a
guess with good posture. You will do this for the rest of the semester.

## What to turn in

Submit both in Learning Suite:

1. A screenshot showing **regexone lesson 15 completed** (all lines green).
2. Your `extract.py`, plus its output pasted into a text file or comment.

And a short note (3-5 sentences) answering:

- Which regexone lesson gave you the most trouble, and what made it click?
- What was the last plural-noun error you tried to fix, and did fixing it break
  something else?

## Reference

Keep these open while you work:

- [regex101.com](https://regex101.com/) -- paste a pattern and it explains every
  piece of it. Set the flavor to **Python** to match what we use in class.
- [Python `re` docs](https://docs.python.org/3/library/re.html) -- especially
  `re.findall`, `re.finditer`, and `re.VERBOSE`, which lets you write a long
  pattern across several lines with comments.
- [dh_listserv_digest.txt](../content/regex/data/dh_listserv_digest.txt) -- the
  text your script reads. Open it in your editor and read it before you write a
  single pattern; you cannot write a good regex for text you have not looked at.
- [el_yhwh_merge.py](../content/regex/el_yhwh_merge.py) -- a worked example from
  class of a regex doing real scholarly work. Not a solution to this
  assignment, but the shape your script can aim for.
