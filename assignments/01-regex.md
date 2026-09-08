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

Write a script called `extract.py` that runs the text at the bottom of this
page through two regexes and prints what it finds.

Paste the text into your script as a triple-quoted string:

```python
TEXT = """
DIGHT 360 LISTSERV DIGEST -- WEEK FOUR
...paste the whole thing here...
"""
```

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
- [el_yhwh_merge.py](../content/regex/el_yhwh_merge.py) -- a worked example from
  class of a regex doing real scholarly work. Not a solution to this
  assignment, but the shape your script can aim for.

---

## The text

Copy everything between the rules.

---

```text
DIGHT 360 LISTSERV DIGEST -- WEEK FOUR
Compiled for the Digital Humanities Working Group, Harold B. Lee Library

ANNOUNCEMENTS

The corpus workshop scheduled for Thursday has moved to the media lab on the
fourth floor. Two of the three rooms we normally use are being rewired this
month, and the technicians promised the cables would be finished by Friday.
They always promise that. If you cannot find the room, text Marta Villalobos at
the number on the door or write to marta.villalobos@example.edu and she will
come down and let you in.

Thomas Brandt has finished migrating the department's older projects off the
retired server. All of the files are now in the shared archive, though the
directory names are a mess and several of the folders contain nothing but empty
subfolders. If you are missing something, send the project name and the
approximate dates to t.brandt+archive@example.edu rather than filing a ticket
with campus IT; the ticket system routes everything to a queue nobody watches.

Registration for the spring practicum opens Monday. Nine seats, and last year
the seats were gone in under an hour. The prerequisites have changed: two
courses in text analysis instead of one, plus either a statistics course or the
proseminar. Questions about the criteria go to the graduate secretary,
gradsec@dh.example.edu. Questions about whether your particular transcripts
satisfy the criteria go to the director, not to the secretary, because she has
been fielding those emails for years and it is not her job.

PROJECT REPORTS

Ana Reyes reports that the newspaper digitization project has cleared its
second milestone. The team has now processed roughly forty thousand pages from
eleven regional papers, and the optical character recognition is good enough
that the searches return usable results for anything printed after 1890. Before
1890 the scans are poor, the typefaces are strange, and the machines produce
garbage. Ana notes that the errors are not random: certain letters fail in
predictable ways, and a handful of substitution rules fix a surprising
proportion of them. She is writing the rules up as a short paper and would
welcome readers. Reach her at a.reyes@newspapers.example.org.

The manuscript group has hit a wall. Their images are fine, their metadata is
fine, but the two catalogs they are trying to merge disagree about nearly every
date, and the disagreements are not small. One catalog assigns a manuscript to
the twelfth century; the other assigns the same manuscript to the fourteenth.
The group's working hypothesis is that the earlier catalog copied its dates
from a nineteenth-century index whose compilers were guessing. Resolving this
will take months. Volunteers with paleography training are welcome; write to
manuscripts@example.edu.

Jae-won Park's dissertation corpus is finally public. It contains sermons,
diaries, and letters from three Utah settlements between 1852 and 1910, cleaned
and encoded in TEI. The children of the original donors placed a few
restrictions on the diaries, so some documents are available only on the
library's machines. Everything else can be downloaded. Jae-won asks that
analyses drawing on the corpus cite the version number, since the corpus is
still being corrected and the indices shift between releases. Contact:
jaewon.park@example.edu.

METHODS CORNER

Several people asked after last week's session how the extraction scripts work.
The short answer is that regular expressions are excellent at finding things
with a fixed shape and poor at finding things defined by meaning. Email
addresses have a shape: some characters, an at sign, a domain with at least one
dot. A pattern catches them almost perfectly.

Nouns are a different problem. English plurals mostly end in -s, which sounds
like a rule until you look at real sentences. The word "analysis" ends in -s
and is singular. So do "campus" and "bus" and "chorus" and the name Thomas. The
words "always" and "across" and "perhaps" end in -s and are not nouns at all.
Verbs end in -s constantly: he runs, she focuses, the argument depends on the
evidence. Meanwhile the plurals a scholar most wants -- children, women, men,
data, criteria, phenomena, corpora, indices, theses, feet -- do not end in -s in
any way a pattern can predict, or do not follow the pattern at all.

So a plural-noun regex is a filter, not an oracle. It gives you candidates. You
prune the candidates with a stoplist, you check the neighbors of each word, and
you accept that some errors remain. This is a completely ordinary situation in
computational text analysis, and the mature response is not to abandon the
method but to measure how wrong it is and report the number.

The counterargument, which Devin Osei made forcefully on Tuesday, is that a
part-of-speech tagger solves this properly and takes four lines of code. He is
right. He is also assuming the tagger was trained on text resembling yours,
which for nineteenth-century diaries and Middle English lyrics and OCR sludge is
frequently false. Taggers fail quietly on unfamiliar material; a regex fails
loudly, and you can read it. Devin's slides are at
devin.osei@linguistics.example.edu if you want them.

CALLS AND DEADLINES

Abstracts for the regional DH conference are due 15 November. Two hundred and
fifty words, no citations. The organizers ask that presenters indicate whether
they need machines or will bring laptops. Submissions go through the conference
site, but technical problems go to Support@DHConf.example.net, which is
monitored by actual humans.

The library has funds for three graduate travel awards this year. Applications
close 1 December and require a budget, a letter, and evidence that the papers
have been accepted somewhere. Previous winners are not eligible. Details from
awards-committee@library.example.edu.

Finally: the alumni panel is 20 November at four o'clock. Six panelists, all
working outside the academy, on what the degree turned out to be good for.
Refreshments afterward. RSVP to events@example.edu so we know how many chairs
to set out.
```
