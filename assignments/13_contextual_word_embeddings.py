"""Plot BERT's vector for one word across many sentences, colored by sense.

    uv run assignments/13_contextual_word_embeddings.py

Saves one PNG per entry in EXAMPLES, in the folder you run this from.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

MODEL = "bert-base-uncased"  # "uncased" = lowercases everything: `Light` == `light`

# BERT = a static word-vector table (layer 0) + 12 transformer layers. Each
# layer rewrites every token's vector by mixing in the other tokens, so the
# higher the layer, the more "context" it holds. -1 is the last layer.
LAYER = -1

# (sense, label, sentence). The model never sees the sense; it only colors
# the plot. The label marks the point on the plot.
EXAMPLES = {
    "bank": [
        ("money", "paycheck", "She deposited her paycheck at the [bank] on Friday."),
        ("money", "mortgage", "The [bank] approved our mortgage after three weeks."),
        ("money", "robbed", "He robbed two [banks] in Ohio and was caught a day later."),
        ("money", "ATM", "My [bank] charges a fee every time I use another ATM."),
        ("money", "interest", "The central [bank] raised interest rates again."),
        ("river", "picnic", "We had a picnic on the grassy [bank] of the river."),
        ("river", "canoe", "The canoe drifted toward the muddy [bank]."),
        ("river", "herons", "Herons nested along both [banks] of the stream."),
        ("river", "creek", "After the storm, the creek overflowed its [banks]."),
        ("river", "feet", "He sat on the [bank] and dangled his feet in the water."),
    ],
}

# Traps: one sense of `bank`, surrounded by the other sense's vocabulary.
EXAMPLES["bank, with traps"] = EXAMPLES["bank"] + [
    ("money", "tellers", "The [bank] by the river flooded, and the tellers waded out to their cars."),
    ("river", "fishermen", "Fishermen lined the [bank], counting their money as the boats came in."),
]

# Watch where the two sentences that *start* with Light end up.
EXAMPLES["light"] = [
    ("shine", "let", "Let there be [light]."),
    ("shine", "window", "The [light] from the window woke me."),
    ("shine", "plants", "Plants need [light] to grow."),
    ("shine", "travels", "[Light] travels faster than sound."),
    ("weight", "packed", "She packed a [light] bag for the trip."),
    ("weight", "jacket", "This jacket is [light] enough to carry."),
    ("weight", "suitcases", "[Light] suitcases are easier to lift."),
    ("weight", "box", "The box was surprisingly [light]."),
]

SENSE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # colorblind-safe

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# Return every layer's output, not just the last one.
model = AutoModel.from_pretrained(MODEL, output_hidden_states=True)
model.eval()  # Signal that we are NOT training (disable "dropout" and fix batch normalization)


def find_target(marked: str) -> tuple[str, int, int]:
    """"The [bank] closed." -> ("The bank closed.", 4, 8)"""
    start = marked.index("[")
    end = marked.index("]") - 1  # - 1 because "[" is removed
    text = marked.replace("[", "").replace("]", "")
    return text, start, end


def word_vector(marked: str) -> np.ndarray:
    """The vector BERT gives the bracketed word in this sentence."""
    text, start, end = find_target(marked)

    # offset_mapping: the (start, end) characters each token came from.
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():  # skip the bookkeeping overhead needed for training (we are not training the model, so we don't need it)
        output = model(**encoded)
    # hidden_states: 13 tensors (layer 0 + 12 layers), each shaped
    # (sentences, tokens, 768). [0] takes our one sentence: (tokens, 768).
    layer = output.hidden_states[LAYER][0]

    # Tokens inside the brackets. [CLS] and [SEP] have offsets (0, 0).
    positions = [
        i
        for i, (piece_start, piece_end) in enumerate(offsets)
        if piece_end > piece_start and piece_start >= start and piece_end <= end
    ]
    # Rows for those tokens, averaged into one row: (768,).
    vector = layer[positions].mean(dim=0).numpy()
    # np.linalg.norm is the vector's length; dividing makes it length 1.
    return vector / np.linalg.norm(vector)


def plot(name: str, examples: list[tuple[str, str, str]]) -> None:
    senses = sorted({sense for sense, _, _ in examples})
    # Stack the vectors into one array: (sentences, 768).
    vectors = np.array([word_vector(marked) for _, _, marked in examples])
    # PCA (principal component analysis) squeezes 768 columns into 2 new
    # theoretical "components" that capture the variation: (sentences, 2).
    xy = PCA(n_components=2).fit_transform(vectors)

    # fig is the whole image; ax is the plot area inside it.
    fig, ax = plt.subplots(figsize=(7, 6))  # inches
    for (x, y), (sense, label, _) in zip(xy, examples):
        color = SENSE_COLORS[senses.index(sense)]
        ax.scatter(x, y, s=120, color=color, edgecolor="black")  # s = dot size
        ax.annotate(label, (x, y), xytext=(7, 5), textcoords="offset points")

    # One legend entry per sense: empty scatters that only carry a label.
    handles = [
        ax.scatter([], [], s=120, color=SENSE_COLORS[i], edgecolor="black", label=sense)
        for i, sense in enumerate(senses)
    ]
    ax.legend(handles=handles, title="sense", loc="upper left")

    ax.set_title(f"{name}  ({MODEL}, layer {LAYER})")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.margins(0.25)  # padding around the points
    fig.tight_layout()

    slug = re.sub(r"\W+", "_", name).strip("_")  # "bank, with traps" -> "bank_with_traps"
    fig.savefig(f"13_{slug}_layer{LAYER}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for name, examples in EXAMPLES.items():
        plot(name, examples)
