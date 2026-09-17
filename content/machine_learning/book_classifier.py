"""Predict narrative vs. epistle from the features you extracted in Assignment 2.

    uv run content/machine_learning/book_classifier.py verse_features.json

The whole shape of a supervised experiment, start to finish: load the features,
hold out a test set, compare three models by cross-validation on the *training*
set, score the winner exactly once on the test set, and check it against the
dumbest possible answer.

Nothing here knows a word of the New Testament. It only sees the numbers you
chose to compute -- which is the thing worth arguing about afterward.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Fixed once, used everywhere a coin gets flipped: the train/test split and the
# two tree models. Same seed, same numbers on your machine and mine.
RANDOM_SEED = 42

# The label. Five narrative books; the other 22 are letters (plus Revelation).
NARRATIVE = {"Matthew", "Mark", "Luke", "John", "Acts"}

# The metric. beta=2 weights recall twice as heavily as precision: we would
# rather over-flag a narrative verse than miss one. This single number decides
# which model wins -- change it and see, as the last block explains. beta=1 weights
# precision and recall equally. beta=0.5 weights precision twice as much as recall.
BETA = 2
SCORING = {
    "precision": "precision",
    "recall": "recall",
    "fbeta": make_scorer(fbeta_score, beta=BETA),
}

# The three candidates. Only logistic regression needs the scaler -- n_tokens
# runs around 20 while the rates run around 0.1 -- so it gets wrapped in a
# Pipeline. Trees split one column at a time and do not care about units.
MODELS = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "decision tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
    "random forest": RandomForestClassifier(random_state=RANDOM_SEED),
}

ROW = "{:<22}{:>11}{:>9}{:>7}"


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "verse_features.json")

    # 1. Load. Every column but the two bookkeeping ones is a feature, so a
    #    feature you add in Assignment 2 shows up here for free.
    df = pd.read_json(path)
    X = df.drop(columns=["id", "book"])
    y = df["book"].isin(NARRATIVE)
    print(X.describe())
    print(f"{len(df)} verses, {X.shape[1]} features, {y.mean():.1%} narrative")
    print(f"features: {', '.join(X.columns)}\n")

    # 2. Hold out a test set and do not touch it again until step 4.
    #    stratify=df["book"] keeps all 27 books proportional in both halves --
    #    stronger than stratifying on y, which would happily put most of
    #    Revelation on one side and skew the epistle class.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=df["book"], random_state=RANDOM_SEED
    )

    # 3. Compare the models by 5-fold cross-validation on the TRAINING set only.
    #    Each model is fit five times, on 4/5 of train, and scored on the 1/5 it
    #    did not see. Averaging five estimates beats trusting one lucky split.
    print(ROW.format("5-fold CV on train", "precision", "recall", f"F{BETA:g}"))
    cv_fbeta = {}
    for name, model in MODELS.items():
        scores = cross_validate(model, X_train, y_train, cv=5, scoring=SCORING)
        cv_fbeta[name] = scores["test_fbeta"].mean()
        print(ROW.format(
            name,
            f"{scores['test_precision'].mean():.3f}",
            f"{scores['test_recall'].mean():.3f}",
            f"{cv_fbeta[name]:.3f}",
        ))

    # 4. Pick the winner, refit it on ALL of train, score it once on test.
    #    Once is the whole point: tune against the test set and it stops being a
    #    measurement of anything.
    best = max(cv_fbeta, key=cv_fbeta.get)
    predicted = MODELS[best].fit(X_train, y_train).predict(X_test)
    print(f"\nbest by F{BETA:g}: {best}")
    print(ROW.format("held-out test", "precision", "recall", f"F{BETA:g}"))
    print(ROW.format(
        best,
        f"{precision_score(y_test, predicted):.3f}",
        f"{recall_score(y_test, predicted):.3f}",
        f"{fbeta_score(y_test, predicted, beta=BETA):.3f}",
    ))

    # 5. Score the stupidest possible answer right next to the real one.
    #    Guessing "narrative" for every verse gets recall 1.000 for free, and at
    #    beta=2 that is worth so much it beats all three models. Not a bug in
    #    the models -- it is what beta=2 asks for on a 60/40 split. Set BETA=0.5
    #    and two things change: the decision tree wins instead of logistic
    #    regression, and it finally clears the baseline. The metric is not a
    #    neutral scoreboard. It picks the winner.
    always = [True] * len(y_test)
    print(ROW.format(
        "always narrative",
        f"{precision_score(y_test, always):.3f}",
        f"{recall_score(y_test, always):.3f}",
        f"{fbeta_score(y_test, always, beta=BETA):.3f}",
    ))


if __name__ == "__main__":
    main()
