"""Predict narrative vs. epistle from the features you extracted in Assignment 2.

    uv run content/machine_learning/book_classifier.py verse_features.json

The whole shape of a supervised experiment, start to finish: load the features,
hold out a test set, compare three models by cross-validation on the *training*
set, score the winner exactly once on the test set, check it against the
dumbest possible answer, look at the mistakes it made, and ask which of your
features it was actually using.

Nothing here knows a word of the New Testament. It only sees the numbers you
chose to compute -- which is the thing worth arguing about afterward.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
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
BETA = 1
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


def importances(model):
    """One number per feature, wherever this model happens to keep them.

    Logistic regression stores signed weights in .coef_; the trees store
    non-negative .feature_importances_ that sum to 1. Pipelines store neither,
    so reach past the scaler to the classifier at the end (model[-1]).
    """
    final = model[-1] if hasattr(model, "steps") else model
    if hasattr(final, "coef_"):
        return final.coef_[0]
    return final.feature_importances_


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

    # 6. The confusion matrix: the four raw counts every score above is built
    #    from. Rows are the truth, columns are the guess, and the diagonal is
    #    what went right. Read precision down the "said narrative" column (of
    #    the verses it flagged, how many deserved it) and recall across the
    #    "actually narrative" row (of the verses that deserved it, how many did
    #    it flag). Two ways to be wrong, and they are not the same mistake.
    print("\nconfusion matrix (held-out test)")
    print(pd.DataFrame(
        confusion_matrix(y_test, predicted),
        index=["actually epistle", "actually narrative"],
        columns=["said epistle", "said narrative"],
    ))

    # 7. Which features earned their keep? Three answers, because no single one
    #    is trustworthy alone:
    #
    #    model weight -- what the fitted model says about itself. Free, but
    #      only comparable within one model (signed log-odds for logistic
    #      regression, impurity drops for the trees) and famously generous to
    #      features with many distinct values, like n_tokens.
    #
    #    shuffle drop -- permutation importance: shuffle one column of the TEST
    #      set, predict again, see how far the score falls; repeat 10x and
    #      average. A feature the model leans on hurts when scrambled. Near
    #      zero means the model was ignoring it. Negative means shuffling
    #      *helped* -- that column was noise the model was chasing.
    #
    #    RFE rank -- recursive feature elimination: fit, drop the weakest
    #      feature, refit, repeat, cross-validating each subset to find the
    #      size that scores best. Rank 1 = made the final cut.
    fitted = MODELS[best]  # already fit on all of train back in step 4
    shuffled = permutation_importance(
        fitted, X_test, y_test, scoring=SCORING["fbeta"],
        n_repeats=10, random_state=RANDOM_SEED,
    )
    rfe = RFECV(  # clones the winner and refits it from scratch, subset by subset
        fitted, cv=5, scoring=SCORING["fbeta"], importance_getter=importances,
    ).fit(X_train, y_train)
    report = pd.DataFrame({
        "model weight": importances(fitted),
        f"F{BETA:g} shuffle drop": shuffled.importances_mean,
        "RFE rank": rfe.ranking_,
    }, index=X.columns)
    print(f"\nfeature evaluation for {best}")
    print(report.sort_values(f"F{BETA:g} shuffle drop", ascending=False))
    print(f"RFE keeps {rfe.n_features_} of {X.shape[1]}: "
          f"{', '.join(X.columns[rfe.support_])}")

    # Two warnings before you act on that table. First, the columns can
    # disagree: set BETA=0.5 and the decision tree wins, and it spends a real
    # share of its weight on n_tokens while that column's shuffle drop sits at
    # or below zero. The tree kept splitting on verse length and gained nothing
    # by it. Believe the shuffle -- it is measured on data the model never saw.
    # Second, both measures split the credit between features that say the same
    # thing: near-duplicate columns each look useless, because shuffling one
    # leaves the other to cover for it. A feature scoring zero is either dead
    # weight or a twin, and the fix is different -- check before you delete it.


if __name__ == "__main__":
    main()
