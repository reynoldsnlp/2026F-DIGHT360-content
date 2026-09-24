"""Compare a shelf of classifiers on a multiclass problem, then interrogate the features.

    uv run content/machine_learning/multiclass_classifier.py readability_features.json

The same experiment as book_classifier.py -- hold out a test set, compare models
by cross-validation on the *training* set, score the winner exactly once on the
test set, check it against the dumbest possible answer, read the mistakes, ask
which features it was using -- but with three things that only matter once you
have more than two classes:

  * Every metric has to be told how to average across classes. The binary
    defaults do not raise an error here; they quietly return nan.
  * If the classes are ORDERED, as reading levels are, most metrics do not know
    it. Confusing "elementary" with "advanced" counts exactly the same as
    confusing it with "intermediate", which is not how anyone grades anything.
  * If rows come in related groups -- three rewrites of one article -- a random
    split puts siblings on both sides and the test score becomes fiction.

Nothing here knows what a reading level is. Point it at any table of numbers
with a label column and it will run; the constants directly below are the only
thing you change.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Everything you change for a new dataset lives in this block.
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# The column holding the answer, and the columns that are bookkeeping rather
# than features. Everything else in the file is treated as a feature, so a
# feature you add in Assignment 3 shows up here for free.
LABEL_COLUMN = "level"
ID_COLUMNS = ("id", "group")

# Rows that must not be split across train and test. In the OneStopEnglish
# corpus the same article exists at all three levels, so "Amazon" names three
# rows sharing a topic, a vocabulary and most of their proper nouns.
#
# Textbook leakage would mean the model studies the test set's subject matter in
# advance and posts a flattering score. On a corpus built in matched sets it
# goes the other way: topic is balanced across the classes by construction, and
# a model that has seen this article's words under the other two labels learns
# to rule out the right one. On tf-idf features over these texts a group-aware
# split scores 0.77 and a random split 0.28, which is below chance. Either way
# the number you get is about the corpus design rather than about reading level.
# Set to None only when rows are genuinely independent.
GROUP_COLUMN = "group"

# The classes IN ORDER, if they have one. This single list does three jobs:
# it puts the confusion matrix in a readable sequence, it lets us score how FAR
# wrong a wrong answer was, and it turns on quadratic weighted kappa below.
# Leave it empty for unordered classes (topic, author, genre) and the script
# falls back to plain macro-averaged scores.
#
# Do not be tempted to skip it and let sklearn sort the labels itself: sklearn
# sorts alphabetically, which here would give advanced, elementary, intermediate
# -- an order in which the middle class is printed on the outside and the
# confusion matrix becomes unreadable. With numeric grade levels it is worse,
# since "10" sorts before "2".
CLASS_ORDER = ["elementary", "intermediate", "advanced"]

# beta for the F-score: 1 weights precision and recall equally, 2 favors recall,
# 0.5 favors precision. Averaged over classes with average="macro", which gives
# every class an equal vote regardless of size. Switch to "weighted" and big
# classes dominate; on a balanced corpus the two agree, on a skewed one they can
# name different winners. That disagreement is worth looking at, not smoothing over.
BETA = 1
AVERAGE = "macro"

# ---------------------------------------------------------------------------

# The shelf. Grouped by family, because the families fail in different ways and
# a comparison across 13 near-identical linear models teaches nothing.
#
# The first two groups are the historical canon of text classification: before
# neural models, a linear SVM or a naive Bayes over word counts was the thing to
# beat, and maximum-entropy (which sklearn calls LogisticRegression) was the
# other standard answer. Worth knowing that they earned those reputations on
# SPARSE, high-dimensional bag-of-words vectors -- tens of thousands of columns,
# almost all zero. What you are feeding them here is the opposite: a dozen dense
# engineered numbers. Do not be surprised when the trees and the boosted
# ensemble, which are built for exactly that shape of data, beat the classics.
# Being clear about why a famous model underperforms is worth more than picking
# the winner.
#
# Scaling: anything that measures distances or follows gradients needs the
# columns on a comparable scale, since words_per_sentence runs around 20 while
# type_token_ratio runs around 0.4. Those get wrapped in a Pipeline with a
# StandardScaler. Trees split one column at a time and are indifferent to units,
# so they get nothing. MultinomialNB is a third case: it models counts and
# rejects negative numbers outright, so it gets MinMaxScaler instead.
MODELS = {
    # Linear and margin-based -- the classic text-classification workhorses.
    "logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)
    ),
    "linear SVM": make_pipeline(
        StandardScaler(), LinearSVC(max_iter=10000, random_state=RANDOM_SEED)
    ),
    "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_SEED)),
    "ridge classifier": make_pipeline(StandardScaler(), RidgeClassifier(random_state=RANDOM_SEED)),
    "SGD (hinge loss)": make_pipeline(
        StandardScaler(), SGDClassifier(max_iter=5000, random_state=RANDOM_SEED)
    ),
    "perceptron": make_pipeline(StandardScaler(), Perceptron(random_state=RANDOM_SEED)),
    # Probabilistic. MultinomialNB is the textbook text classifier and is being
    # used here well outside its assumptions; GaussianNB is the version that
    # actually fits continuous features like these.
    "multinomial NB": make_pipeline(MinMaxScaler(), MultinomialNB()),
    "gaussian NB": make_pipeline(StandardScaler(), GaussianNB()),
    # Instance-based: no training to speak of, just a vote among neighbors.
    "k-nearest neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    # Trees and ensembles: no scaling, and they find interactions between
    # features without being told the interactions exist.
    "decision tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
    "random forest": RandomForestClassifier(random_state=RANDOM_SEED),
    "gradient boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    # A small neural net, for scale: one hidden layer, no embeddings, no
    # pretraining. On a dozen features and a few hundred rows it has nothing to
    # work with that the simpler models do not.
    "neural net (MLP)": make_pipeline(
        StandardScaler(), MLPClassifier(max_iter=3000, random_state=RANDOM_SEED)
    ),
}


def build_scoring(class_order):
    """The metrics, every one of them told explicitly how to average.

    This is the part that does not survive the jump from two classes to three.
    Left at their defaults, precision/recall/f-score use average="binary", and
    inside cross_validate that failure is caught and recorded as nan rather than
    raised -- so you get a full table of nan, and max() over it returns whichever
    model happens to be listed first. A tidy printout and an arbitrary winner.
    """
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=AVERAGE, zero_division=0),
        "recall": make_scorer(recall_score, average=AVERAGE, zero_division=0),
        f"F{BETA:g}": make_scorer(fbeta_score, beta=BETA, average=AVERAGE, zero_division=0),
    }
    if class_order is not None:
        # Quadratic weighted kappa: the standard metric for ordered classes, and
        # the only one here that cares HOW wrong a wrong answer is. Guessing
        # "advanced" for an elementary text is penalized four times as hard as
        # guessing "intermediate" -- (2 rungs)^2 versus (1 rung)^2. It is also
        # chance-corrected, so 0 means "no better than guessing at random" and
        # negative means worse than that.
        scoring["QWK"] = make_scorer(
            cohen_kappa_score, weights="quadratic", labels=class_order
        )
    return scoring


def importances(model):
    """One number per feature, or None if this model cannot say.

    Logistic regression and the other linear models keep signed weights in
    .coef_; the trees keep non-negative .feature_importances_; pipelines keep
    neither, so reach past the scaler to the classifier at the end (model[-1]).

    The multiclass wrinkle is in the shape. With two classes .coef_ is
    (1, n_features) and coef_[0] is the whole story. With three it is
    (3, n_features) -- one row per class -- and coef_[0] is a correctly shaped,
    entirely plausible, completely wrong answer describing only the first
    class's one-vs-rest boundary. Nothing errors. Taking the mean magnitude
    across rows asks the question we actually meant: how much does this feature
    move any decision at all? The cost is the sign, which no longer means
    anything once several classes disagree about the direction.

    kNN, the RBF SVM, naive Bayes and the MLP have no per-feature weights of any
    kind. That is a fact about those models, not a bug: returning None here
    means the caller falls back to permutation importance, which works on
    anything you can call .predict() on.
    """
    final = model[-1] if hasattr(model, "steps") else model
    coefficients = getattr(final, "coef_", None)
    if coefficients is not None:
        return np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    return getattr(final, "feature_importances_", None)


def ordinal_errors(y_true, y_pred, class_order):
    """How far off, in rungs, averaged -- and how often within one rung.

    Accuracy and F-score cannot see this. A model that is always off by exactly
    one level and a model that reliably calls advanced texts elementary can post
    the same macro F1; these two numbers separate them immediately.
    """
    rung = {label: position for position, label in enumerate(class_order)}
    distance = np.abs(
        np.array([rung[label] for label in y_true]) - np.array([rung[label] for label in y_pred])
    )
    return distance.mean(), (distance <= 1).mean()


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "readability_features.json")

    # 1. Load. Split the table into features, label, and grouping.
    df = pd.read_json(path)
    y = df[LABEL_COLUMN].astype(str)
    X = df.drop(columns=[LABEL_COLUMN, *[c for c in ID_COLUMNS if c in df.columns]])

    # A feature that returned text instead of a number would otherwise crash
    # several models with a confusing error a long way from the cause.
    non_numeric = X.columns.difference(X.select_dtypes(include="number").columns)
    if len(non_numeric):
        print(f"WARNING: dropping non-numeric feature(s): {', '.join(non_numeric)}\n")
        X = X.select_dtypes(include="number")

    groups = df[GROUP_COLUMN] if GROUP_COLUMN and GROUP_COLUMN in df.columns else None

    observed = set(y.unique())
    class_order = CLASS_ORDER or None
    if class_order is not None and set(class_order) != observed:
        # Better to stop than to silently score against a stale label list.
        raise SystemExit(
            f"CLASS_ORDER is {CLASS_ORDER} but the data contains {sorted(observed)}. "
            "Fix CLASS_ORDER, or set it to [] if these classes are unordered."
        )
    labels = class_order or sorted(observed)

    scoring = build_scoring(class_order)
    # The metric that picks the winner. With ordered classes that should be the
    # one that knows they are ordered.
    deciding = "QWK" if class_order else f"F{BETA:g}"

    print(X.describe().round(3).to_string())
    print(f"\n{len(df)} rows, {X.shape[1]} features, {len(labels)} classes")
    print(f"features: {', '.join(X.columns)}")
    print(f"class balance: {dict(y.value_counts().reindex(labels))}")
    if groups is not None:
        print(f"grouping: {groups.nunique()} groups over {len(df)} rows ({GROUP_COLUMN})")

    # 2. Hold out a test set and do not touch it again until step 4.
    #    With groups, the split has to be group-disjoint AND still keep the
    #    classes proportional, which is what StratifiedGroupKFold does; taking
    #    its first fold gives an 80/20 split. Without groups, an ordinary
    #    stratified 5-fold split does the same job.
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        train_rows, test_rows = next(splitter.split(X, y, groups))
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_groups = groups.iloc[train_rows]
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        train_rows, test_rows = next(splitter.split(X, y))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_groups = None

    X_train, X_test = X.iloc[train_rows], X.iloc[test_rows]
    y_train, y_test = y.iloc[train_rows], y.iloc[test_rows]
    print(f"\ntrain: {len(X_train)} rows, test: {len(X_test)} rows")

    # 3. Compare every model by 5-fold cross-validation on the TRAINING set.
    #    Each is fit five times on 4/5 of train and scored on the 1/5 it did not
    #    see. Averaging five estimates beats trusting one lucky split -- and with
    #    thirteen models on a few hundred rows, one lucky split is very likely.
    #
    #    error_score="raise" is deliberate. The default is nan, which turns a
    #    broken model or a misconfigured metric into a blank cell you will
    #    scroll past. Loud beats tidy.
    print(f"\n5-fold CV on train, sorted by {deciding}")
    results = {}
    for name, model in MODELS.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, groups=cv_groups,
            scoring=scoring, error_score="raise",
        )
        results[name] = {metric: scores[f"test_{metric}"].mean() for metric in scoring}
    table = pd.DataFrame(results).T.sort_values(deciding, ascending=False)
    print(table.round(3).to_string())

    # 4. Pick the winner, refit on ALL of train, score it once on test.
    #    Once is the whole point: tune against the test set and it stops being a
    #    measurement of anything. Thirteen models make this rule harder to keep
    #    and more important -- every extra model is another chance for one of
    #    them to look good on a particular split by luck alone.
    best = table.index[0]
    fitted = MODELS[best].fit(X_train, y_train)
    predicted = fitted.predict(X_test)

    # 5. Score the stupidest possible answers right next to the real one.
    #    "Always guess the biggest class" is the baseline every published
    #    accuracy should be read against. On three balanced classes it lands
    #    near 33% accuracy, and its macro F1 is far worse than that -- it scores
    #    zero on two of the three classes, and macro-averaging gives those
    #    classes the same vote as the one it gets right.
    print(f"\nheld-out test -- winner ({best}) vs. baselines")
    final = {}
    contenders = {
        best: predicted,
        "always most frequent": DummyClassifier(strategy="most_frequent")
        .fit(X_train, y_train).predict(X_test),
        "random by class rate": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
        .fit(X_train, y_train).predict(X_test),
    }
    for name, guess in contenders.items():
        row = {
            "accuracy": accuracy_score(y_test, guess),
            "precision": precision_score(y_test, guess, average=AVERAGE, zero_division=0),
            "recall": recall_score(y_test, guess, average=AVERAGE, zero_division=0),
            f"F{BETA:g}": fbeta_score(y_test, guess, beta=BETA, average=AVERAGE, zero_division=0),
        }
        if class_order:
            row["QWK"] = cohen_kappa_score(y_test, guess, weights="quadratic", labels=class_order)
            row["rungs off"], row["within 1"] = ordinal_errors(y_test, guess, class_order)
        final[name] = row
    print(pd.DataFrame(final).T.round(3).to_string())

    # 6. The confusion matrix: the raw counts every score above is built from.
    #    Rows are the truth, columns are the guess, the diagonal is what went
    #    right. With two classes there were two ways to be wrong; with k classes
    #    there are k*(k-1), and they are not equally bad. Read it for SHAPE:
    #    errors packed against the diagonal mean a model with the right idea and
    #    a shaky threshold, while mass in the far corners means it has the scale
    #    backwards somewhere. This is the payoff for putting CLASS_ORDER in
    #    order -- alphabetized labels scatter that structure into noise.
    print("\nconfusion matrix (held-out test)")
    print(pd.DataFrame(
        confusion_matrix(y_test, predicted, labels=labels),
        index=[f"actually {label}" for label in labels],
        columns=[f"said {label}" for label in labels],
    ).to_string())

    # The per-class breakdown behind the macro average. Worth reading whenever
    # one class drags the average down: the fix for a class the model never
    # predicts is different from the fix for one it predicts constantly.
    print("\nper-class detail (held-out test)")
    print(classification_report(y_test, predicted, labels=labels, zero_division=0))

    # 7. Which features earned their keep? Up to three answers, because no
    #    single one is trustworthy alone:
    #
    #    model weight -- what the fitted model says about itself. Free, but only
    #      comparable within one model, and for a multiclass linear model it is
    #      an average of magnitudes across the per-class boundaries (see
    #      importances above), so the sign is gone. Tree importances are
    #      famously generous to features with many distinct values.
    #
    #    shuffle drop -- permutation importance: shuffle one column of the TEST
    #      set, predict again, see how far the score falls; repeat 10x and
    #      average. A feature the model leans on hurts when scrambled. Near zero
    #      means it was being ignored. Negative means shuffling *helped* -- that
    #      column was noise the model was chasing.
    #
    #    RFE rank -- recursive feature elimination: fit, drop the weakest
    #      feature, refit, repeat, cross-validating each subset to find the size
    #      that scores best. Rank 1 = made the final cut. Needs model weights,
    #      so it is skipped for models that have none.
    print(f"feature evaluation for {best}")
    shuffled = permutation_importance(
        fitted, X_test, y_test, scoring=scoring[deciding],
        n_repeats=10, random_state=RANDOM_SEED,
    )
    report = pd.DataFrame({f"{deciding} shuffle drop": shuffled.importances_mean}, index=X.columns)

    weights = importances(fitted)
    if weights is None:
        print(f"({best} exposes no per-feature weights, so only the shuffle test applies.)")
    else:
        report.insert(0, "model weight", weights)
        rfe = RFECV(  # clones the winner and refits it from scratch, subset by subset
            fitted, cv=cv, scoring=scoring[deciding], importance_getter=importances,
        ).fit(X_train, y_train, groups=cv_groups)
        report["RFE rank"] = rfe.ranking_

    print(report.sort_values(f"{deciding} shuffle drop", ascending=False).round(4).to_string())
    if weights is not None:
        print(f"RFE keeps {rfe.n_features_} of {X.shape[1]}: "
              f"{', '.join(X.columns[rfe.support_])}")

    # Three warnings before you act on that table.
    #
    # First, the columns can disagree, and when they do, believe the shuffle --
    # it is the only one measured on data the model never saw. A tree can spend
    # a real share of its weight on a column whose shuffle drop sits at or below
    # zero, which means it kept splitting on that column and gained nothing.
    #
    # Second, both measures split the credit between features that say the same
    # thing: near-duplicate columns each look useless, because shuffling one
    # leaves the other to cover for it. A feature scoring zero is either dead
    # weight or a twin, and the fix is different -- check the correlations
    # before you delete anything.
    #
    # Third, this table describes ONE model. Run it again with a different
    # winner at the top and the ranking will move, sometimes a lot. "Which
    # features matter" is not a question the data answers by itself; it is a
    # question a model answers, and different models disagree.


if __name__ == "__main__":
    main()
