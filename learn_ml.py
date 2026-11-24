"""A guided tour of ORCA's machine learning, one step at a time.

Run it with:   ~/anaconda3/envs/orca/bin/python learn_ml.py

Nothing here touches the saved model in data/models/ or the database.
Read the comments next to the output; change numbers and re-run to see what moves.
"""

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

from orca.ml.dataset import synthetic_frame
from orca.ml.features import FEATURE_NAMES
from orca.ml.heuristic import score as heuristic_score
from orca.ml.train import train_model


def step(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
step("1. THE DATA: a table where each row is one trade idea")
# ---------------------------------------------------------------------------
# Machine learning starts with a table. Each ROW is one example (here: one
# option-selling candidate at one moment). Each COLUMN is a fact about it.
df = synthetic_frame(1000, seed=11)

print("Table size:", df.shape, "-> (rows, columns)")

example = df.iloc[-1]  # Last row

columns = ["dte", "n_legs", "iv_rank", "label", "return_on_credit"]
print("\nOur example:")
print(example[columns])

# ---------------------------------------------------------------------------
step("2. FEATURES (X) and LABEL (y): the question and the answer")
# ---------------------------------------------------------------------------
# FEATURES = what we know at the moment we decide (the 28 columns in FEATURE_NAMES).
# LABEL    = what happened afterwards: 1 = hit profit target before the stop, 0 = not.
# The model's whole job: learn a function  features -> probability(label == 1).
X = df[FEATURE_NAMES]
y = df["label"]

print("Features (inputs): ", len(FEATURE_NAMES), "columns, e.g.", FEATURE_NAMES[:4])
print("Label (the answer):", sorted(y.unique()), "-> 0 = loss, 1 = win")
print(f"Share of wins in the data (the 'base rate'): {y.mean():.1%}")
# Remember the base rate. A "model" that always says WIN is right this often
# while knowing nothing. Every result below must be judged against it.

# ---------------------------------------------------------------------------
step("3. TRAIN vs TEST: never grade the model on questions it has already seen")
# ---------------------------------------------------------------------------
# We hide 25% of the rows. The model learns from the other 75% (train) and is
# then graded on the hidden 25% (test). That grade estimates how it will do on
# trades it has never seen, which is the only thing that matters.
# train_model() does the split, the fitting and the grading (orca/ml/train.py).
# save=False keeps this experiment away from the real model file.
model, metrics = train_model(df, name="learning", synthetic=True, save=False)

print("Rows used to learn:", metrics["n_train"])
print("Rows held back to grade:", metrics["n_test"])

# ---------------------------------------------------------------------------
step("4. THE GRADE: how good is it, honestly?")
# ---------------------------------------------------------------------------
cut = metrics["n_train"]
test = df.iloc[cut:]
always_win = accuracy_score(test["label"], np.ones(len(test)))

print(f"Accuracy of 'always say WIN' : {always_win:.1%}   <- the knows-nothing baseline")
print(f"Accuracy of the model        : {metrics['accuracy']:.1%}")
print(f"ROC AUC                      : {metrics['roc_auc']:.3f}   (0.5 = coin flip, 1.0 = perfect)")
print(f"Brier score                  : {metrics['brier']:.3f}   (lower is better, 0.25 = useless)")
# Accuracy: how often the yes/no call was right.
# AUC:      pick one real winner and one real loser at random; how often did the
#           model give the winner the higher score? This measures RANKING, and
#           ranking candidates is exactly what ORCA uses the model for.
# Brier:    are the probabilities themselves trustworthy? When it says 70%, does
#           it win about 70% of the time?

# ---------------------------------------------------------------------------
step("5. OVERFITTING: memorising the past instead of learning from it")
# ---------------------------------------------------------------------------
train = df.iloc[:cut]
train_probs = model.classifier.predict_proba(train[FEATURE_NAMES].to_numpy(dtype=float))[:, 1]
train_auc = roc_auc_score(train["label"], train_probs)

print(f"AUC on rows it trained on : {train_auc:.3f}")
print(f"AUC on rows it never saw  : {metrics['roc_auc']:.3f}")
# The gap between these two numbers is overfitting. The model looks brilliant
# on its own training rows because it partly memorised them, noise included.
# Only the second number is real. This is why step 3 exists.

# ---------------------------------------------------------------------------
step("6. PREDICTING: score our example row")
# ---------------------------------------------------------------------------
prob, expected_return = model.score(example[FEATURE_NAMES].to_dict())

print(f"Model says: {prob:.1%} chance of a win, expected return {expected_return:+.2f}x credit")
print(f"What actually happened: label = {int(example['label'])}")
# One row proves nothing either way: a 70% call that loses is not a wrong call.
# Probabilities can only be judged over many rows, which is what step 4 did.

# ---------------------------------------------------------------------------
step("7. WHAT DID IT LEARN? Which features matter")
# ---------------------------------------------------------------------------
# Permutation importance: shuffle ONE column in the test set so it becomes
# garbage, and measure how much the AUC drops. Big drop = the model relied on it.
result = permutation_importance(
    model.classifier,
    test[FEATURE_NAMES].to_numpy(dtype=float),
    test["label"].to_numpy(dtype=int),
    scoring="roc_auc",
    n_repeats=10,
    random_state=0,
)
ranked = sorted(zip(FEATURE_NAMES, result.importances_mean, strict=True), key=lambda kv: -kv[1])

print("AUC lost when the column is scrambled (top 8):")
for name, drop in ranked[:8]:
    print(f"  {name:28s} {drop:+.3f}  {'#' * max(0, int(drop * 300))}")
# Because this data is synthetic we know the true recipe: open
# orca/ml/dataset.py and find the `logit = (...)` formula inside synthetic_frame().
# The features in that formula should be the ones at the top of this list.
# With real market data nobody hands you the recipe - this list is all you get.

# ---------------------------------------------------------------------------
step("8. THE BASELINE: is ML actually better than simple rules?")
# ---------------------------------------------------------------------------
# orca/ml/heuristic.py is a hand-written score (weights chosen by a human).
# If the trained model cannot rank better than that, the ML adds nothing.
heuristic_scores = [heuristic_score(row)[0] for row in test[FEATURE_NAMES].to_dict("records")]
heuristic_auc = roc_auc_score(test["label"], heuristic_scores)

print(f"AUC of hand-written rules : {heuristic_auc:.3f}")
print(f"AUC of the trained model  : {metrics['roc_auc']:.3f}")

# ---------------------------------------------------------------------------
step("9. MORE DATA: the cheapest way to a better model")
# ---------------------------------------------------------------------------
for n in [300, 1000, 3000, 10000]:
    _, m = train_model(synthetic_frame(n, seed=11), name="learning", synthetic=True, save=False)
    print(f"  {n:6d} rows -> test AUC {m['roc_auc']:.3f}")
# The noise in this data puts a ceiling on how good any model can get, so the
# number climbs and then flattens. Real markets are far noisier than this.

print("\nDone. Things to try next:")
print("  - change seed=11 to another number in step 1: how much do the grades move?")
print("  - in step 9, which row count is 'enough'?")
print("  - change 1000 to 5000 in step 1, re-run and watch steps 4, 5 and 8")
