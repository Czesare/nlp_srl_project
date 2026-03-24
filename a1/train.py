"""
Training, evaluation, and inference module for SRL.

Provides:
    - build_features(): extract features from preprocessed word_set
    - train_and_evaluate(): full pipeline from raw data to evaluation
    - predict_srl(): inference on standalone sentences
    - load_model(): load saved model artifacts
"""

import joblib
import csv
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix

from .preprocessing import parse_conllu
from .features import extract_features, get_spacy_doc


def build_features(word_set):
    """
    Extract features for all rows in word_set.
    Caches SpaCy parses per unique sentence to avoid redundant parsing.
    A sentence with 3 predicates is parsed once, not 3 times.
    Args: word_set: list of rows from parse_conllu
    Returns: (feature_dicts, labels): parallel lists
    """
    feature_dicts = []
    labels = []
    parse_cache = {}
    total = len(word_set)

    for i, row in enumerate(word_set):
        if i % 50000 == 0:
            print(f"  row {i}/{total} ({100*i//total}%)")

        token_id = int(row[1]) - 1   # convert 1-based to 0-based
        predicate_idx = row[4]        # already 0-based
        full_tokens = row[5]
        label = row[6]

        cache_key = tuple(full_tokens)
        if cache_key not in parse_cache:
            parse_cache[cache_key] = get_spacy_doc(full_tokens)
        doc = parse_cache[cache_key]

        all_feats = extract_features(doc, full_tokens, predicate_idx)
        feature_dicts.append(all_feats[token_id])
        labels.append(label)

    print(f"  {len(parse_cache)} unique sentences parsed.")
    return feature_dicts, labels


def train_and_evaluate(train_path, test_path, output_dir="model"):
    """
    Full pipeline: parse data, extract features, train, evaluate, save.
    Args: train_path: path to training CoNLL-U file
        test_path:  path to test CoNLL-U file
        output_dir: directory to save model artifacts and predictions
    Returns: (model, vectorizer) tuple
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === TRAINING ===
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    print("\nParsing training data...")
    train_ws, train_stats = parse_conllu(train_path)
    print(f"  Pre-replication:  {train_stats['pre_replication_sentences']} sentences, "
          f"{train_stats['pre_replication_tokens']} tokens")
    print(f"  Post-replication: {train_stats['post_replication_sentences']} sentences, "
          f"{train_stats['post_replication_tokens']} tokens")

    print("\nExtracting training features...")
    train_feats, train_labels = build_features(train_ws)

    print("\nVectorizing...")
    vec = DictVectorizer(sparse=True)
    X_train = vec.fit_transform(train_feats)
    print(f"  Feature matrix shape: {X_train.shape}")

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1, verbose=1)
    model.fit(X_train, train_labels)
    print("  Done.")

    # === EVALUATION ===
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    print("\nParsing test data...")
    test_ws, test_stats = parse_conllu(test_path)
    print(f"  Pre-replication:  {test_stats['pre_replication_sentences']} sentences, "
          f"{test_stats['pre_replication_tokens']} tokens")
    print(f"  Post-replication: {test_stats['post_replication_sentences']} sentences, "
          f"{test_stats['post_replication_tokens']} tokens")

    print("\nExtracting test features...")
    test_feats, test_labels = build_features(test_ws)

    print("\nPredicting...")
    X_test = vec.transform(test_feats)
    preds = model.predict(X_test)

    labels_sorted = sorted(set(model.classes_) | set(test_labels))


    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(test_labels, preds, labels=labels_sorted, zero_division=0))

    # Confusion matrix
    
    cm = confusion_matrix(test_labels, preds, labels=labels_sorted)
    print("Confusion matrix labels:", labels_sorted)
    print(cm)

    # === SAVE ===
    print("\nSaving predictions...")
    pred_path = output_dir / "predictions.tsv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["token", "gold_label", "predicted_label"])
        for row, g, p in zip(test_ws, test_labels, preds):
            w.writerow([row[2], g, p])
    print(f"  Predictions: {pred_path}")

    print("Saving model...")
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(vec, output_dir / "vectorizer.joblib")
    print(f"  Model: {output_dir}")

    return model, vec


def load_model(model_dir):
    """
    Load saved model and vectorizer.
    Args: model_dir: path to directory containing model.joblib and vectorizer.joblib
    Returns: (model, vectorizer) tuple
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    vec = joblib.load(model_dir / "vectorizer.joblib")
    return model, vec


def predict_srl(sentence_tokens, predicate_indicators, model, vectorizer):
    """
    Perform SRL on a standalone sentence for a single predicate.
    Args:sentence_tokens:      
    list of strings e.g. ['Pia', 'asked', 'Luis', 'to', 'write', 'this', 'sentence', '.']
    predicate_indicators: list of 0/1 marking the predicate position e.g. [0, 0, 0, 0, 1, 0, 0, 0] for predicate 'write'
    model: trained LogisticRegression
    vectorizer: fitted DictVectorizer
    Returns:
        list of (token, predicted_label) tuples
        The predicate token itself is labeled 'V'.
    """
    pred_idx = predicate_indicators.index(1)

    doc = get_spacy_doc(sentence_tokens)
    feats = extract_features(doc, sentence_tokens, pred_idx)
    X = vectorizer.transform(feats)
    labels = model.predict(X)

    results = []
    for i, (tok, lbl) in enumerate(zip(sentence_tokens, labels)):
        if i == pred_idx:
            results.append((tok, "V"))
        else:
            results.append((tok, lbl))
    return results