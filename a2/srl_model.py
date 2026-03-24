"""
Training, evaluation, and inference for BERT-based SRL.
"""

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from sklearn.metrics import classification_report, confusion_matrix
from .preprocessing_bert import insert_predicate_markers


def extract_word_level_predictions(predictions, labels, tokenized_dataset,
                                    id2label, records):
    """
    Convert subword-level model predictions to word-level predictions,
    matching the original token count expected by the shared task.
    Args:
        predictions:      raw logits 
        labels:           gold label
        tokenized_dataset: the tokenized HuggingFace dataset (for word_ids)
        id2label:         dict mapping int -> label string
        records:          original preprocessed records (for token strings)
    Returns:
        all_results: list of (token, gold_label, pred_label) tuples
    """
    pred_ids = np.argmax(predictions, axis=2)
    all_results = []

    for i in range(len(records)):
        rec = records[i]
        original_tokens = rec["tokens"]
        original_labels = rec["labels"]
        pred_idx = rec["predicate_idx"]

        # Get the tokenized example's word_ids
        example = tokenized_dataset[i]
        input_ids = example["input_ids"]
        label_ids = example["labels"]

        # Reconstruct word_ids from the label alignment
        seq_preds = pred_ids[i]
        seq_labels = labels[i]

        # Walk through and collect first-subtoken predictions per original word
        seen_word_indices = set()
        word_predictions = {}

        for pos in range(len(seq_labels)):
            if seq_labels[pos] == -100:
                continue
            word_predictions[len(seen_word_indices)] = (
                seq_labels[pos],
                seq_preds[pos],
            )
            seen_word_indices.add(len(seen_word_indices))

        token_idx = 0
        labeled_idx = 0
        for t_i, token in enumerate(original_tokens):
            if original_labels[t_i] == "V":
                # Predicate: trained as O, use model's actual prediction
                if labeled_idx in word_predictions:
                    gold_id, pred_id = word_predictions[labeled_idx]
                    all_results.append((token, "O", id2label[pred_id]))
                else:
                    all_results.append((token, "O", "O"))
                labeled_idx += 1
            else:
                if labeled_idx in word_predictions:
                    gold_id, pred_id = word_predictions[labeled_idx]
                    gold_str = id2label[gold_id]
                    pred_str = id2label[pred_id]
                    all_results.append((token, gold_str, pred_str))
                else:
                    all_results.append((token, original_labels[t_i], "O"))
                labeled_idx += 1

    return all_results


def evaluate_predictions(all_results, label_list):
    """
    Produce sklearn classification report and confusion matrix.
    Args:
        all_results: list of (token, gold, pred) 
        label_list:  list of label strings 
    Returns:
        report_str: classification report as string
        cm:         confusion matrix
        cm_labels:  labels used in confusion matrix
    """
    golds = [r[1] for r in all_results]
    preds = [r[2] for r in all_results]

    present_labels = label_list

    report_str = classification_report(golds, preds, labels=present_labels,
                                        zero_division=0)

    cm = confusion_matrix(golds, preds, labels=present_labels)

    return report_str, cm, present_labels


def save_predictions_tsv(all_results, output_path):
    """Save predictions as TSV: token, gold_label, predicted_label."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("token\tgold_label\tpredicted_label\n")
        for token, gold, pred in all_results:
            f.write(f"{token}\t{gold}\t{pred}\n")


def predict_srl(sentence_tokens, predicate_indicators, model, tokenizer,
                id2label, device=None):
    """
    Perform SRL on a standalone sentence given the predicate.
    Args:
        sentence_tokens:     list of strings
        predicate_indicators: list of 0/1
        model:               trained AutoModelForTokenClassification
        tokenizer:           corresponding tokenizer
        id2label:            dict mapping int -> label string
        device:              torch device 
    Returns:
        list of (token, label)
        Predicate token always gets label 'V'.
    """
    if device is None:
        device = next(model.parameters()).device

    pred_idx = predicate_indicators.index(1)

    # Insert markers
    marked_tokens = (
        sentence_tokens[:pred_idx]
        + ["[unused0]"]
        + [sentence_tokens[pred_idx]]
        + ["[unused1]"]
        + sentence_tokens[pred_idx + 1:]
    )

    inputs = tokenizer(
        marked_tokens,
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]  # [seq_len, num_labels]
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

    # Map back to original
    word_ids = tokenizer(
        marked_tokens,
        is_split_into_words=True,
        truncation=True,
    ).word_ids()

    # Collect first-subtoken prediction for each marked-up word
    marked_word_preds = {}
    for pos, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in marked_word_preds:
            marked_word_preds[word_idx] = pred_ids[pos]

    results = []
    for orig_i, token in enumerate(sentence_tokens):
        if orig_i == pred_idx:
            results.append((token, "V"))
        else:
            if orig_i < pred_idx:
                marked_i = orig_i
            else:
                marked_i = orig_i + 2 

            if marked_i in marked_word_preds:
                label_id = marked_word_preds[marked_i]
                results.append((token, id2label[int(label_id)]))
            else:
                results.append((token, "O"))

    return results
