"""
Preprocessing module for BERT-based SRL.

Adapted A1's CoNLL-U parsing to produce sentence-level records suitable
for BERT/transformer models. Each record represents one sentence-predicate pair
containing the full token list, labels, and predicate position.
"""

import re
from collections import defaultdict


def is_predicate_sense(value):
    """
    Checks if a value is a valid PropBank predicate sense.
    """
    if not value or value == "_":
        return False
    return bool(re.match(r'^.+\.[A-Za-z0-9]+$', value))


def parse_conllu(file_path):
    """
    Parse a Universal PropBank CoNLL-U file into sentence-predicate records.
    Each record is a dict:
        {
            "sent_id":       int,     # original sentence index 
            "tokens":        [str],   # full sentence tokens
            "labels":        [str],   # SRL labels per token 
                                      # predicate token "V" 
            "predicate_idx": int,     # 0-based index of the predicate
        }
    Returns:
        records: list of dicts (one per sentence-predicate pair)
        stats:   dict with dataset statistics
    """
    records = []
    pre_sentences = 0
    pre_tokens = 0

    with open(file_path, "r", encoding="utf-8") as f:
        sent_idx = 0
        token_rows = []

        for line in f:
            if line.startswith("# sent_id"):
                sent_idx += 1
            elif line.strip() == "":
                if token_rows:
                    pre_sentences += 1
                    pre_tokens += len(token_rows)
                    _process_sentence(sent_idx, token_rows, records)
                    token_rows = []
            elif not line.startswith("#"):
                fields = line.strip("\n").split("\t")
                # Skip multiword tokens 
                if "-" in fields[0] or "." in fields[0]:
                    continue
                # Strip trailing empty fields
                while fields and fields[-1].strip() == "":
                    fields.pop()
                token_rows.append(fields)
        # Handle last sentence if file doesn't end with blank line
        if token_rows:
            pre_sentences += 1
            pre_tokens += len(token_rows)
            _process_sentence(sent_idx, token_rows, records)
    # Post-replication stats
    post_sentences = len(records)
    # Count tokens excluding V-labeled (predicate) tokens
    post_tokens = sum(
        sum(1 for label in rec["labels"] if label != "V")
        for rec in records
    )
    stats = {
        "pre_replication_sentences": pre_sentences,
        "pre_replication_tokens": pre_tokens,
        "post_replication_sentences": post_sentences,
        "post_replication_tokens": post_tokens,
    }
    return records, stats


def _process_sentence(sent_idx, token_rows, records):
    """
    Process one sentence: find predicates and build one record per predicate.
    Each record contains the full token list with corresponding SRL labels.
    The predicate token is labeled "V" (will be masked to -100 during training).
    """
    if not token_rows:
        return

    n_cols = len(token_rows[0])
    n_predicates = max(0, n_cols - 11)

    if n_predicates == 0:
        return

    tokens = [row[1] for row in token_rows]

    # Identify predicates
    predicates = []
    col_offset = 0
    for row in token_rows:
        if len(row) > 10 and is_predicate_sense(row[10]):
            predicates.append((col_offset, int(row[0]) - 1))
            col_offset += 1

    if len(predicates) != n_predicates:
        return  # Malformed sentence, skip

    # Build one record per predicate
    for col_offset, pred_pos in predicates:
        labels = []
        for row in token_rows:
            col_idx = 11 + col_offset
            label = row[col_idx] if col_idx < len(row) else "_"
            label = "O" if label == "_" else label
            labels.append(label)

        records.append({
            "sent_id": sent_idx,
            "tokens": tokens,       
            "labels": labels,
            "predicate_idx": pred_pos,
        })


def get_label_vocabulary(*record_sets):
    """
    Build a sorted label list from one or more record sets.
    Ensures "O" is always index 0 for consistency.
    Excludes "V" since it's masked during training.
    Args:
        *record_sets: one or more lists of records 
    Returns:
        label_list: sorted list of unique labels (O first)
        label2id:   dict mapping label string -> int
        id2label:   dict mapping int -> label string
    """
    all_labels = set()
    for records in record_sets:
        for rec in records:
            for label in rec["labels"]:
                # CHANGE TO HARDCODED O according to feedback to label predicates as O       
                if label != "V":
                    all_labels.add(label)

    # O first, then sorted
    all_labels.discard("O")
    label_list = ["O"] + sorted(all_labels)

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    return label_list, label2id, id2label


def insert_predicate_markers(tokens, labels, predicate_idx,
                              start_marker="[unused0]",
                              end_marker="[unused1]"):
    """
    Insert predicate marker tokens around the predicate.
    Returns:
        new_tokens:        list of strings with markers inserted
        new_labels:        list with None at marker positions
        new_predicate_idx: updated predicate index (shifted by 1)
    """
    new_tokens = (
        tokens[:predicate_idx]
        + [start_marker]
        + [tokens[predicate_idx]]
        + [end_marker]
        + tokens[predicate_idx + 1:]
    )

    new_labels = (
        labels[:predicate_idx]
        + [None]                    # start marker -> will become -100
        + [labels[predicate_idx]]   # predicate label (V -> also -100)
        + [None]                    # end marker -> will become -100
        + labels[predicate_idx + 1:]
    )

    new_predicate_idx = predicate_idx + 1  # shifted by the start marker

    return new_tokens, new_labels, new_predicate_idx