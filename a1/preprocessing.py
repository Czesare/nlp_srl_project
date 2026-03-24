"""
Preprocessing module for Universal PropBank CoNLL-U files.

Handles parsing and sentence replication for multi-predicate sentences.
Each sentence is replicated once per predicate, so every training instance
has a single predicate with its corresponding argument labels.
"""

import re


def is_predicate_sense(value):
    """
    Checks if a value is a valid PropBank predicate sense.
    Valid: 'enjoy.01', 'make.LV', 'break_up.08'
    Invalid: '_', 'SpaceAfter=No', None
    """
    if not value or value == "_":
        return False
    return bool(re.match(r'^.+\.[A-Za-z0-9]+$', value))


def parse_conllu(file_path):
    """
    Parses a Universal PropBank CoNLL-U file.
    The CoNLL format has 10 fixed columns (id, form, lemma, upos, xpos,
    feats, head, deprel, deps, misc/predicate-sense). Columns 11+ contain
    argument labels, one column per predicate in the sentence.
    Returns:
        word_set: list of rows, one per token per predicate (V-labeled tokens skipped)
        stats: dict with pre/post replication counts
    word_set row format:
        [0] sent_idx        - sentence index (1-based)
        [1] token_id        - token id (1-based string from CoNLL)
        [2] word            - word form
        [3] predicate_word  - the predicate word this row belongs to
        [4] predicate_idx   - predicate position in sentence (0-based int)
        [5] full_tokens     - full sentence token list (shared reference)
        [6] label           - argument label (ARG0, ARG1, O, etc.)
    """
    word_set = []
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
                    _process_sentence(sent_idx, token_rows, word_set)
                    token_rows = []
            elif not line.startswith("#"):
                fields = line.strip("\n").split("\t")
                # Skip multiword tokens (1-2) and empty nodes (1.1)
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
            _process_sentence(sent_idx, token_rows, word_set)

    post_sentences = len(set((row[0], row[3], row[4]) for row in word_set))
    post_tokens = len(word_set)

    stats = {
        "pre_replication_sentences": pre_sentences,
        "pre_replication_tokens": pre_tokens,
        "post_replication_sentences": post_sentences,
        "post_replication_tokens": post_tokens,
    }
    return word_set, stats


def _process_sentence(sent_idx, token_rows, word_set):
    """
    Process one sentence: find predicates and build one row per token
    per predicate, skipping V-labeled (predicate) tokens.
    """
    if not token_rows:
        return

    n_cols = len(token_rows[0])
    n_predicates = max(0, n_cols - 11)

    if n_predicates == 0:
        return

    full_tokens = [row[1] for row in token_rows]

    # Find predicates: (column_offset, word, 0-based position)
    predicates = []
    col_offset = 0
    for row in token_rows:
        if len(row) > 10 and is_predicate_sense(row[10]):
            predicates.append((col_offset, row[1], int(row[0]) - 1))
            col_offset += 1

    if len(predicates) != n_predicates:
        return  # Malformed sentence, skip

    # Build rows: one per token per predicate
    for row in token_rows:
        token_id = row[0]
        word = row[1]

        for col_offset, pred_word, pred_pos in predicates:
            col_idx = 11 + col_offset
            label = row[col_idx] if col_idx < len(row) else "_"
            label = "O" if label == "_" else label

            if label == "V":    # CHANGE TO HARDCODED O according to feedback to label predicates as O   
                word_set.append([
                    sent_idx,       # [0] sentence index
                    token_id,       # [1] token id (1-based string)
                    word,           # [2] word form
                    pred_word,      # [3] predicate word
                    pred_pos,       # [4] predicate index (0-based)
                    full_tokens,    # [5] full sentence tokens
                    "O",          # [6] argument label
                ])               

            else:                           
                word_set.append([
                    sent_idx,       # [0] sentence index
                    token_id,       # [1] token id (1-based string)
                    word,           # [2] word form
                    pred_word,      # [3] predicate word
                    pred_pos,       # [4] predicate index (0-based)
                    full_tokens,    # [5] full sentence tokens
                    label,          # [6] argument label
                ])