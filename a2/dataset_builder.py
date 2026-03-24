"""
Dataset builder for BERT-based SRL.

Converts preprocessed sentence-predicate records into a HuggingFace Dataset
with proper subword tokenization and label alignment.
"""

from datasets import Dataset, Features, Sequence, Value
from preprocessing_bert import insert_predicate_markers


def build_hf_dataset(records, tokenizer, label2id):
    """
    Convert preprocessed records into a tokenized HuggingFace Dataset.
    Steps per record:
        1. Insert predicate markers around the predicate token
        2. Store the marked-up tokens and integer labels
    The actual subword tokenization + label alignment happens in
    tokenize_and_align_labels(), applied via dataset.map().
    Args:
        records:   list of dicts from parse_conllu()
        tokenizer: HuggingFace tokenizer
        label2id:  dict mapping label strings to integer IDs
    Returns:
        HuggingFace Dataset ready for tokenization
    """
    all_tokens = []
    all_labels = []
    all_sent_ids = []

    for rec in records:
        tokens, labels, _ = insert_predicate_markers(
            rec["tokens"], rec["labels"], rec["predicate_idx"]
        )

        # Convert string labels to integer IDs
        # None (markers) and "V" (predicate) -> -100
        int_labels = []
        for label in labels:
            if label is None:
                int_labels.append(-100)
            elif label == "V":
                int_labels.append(label2id["O"])  # train as O, but we assign V in post-processing
            else:
                int_labels.append(label2id[label])

        all_tokens.append(tokens)
        all_labels.append(int_labels)
        all_sent_ids.append(rec["sent_id"])

    dataset = Dataset.from_dict({
        "tokens": all_tokens,
        "labels": all_labels,
        "sent_id": all_sent_ids,
    })

    return dataset


def get_tokenize_and_align_fn(tokenizer):
    """
    Returns a tokenize_and_align_labels function bound to the given tokenizer.
    This function:
    - Subword-tokenizes pre-tokenized input (with markers already inserted)
    - Aligns labels: first subtoken of each word gets the word's label,
      subsequent subtokens get -100
    - Special tokens ([CLS], [SEP]) get -100
    """

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        aligned_labels = []
        for i, word_labels in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens ([CLS], [SEP], padding)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First subtoken of a word -> use the word's label
                    label_ids.append(word_labels[word_idx])
                else:
                    # Subsequent subtokens of same word -> mask
                    label_ids.append(-100)
                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    return tokenize_and_align_labels
