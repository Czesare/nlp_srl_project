"""
Feature extraction module for SRL.
Extracts exactly three features for token-level semantic role classification
with Logistic Regression:
Feature 1 (The mandatory one): Directed dependency path from token to predicate + predicate lemma.
Feature 2: Token lemma (the base form of the token being classified).
Feature 3: Named entity type of the token.
"""

import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_lg")


def get_spacy_doc(tokens):
    """
    Parse pre-tokenized tokens with SpaCy while preserving exact tokenization.
    Creates a Doc with our tokens then runs the full pipeline via nlp(doc).
    """
    doc = Doc(nlp.vocab, words=tokens)
    doc = nlp(doc)
    return doc


def _build_path(doc, token_idx, predicate_idx):
    """
    Build the directed dependency path from token to predicate through
    the dependency tree via the Lowest Common Ancestor (LCA).
    Direction markers:
        :up   = moving toward the head (token -> LCA)
        :down = moving toward a dependent (LCA -> predicate)
    Returns:
        (path_string, predicate_lemma) as separate strings.
    """
    pred_lemma = doc[predicate_idx].lemma_

    if token_idx == predicate_idx:
        return "SELF", pred_lemma

    token = doc[token_idx]
    predicate = doc[predicate_idx]

    tok_anc_ids = {t.i for t in token.ancestors}
    pred_anc_ids = {t.i for t in predicate.ancestors}

    # Find Lowest Common Ancestor
    lca = None
    if predicate.i in tok_anc_ids:
        lca = predicate
    elif token.i in pred_anc_ids:
        lca = token
    else:
        for a in predicate.ancestors:
            if a.i in tok_anc_ids:
                lca = a
                break

    if lca is None:
        return "NOPATH", pred_lemma

    # Upward path: token -> LCA
    up = []
    cur = token
    while cur.i != lca.i:
        up.append(cur.dep_ + ":up")
        cur = cur.head

    # Downward path: LCA -> predicate (collect pred->LCA, then reverse)
    down = []
    cur = predicate
    while cur.i != lca.i:
        down.append(cur.dep_ + ":down")
        cur = cur.head
    down.reverse()

    return ">".join(up + down), pred_lemma


def _get_token_lemma(doc, token_idx):
    """
    Feature 2: Lemma of the token being classified.

    Motivation for SRL:
        The lexical identity of a token is a strong signal for its semantic role,
        independent of its syntactic position. Person names and organization words
        (e.g. 'John', 'government') are semantically predisposed to agent roles
        (ARG0), while inanimate nouns tend toward patient roles (ARG1). Pronouns
        like 'it' or 'they' also carry strong role tendencies. Using the lemma
        rather than the raw form groups inflected variants together (e.g.
        'companies' and 'company' map to the same feature value), improving
        generalization.

    Suitable for Logistic Regression: each lemma becomes a sparse binary column
    via DictVectorizer. High-frequency lemmas provide reliable signal; rare ones
    fall back to the dependency path feature.
    """
    return doc[token_idx].lemma_


def _get_ner_type(doc, token_idx):
    """
    Feature 3: Named entity type of the token.

    Motivation for SRL:
        Named entity type provides semantic category information that pure syntax
        cannot capture. SpaCy's NER assigns types such as PERSON, ORG, GPE, LOC,
        DATE, and TIME. These correlate strongly with semantic roles:
            PERSON, ORG  -> likely ARG0 (animate agents)
            GPE, LOC     -> likely ARGM-LOC (locative modifier)
            DATE, TIME   -> likely ARGM-TMP (temporal modifier)
        Tokens that are not part of any named entity return 'NONE', which is
        itself informative (non-entities are less likely to be core arguments).

    Suitable for Logistic Regression: small fixed vocabulary of entity types
    (~20 SpaCy NER categories + NONE), producing clean one-hot columns.
    """
    return doc[token_idx].ent_type_ or "NONE"


def extract_features(doc, tokens, predicate_idx):
    """
    Extract all three features for every token in a sentence.

    The mandatory feature (dep path + predicate lemma) is represented via
    three dict keys for better generalization:
        dep_path_full: combined path|lemma string (the full mandatory feature)
        dep_path_only: just the syntactic path (generalizes across verbs)
        pred_lemma:    just the predicate lemma (captures verb-specific patterns)

    Args:
        doc:           SpaCy Doc (from get_spacy_doc)
        tokens:        list of token strings
        predicate_idx: 0-based index of the predicate
    Returns:
        list of feature dicts, one per token
    """
    feats = []
    for i in range(len(tokens)):
        path_str, lemma = _build_path(doc, i, predicate_idx)
        feats.append({
            # Feature 1: dependency path + predicate lemma (mandatory)
            "dep_path_full": path_str + "|" + lemma,
            "dep_path_only": path_str,
            "pred_lemma":    lemma,
            # Feature 2: token lemma
            "token_lemma":   _get_token_lemma(doc, i),
            # Feature 3: named entity type
            "ner_type":      _get_ner_type(doc, i),
        })
    return feats