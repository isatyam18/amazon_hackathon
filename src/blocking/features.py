"""
Feature "views" for candidate generation and their parallel hashed vectorization.

A view turns one record (Stage 1 `clean_name`, `clean_address`) into a bag of
string features. Every retrieval pass is a weighted combination of views
(see config.py), scored by TF-IDF cosine similarity.

Views:
    name_char   char n-grams of the name with spaces removed. Robust to typos, word
                order, and concatenation ('kaiagouldreliable.com' vs 'Kaia Gould Reliable').
    name_phon   char n-grams of the phonetic skeleton. Robust to transliteration
                and vowel noise ('shiv entarpraisas' vs 'Shiv Enterprises').
    name_word   whole name tokens (exact-token agreement, cheap and precise).
    name_pkey   phonetic key per name token.
    name_affix  4-char prefixes / suffixes of name tokens.
    name_compact first 8 chars of the space-free name (domain names, joined words).
    name_del    each name token plus its single-character deletions (1-edit typos).
    name_pair   unordered pairs of name tokens. The corpus reuses a small business
                vocabulary, so single words are common but word *pairs* identify a business.
    name_pkey_pair  unordered pairs of phonetic keys ('gold prodakts' ~ 'gold products').
    addr_word   address tokens incl. house numbers. Catches records whose name is
                a DBA/trade name or random noise but whose address matches.
    addr_num    numeric address tokens (house / plot / unit numbers).
    addr_bigram adjacent address token pairs ('10528_cedar', 'cedar_falls').
    addr_numstreet  house number + following word.
    name_addr   name token x address token cross keys. Survives heavy noise on either
                side as long as one name word and one address word agree.
"""

from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from src.blocking.normalize import address_tokens, name_tokens, phonetic_key, phonetic_name

VIEW_NAMES = (
    "name_char", "name_phon", "name_word", "name_pkey", "name_affix", "name_compact", "name_del",
    "name_pair", "name_pkey_pair", "addr_word", "addr_num", "addr_bigram", "addr_numstreet", "name_addr",
)


def _char_ngrams(text: str, n: int) -> List[str]:
    if not text:
        return []
    padded = f"^{text}$"
    if len(padded) <= n:
        return [padded]
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def _deletions(token: str) -> List[str]:
    """The token plus all its single-character deletions (1-edit typo tolerance)."""
    return [token] + [token[:i] + token[i + 1:] for i in range(len(token))]


def _pairs(tokens: List[str]) -> List[str]:
    """Unordered pairs of distinct tokens (word-order invariant, far rarer than single tokens)."""
    u = sorted(set(tokens))
    return [f"{a}|{b}" for i, a in enumerate(u) for b in u[i + 1:]]


def _view_features(view: str, ntoks: List[str], atoks: List[str], ngram: int) -> List[str]:
    if view == "name_char":
        return _char_ngrams("".join(ntoks), ngram)
    if view == "name_phon":
        return _char_ngrams(phonetic_name(ntoks).replace(" ", ""), ngram)
    if view == "name_word":
        return ntoks
    if view == "name_pkey":
        return [k for k in (phonetic_key(t) for t in ntoks) if len(k) >= 2]
    if view == "name_affix":
        return ([f"p:{t[:4]}" for t in ntoks if len(t) >= 4 and t.isalpha()]
                + [f"s:{t[-4:]}" for t in ntoks if len(t) >= 6 and t.isalpha()])
    if view == "name_compact":
        compact = "".join(ntoks)
        return [compact[:8]] if len(compact) >= 6 else []
    if view == "name_del":
        return [d for t in ntoks if len(t) >= 5 and t.isalpha() for d in _deletions(t)]
    if view == "name_pair":
        return _pairs(ntoks)
    if view == "name_pkey_pair":
        return _pairs([k for k in (phonetic_key(t) for t in ntoks) if len(k) >= 2])
    if view == "addr_word":
        return atoks
    if view == "addr_num":
        return [t for t in atoks if t.isdigit()]
    if view == "addr_bigram":
        return [f"{a}_{b}" for a, b in zip(atoks, atoks[1:])]
    if view == "name_addr":
        return [f"{n}|{a}" for n in set(ntoks) for a in set(atoks)]
    if view == "addr_numstreet":
        return [f"{a}_{b}" for a, b in zip(atoks, atoks[1:]) if a.isdigit() and b.isalpha()]
    raise ValueError(f"Unknown view '{view}'. Known views: {VIEW_NAMES}")


def record_features(name: str, address: str, view: str, ngram: int = 3) -> List[str]:
    """Feature strings of one record for one view."""
    return _view_features(view, name_tokens(name), address_tokens(address), ngram)


def _identity(x):
    return x


def _hash_chunk(
    names: Sequence[str], addresses: Sequence[str], views: Sequence[str], hash_bits: int, ngram: int
) -> Dict[str, sp.csr_matrix]:
    """Hash one chunk of records into binary term matrices, one per view."""
    vec = HashingVectorizer(
        analyzer=_identity,
        n_features=2 ** hash_bits,
        alternate_sign=False,
        norm=None,
        binary=True,
        dtype=np.float32,
    )
    ntoks = [name_tokens(n) for n in names]
    need_addr = any(v.startswith("addr_") for v in views)
    atoks = [address_tokens(a) for a in addresses] if need_addr else [[]] * len(names)
    out = {}
    for view in views:
        feats = [_view_features(view, nt, at, ngram) for nt, at in zip(ntoks, atoks)]
        out[view] = vec.transform(feats).tocsr()
    return out


def featurize(
    names: Sequence[str],
    addresses: Sequence[str],
    views: Sequence[str],
    hash_bits: int = 22,
    ngram: int = 3,
    n_jobs: int = 1,
    chunk_size: int = 100_000,
) -> Dict[str, sp.csr_matrix]:
    """Binary hashed term matrices (n_records x 2**hash_bits) for each requested view.

    Large inputs are split into chunks and hashed in worker processes; hashing
    is stateless, so chunk results are simply stacked back in order.
    """
    views = list(dict.fromkeys(views))
    names = list(names)
    addresses = list(addresses)
    n = len(names)
    if n == 0:
        return {v: sp.csr_matrix((0, 2 ** hash_bits), dtype=np.float32) for v in views}

    bounds = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    if n_jobs == 1 or len(bounds) == 1:
        parts = [_hash_chunk(names[a:b], addresses[a:b], views, hash_bits, ngram) for a, b in bounds]
    else:
        try:
            from joblib import Parallel, delayed

            with Parallel(n_jobs=n_jobs) as parallel:
                parts = parallel(
                    delayed(_hash_chunk)(names[a:b], addresses[a:b], views, hash_bits, ngram) for a, b in bounds
                )
        except Exception:
            parts = [_hash_chunk(names[a:b], addresses[a:b], views, hash_bits, ngram) for a, b in bounds]
    out = {}
    for v in views:  # stack view by view, releasing each chunk as soon as it is consumed
        out[v] = sp.vstack([p.pop(v) for p in parts], format="csr")
    return out
