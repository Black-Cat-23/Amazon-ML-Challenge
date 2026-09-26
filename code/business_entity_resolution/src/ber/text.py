"""Text normalisation shared by every stage and every source (S1, S2, S3; train and test).

Vectorised polars expressions for the bulk work; small Python helpers only where logic is per-token.
"""
from __future__ import annotations

import re

import polars as pl

from .tables import LEET
from .translit import INDIC_RE, translit

TOKEN_RE = r"[a-z0-9]+"


def fold(e: pl.Expr) -> pl.Expr:
    """NFKD accent fold + lowercase + null-safe."""
    return e.fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()


def collapse_dots(e: pl.Expr) -> pl.Expr:
    """'l.l.c.' -> 'llc', 's.a.r.l' -> 'sarl', 'h.no' -> 'hno'; other dots become spaces."""
    for _ in range(3):
        e = e.str.replace_all(r"([a-z])\.([a-z])", "$1$2")
    return e.str.replace_all(r"\.", " ")


def base_clean(e: pl.Expr) -> pl.Expr:
    """fold -> collapse dots -> '&'/'+' -> 'and' -> literal 'none' address -> ''."""
    e = collapse_dots(fold(e)).str.replace_all(r"[&+]", " and ")
    return e.str.replace(r"^\s*none\s*$", "")


def tokens(e: pl.Expr, pattern: str = TOKEN_RE) -> pl.Expr:
    return base_clean(e).str.extract_all(pattern)


def translit_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Rule-based transliteration applied only to rows containing Indic script (fallback path)."""
    vals = df[col].to_list()
    out = [translit(v) if v and re.search(INDIC_RE, v) else v for v in vals]
    return df.with_columns(pl.Series(col, out))


_LEET_TOK = re.compile(r"^[a-z]*[0-9][a-z]*$")


def leet_fix_token(t: str) -> str:
    """'5mart' -> 'smart', 'si1ver' -> 'silver', 'gr0up' -> 'group'. Only name tokens with >=3 letters and one digit."""
    if _LEET_TOK.match(t) and sum(c.isalpha() for c in t) >= 3:
        return "".join(LEET.get(c, c) for c in t)
    return t


def skeleton(e: pl.Expr) -> pl.Expr:
    """Consonant skeleton of a lowercase ascii word (vectorised): 'private'/'praaivet' -> 'prvt'."""
    e = e.str.replace_all(r"c([eiy])", "s$1").str.replace_all("z", "s").str.replace_all("ksh", "x").str.replace_all("ph", "f")
    e = e.str.replace_all(r"([bcdgjkpstz])h", "$1").str.replace_all("ck", "k").str.replace_all("q", "k").str.replace_all("c", "k")
    e = e.str.replace_all("w", "v").str.replace_all(r"\B[aeiouy]", "")
    for ch in "bdfgjklmnprstvx":
        e = e.str.replace_all(ch + ch, ch)
    return e


def skeleton_py(w: str) -> str:
    """Python twin of `skeleton` (for small vocabularies)."""
    return pl.select(skeleton(pl.lit(w))).item()
