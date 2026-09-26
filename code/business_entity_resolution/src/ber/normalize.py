"""Record normaliser: applies the learned artefacts (artifacts.py) + small hand tables (tables.py) to ANY record,
so S1, S2, S3, train and test all go through exactly the same function.

Output columns (per record):
  entity_id, country
  nw     name tokens (Indic mapped via lexicon/siblings, folded, dots collapsed, leet-fixed, web handle segmented)
  core   nw minus legal forms and learned noise words (the "brand" tokens)
  alias  tokens of the other side of 'X f/k/a Y' (empty if none)
  legal  legal-form tokens found in the name
  decoy  learned decoy/qualifier tokens present in the name
  sk     consonant skeletons of core tokens
  aw     address words (street types mapped, function words dropped, >=2 letters, synonyms expanded) ; ad  numbers
  au     unit codes (c502, e6)  ; noaddr  address missing
"""
from __future__ import annotations

import math
import os
import re

import polars as pl

from .tables import (ADDR_STOP, ALIAS_MARKERS, HONORIFIC, LEGAL, NAME_STOP, NUMBER_WORDS, PLACE_ADDR_STOP, PLACE_ALIAS,
                     PLACE_PHRASE, PLACE_SYN_BLOCK, STREET, STREET_COUNTRY, UNIT_EXCLUDE)
from .text import base_clean, leet_fix_token, skeleton
from .translit import INDIC_RE, translit

TOK_ANY = r"[^\s.,()\[\]\-/]+"
WEB_RE = r"(?i)(^@|www\.|\.(com|in|net|org|co|fr|biz|info)\b)"
ALIAS_RE = r"(?i)\s(" + "|".join(m.replace("/", r"\s*/\s*") for m in ALIAS_MARKERS) + r")[:\s]"
# region-specific place handling (tables.PLACE_*): city aliases, multi-word states as one token, no synonym chaining
# through shared state words, generator filler words dropped. Off by default (models trained without it stay valid).
PLACES = os.environ.get("BER_PLACES", "0") == "1"


class Normalizer:
    def __init__(self, art_dir: str):
        rd = lambda f: pl.read_parquet(os.path.join(art_dir, f))
        lex = rd("indic_lexicon.parquet").select("indic", "latin")
        oov = rd("oov_map.parquet").select("indic", "latin") if os.path.exists(os.path.join(art_dir, "oov_map.parquet")) else lex.head(0)
        self.indic = dict(zip(*pl.concat([oov, lex]).unique("indic", keep="last").to_dict(as_series=False).values()))
        # Name-only extra-word stats are dominated by coincidental same-name businesses country-wide (exact-equal names
        # are the same business only 8.6% of the time), so noise words = small hand list; decoy words are learned later
        # from the address-conditioned blocking candidate pool (decoy_words.parquet, optional).
        self.noise = set(HONORIFIC) | NAME_STOP
        dp = os.path.join(art_dir, "decoy_words.parquet")
        self.decoy = set(pl.read_parquet(dp)["t"].to_list()) if os.path.exists(dp) else set()
        syn = rd("addr_synonyms.parquet")
        blk = list(PLACE_SYN_BLOCK)
        if PLACES:
            syn = syn.filter(~pl.col("token").is_in(blk) & ~pl.col("equiv").is_in(blk))
        self.addr_syn = syn.group_by("token").agg(pl.col("equiv"))
        # per-country synonyms mined at test time from pseudo-pairs (artifacts.fit_country_addr_synonyms), optional
        cp = os.path.join(art_dir, "addr_synonyms_country.parquet")
        sc = pl.read_parquet(cp) if os.path.exists(cp) else None
        if sc is not None and PLACES:
            sc = sc.filter(~pl.col("token").is_in(blk) & ~pl.col("equiv").is_in(blk))
        self.addr_syn_c = sc.group_by(["country", "token"]).agg(pl.col("equiv")) if sc is not None else None
        v = rd("vocab.parquet").group_by(["country", "token"]).agg(pl.col("n").sum())
        self.vocab = {}
        for c, g in v.partition_by("country", as_dict=True).items():
            tot = g["n"].sum()
            self.vocab[c[0] if isinstance(c, tuple) else c] = {t: math.log(n / tot) for t, n in zip(g["token"], g["n"]) if len(t) >= 2}

    # ---- per-row python helpers (only applied to the rows that need them) --------------------------------------
    def _indic_name(self, s: str) -> str:
        s = s or ""
        return " ".join(self.indic.get(w, translit(w)) if re.search(INDIC_RE, w) else w for w in re.findall(TOK_ANY, s))

    def _segment(self, h: str, country: str) -> str:
        """Split a concatenated handle ('monamanufacturing') into vocabulary words (Viterbi, unigram log-probs)."""
        lp = self.vocab.get(country) or {}
        n = len(h)
        if n == 0 or not lp:
            return h
        best = [0.0] + [-1e18] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - 20), i):
                w = h[j:i]
                sc = best[j] + lp.get(w, -30.0 - 5 * len(w))
                if sc > best[i]:
                    best[i], back[i] = sc, j
        out, i = [], n
        while i > 0:
            out.append(h[back[i]:i]); i = back[i]
        return " ".join(reversed(out))

    def _web(self, name: str, country: str) -> str:
        h = re.sub(r"(?i)^@|www\.|\.(com|in|net|org|co|fr|biz|info)\b.*$|#\s*\d+", " ", name)
        parts = [self._segment(p, country) if len(p) >= 8 else p for p in re.findall(r"[a-z]+", h.lower())]
        return " ".join(parts)

    # ---- main --------------------------------------------------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        d = df.select("entity_id", "country", pl.col("business_name").fill_null("").alias("n0"),
                      pl.col("business_address").fill_null("").alias("a0"))
        # Indic names: learned lexicon (+ test sibling/OOV map), translit fallback; Indic address phrases: translit
        n0_vals = d["n0"].to_list()
        n0_out = [self._indic_name(v) if v and re.search(INDIC_RE, v) else v for v in n0_vals]
        a0_vals = d["a0"].to_list()
        a0_out = [translit(v) if v and re.search(INDIC_RE, v) else v for v in a0_vals]
        d = d.with_columns(pl.Series("n0", n0_out), pl.Series("a0", a0_out))
        # web handles -> segmented words
        n0_vals = d["n0"].to_list()
        ctry_vals = d["country"].to_list()
        n0_web = [self._web(n or "", c or "") if n and re.search(WEB_RE, n) else n for n, c in zip(n0_vals, ctry_vals)]
        d = d.with_columns(pl.Series("n0", n0_web))
        # alias split: 'X f/k/a Y' -> main Y (right side), alias X
        parts = pl.col("n0").str.replace(ALIAS_RE, "\x00").str.split("\x00")
        d = d.with_columns(pl.when(parts.list.len() == 2).then(parts.list.get(1, null_on_oob=True)).otherwise(pl.col("n0")).alias("nmain"),
                           pl.when(parts.list.len() == 2).then(parts.list.get(0, null_on_oob=True)).otherwise(pl.lit("")).alias("nalias"))
        tok = lambda e: base_clean(e).str.replace_all(r"#\s*\d+|\(id:?\s*\d+\)", " ").str.extract_all(r"[a-z0-9]+")
        d = d.with_columns(tok(pl.col("nmain")).alias("nw"), tok(pl.col("nalias")).alias("alias"))
        # leet fix, only on rows whose name mixes letters and digits inside a word
        nmain_lower = d["nmain"].str.to_lowercase().to_list()
        nw_vals = d["nw"].to_list()
        nw_fixed = []
        leet_check = re.compile(r"[a-z][0-9]|[0-9][a-z]")
        for nm, l in zip(nmain_lower, nw_vals):
            if nm and leet_check.search(nm) and l:
                nw_fixed.append([leet_fix_token(t) for t in l])
            else:
                nw_fixed.append(l if l is not None else [])
        d = d.with_columns(pl.Series("nw", nw_fixed).list.unique(maintain_order=True))
        legal, noise, decoy = list(LEGAL), list(self.noise), list(self.decoy)
        d = d.with_columns(
            pl.col("nw").list.eval(pl.element().filter(~pl.element().is_in(legal) & ~pl.element().is_in(noise))).alias("core"),
            pl.col("nw").list.eval(pl.element().filter(pl.element().is_in(legal))).alias("legal"),
            pl.col("nw").list.eval(pl.element().filter(pl.element().is_in(decoy))).alias("decoy"))
        d = d.with_columns(pl.col("core").list.eval(skeleton(pl.element())).list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).alias("sk"))
        # address
        a = base_clean(pl.col("a0"))
        num_words = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
        a = a.str.replace_all(rf"\b({num_words})\b", "#$1#")
        for w, dgt in NUMBER_WORDS.items():
            a = a.str.replace_all(f"#{w}#", dgt)
        a = a.str.replace_all(r"\b(\d+)(st|nd|rd|th)\b", "$1")  # ordinals, incl. generator typos '7nd'
        ux = list(UNIT_EXCLUDE)
        d = d.with_columns(
            a.str.extract_all(r"[a-z]+").alias("aw0"),
            a.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.element().filter(pl.element() != "")).alias("ad"),
            a.str.extract_all(r"\b[a-z]{1,3}[-/ ]?\d+[a-z]?\b").list.eval(pl.element().str.replace_all(r"[-/ ]", "").str.replace(r"^([a-z]+)0+", "$1"))
             .list.eval(pl.element().filter(~pl.element().str.extract(r"^([a-z]+)").is_in(ux))).alias("au"),
            (pl.col("a0").str.strip_chars() == "").alias("noaddr"))
        # street abbreviations (hand table, per-country overrides) BEFORE the length filter so 'R', 'St', 'Bd' map;
        # 2-letter tokens kept (state codes, so the learned code<->name synonyms apply on both sides)
        stop = list(ADDR_STOP | (PLACE_ADDR_STOP if PLACES else set()))
        aw = pl.col("aw0").list.eval(pl.element().replace(STREET))
        for c, ov in STREET_COUNTRY.items():
            aw = pl.when(pl.col("country") == c).then(pl.col("aw0").list.eval(pl.element().replace({**STREET, **ov}))).otherwise(aw)
        d = d.with_columns(aw.list.eval(pl.element().filter((pl.element().str.len_chars() >= 2) & ~pl.element().is_in(stop))).alias("aw0"))
        if PLACES:
            # flat expressions (a nested when/otherwise chain doubles the expression tree per rule)
            parts = [pl.col("aw0")]
            for c, tab in PLACE_ALIAS.items():
                parts.append(pl.when(pl.col("country") == c).then(pl.col("aw0").list.eval(pl.element().replace_strict(tab, default=None)).list.drop_nulls()))
            hits = [pl.when(pl.all_horizontal([pl.col("aw0").list.contains(w) for w in words]) & (pl.col("country") == c)).then(pl.lit(canon))
                    for c, phrases in PLACE_PHRASE.items() for words, canon in phrases]
            parts.append(pl.concat_list(hits).list.drop_nulls())
            d = d.with_columns(pl.concat_list([x.fill_null(pl.lit([], dtype=pl.List(pl.String))) for x in parts]).list.unique(maintain_order=True).alias("aw0"))
        ex = d.select("entity_id", "country", "aw0").explode("aw0").join(self.addr_syn.rename({"token": "aw0"}), on="aw0", how="left")
        empty = pl.lit([], dtype=pl.List(pl.String))
        if self.addr_syn_c is not None:
            ex = ex.join(self.addr_syn_c.rename({"token": "aw0", "equiv": "equiv_c"}), on=["country", "aw0"], how="left") \
                   .with_columns(pl.concat_list(pl.col("equiv").fill_null(empty), pl.col("equiv_c").fill_null(empty)).alias("equiv"))
        ex = ex.with_columns(pl.concat_list(pl.col("aw0"), pl.col("equiv").fill_null(empty)).alias("x")) \
               .group_by("entity_id", maintain_order=True).agg(pl.col("x").flatten().unique().alias("aw"))
        d = d.join(ex, on="entity_id", how="left").with_columns(pl.col("aw").list.drop_nulls())
        return d.select("entity_id", "country", "nw", "core", "alias", "legal", "decoy", "sk", "aw", "ad", "au", "noaddr")
