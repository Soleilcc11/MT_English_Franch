from __future__ import annotations

import html
import io
import json
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from .common import BOS, EOS, PAD, fingerprint, read_json, read_jsonl, sha256, write_json, write_jsonl

REPO = "IWSLT/iwslt2017"
ARCHIVE = "data/2017-01-trnted/texts/en/fr/en-fr.zip"
EXPECTED = {"train": 232825, "validation": 890, "test": 8597}


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "enfr-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(path)


def parse_archive(path):
    """Parse matching XML segment IDs; never silently zip unequal sentence lists."""
    with zipfile.ZipFile(path) as z:
        def member(name):
            hits = [n for n in z.namelist() if Path(n).name == name]
            if len(hits) != 1:
                raise ValueError(f"Expected one archive member {name}, got {hits}")
            return z.read(hits[0]).decode("utf-8-sig")

        def train_lines(lang):
            # str.splitlines also splits U+2028 inside French sentences, destroying alignment.
            return [html.unescape(line.strip()) for line in io.StringIO(member(f"train.tags.en-fr.{lang}"))
                    if not line.strip().startswith("<")]

        src, tgt = train_lines("en"), train_lines("fr")
        if len(src) != len(tgt):
            raise ValueError("Unaligned training files")
        result = {"train": [{"id": f"train:{i}", "year": None, "doc_id": None,
                              "src_raw": s, "tgt_raw": t} for i, (s, t) in enumerate(zip(src, tgt))]}

        def segments(name):
            root = ET.fromstring(member(name))
            pairs = {}
            for doc in root.iter("doc"):
                doc_id = doc.attrib.get("docid", doc.attrib.get("id"))
                if doc_id is None:
                    raise ValueError("Missing document ID")
                for seg in doc.iter("seg"):
                    key = (doc_id, seg.attrib["id"])
                    if key in pairs:
                        raise ValueError(f"Duplicate segment {key}")
                    pairs[key] = "".join(seg.itertext()).strip()
            return pairs

        for split, names in (("validation", ["dev2010"]),
                             ("test", [f"tst{year}" for year in range(2010, 2016)])):
            rows = []
            for name in names:
                en = segments(f"IWSLT17.TED.{name}.en-fr.en.xml")
                fr = segments(f"IWSLT17.TED.{name}.en-fr.fr.xml")
                if en.keys() != fr.keys():
                    raise ValueError(f"Unaligned segment IDs: {name}")
                for key in en:
                    rows.append({"id": f"{name}:{key[0]}:{key[1]}", "year": int(name[-4:]),
                                 "doc_id": key[0], "src_raw": en[key], "tgt_raw": fr[key]})
            result[split] = rows
    counts = {s: len(v) for s, v in result.items()}
    if counts != EXPECTED:
        raise ValueError(f"Unexpected raw counts {counts}; expected {EXPECTED}. Audit the archive before continuing.")
    return result


def clean_rows(raw):
    normalized = {}
    for split, rows in raw.items():
        normalized[split] = [{**row, "src": normalize(row["src_raw"]), "tgt": normalize(row["tgt_raw"])} for row in rows]
    heldout_sources = {r["src"] for s in ("validation", "test") for r in normalized[s]}
    seen, kept = set(), []
    counts = {"empty": 0, "duplicate_pair": 0, "heldout_source_overlap": 0}
    for row in normalized["train"]:
        pair = row["src"], row["tgt"]
        if not all(pair):
            counts["empty"] += 1
        elif pair in seen:
            counts["duplicate_pair"] += 1
        elif row["src"] in heldout_sources:
            counts["heldout_source_overlap"] += 1
        else:
            kept.append(row)
        seen.add(pair)
    normalized["train"] = kept
    return normalized, counts


def build_prepared(raw, dest, vocab_size=8000, max_train_tokens=128, provenance=None):
    import sentencepiece as spm
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "audit.json").exists():
        raise FileExistsError(f"Prepared data already exists: {dest}. Use another destination to preserve provenance.")
    rows, removed = clean_rows(raw)
    corpus = dest / "tokenizer_corpus.txt"
    with corpus.open("w", encoding="utf-8") as f:
        for row in rows["train"]:
            f.write(row["src"] + "\n" + row["tgt"] + "\n")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(dest / "tokenizer"),
        vocab_size=vocab_size, model_type="bpe", normalization_rule_name="identity",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, character_coverage=1.0,
        input_sentence_size=0, shuffle_input_sentence=False, num_threads=1,
        hard_vocab_limit=True, minloglevel=1)
    sp = spm.SentencePieceProcessor(model_file=str(dest / "tokenizer.model"))
    before_length = len(rows["train"])
    audit = {"provenance": provenance or {}, "raw_counts": {k: len(v) for k, v in raw.items()},
             "removed": removed, "vocab_size": sp.get_piece_size(), "max_train_tokens": max_train_tokens,
             "evaluation_protocol": "dev2010 / pooled tst2010-2015; case-sensitive raw references",
             "tokenizer_sha256": sha256(dest / "tokenizer.model"), "splits": {}}
    for split in rows:
        encoded = []
        for row in rows[split]:
            item = {**row, "src_ids": [BOS] + sp.encode(row["src"]) + [EOS],
                    "tgt_ids": [BOS] + sp.encode(row["tgt"]) + [EOS]}
            if split == "train" and max(len(item["src_ids"]), len(item["tgt_ids"])) > max_train_tokens:
                continue
            encoded.append(item)
        write_jsonl(dest / f"{split}.jsonl", encoded)
        lengths = [len(r["src_ids"]) - 2 for r in encoded]
        audit["splits"][split] = {"count": len(encoded), "sha256": sha256(dest / f"{split}.jsonl"),
            "src_token_length_quantiles": np.quantile(lengths, [0, .5, .9, .99, 1]).tolist() if lengths else []}
    audit["removed"]["length"] = before_length - audit["splits"]["train"]["count"]
    audit["length_filter_fraction"] = audit["removed"]["length"] / max(1, before_length)
    audit["data_id"] = fingerprint(audit)
    write_json(dest / "audit.json", audit)
    return audit


def prepare(cache, dest, archive=None, revision=None, vocab_size=8000, max_train_tokens=128):
    cache = Path(cache)
    if archive is None:
        if revision is None:
            with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60) as r:
                revision = json.load(r)["sha"]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{ARCHIVE}"
        archive = cache / "raw" / f"en-fr-{revision}.zip"
        if not archive.exists():
            fetch(url, archive)
    else:
        url = None
    provenance = {"dataset": REPO, "configuration": "iwslt2017-en-fr", "revision": revision,
                  "download_url": url, "archive_sha256": sha256(archive)}
    return build_prepared(parse_archive(archive), dest, vocab_size, max_train_tokens, provenance)


def collate(rows):
    def padded(field):
        out = np.full((len(rows), max(len(r[field]) for r in rows)), PAD, dtype=np.int64)
        for i, r in enumerate(rows):
            out[i, :len(r[field])] = r[field]
        return out
    src, target = padded("src_ids"), padded("tgt_ids")
    return src, target[:, :-1], target[:, 1:]


def batch_manifest(rows, folder, data_id, seed, epoch, batch_size, bucket_window=2048):
    """An epoch manifest is shared across frameworks and architecture ablations."""
    path = Path(folder) / f"seed{seed}-epoch{epoch}-batch{batch_size}.json"
    spec = {"data_id": data_id, "seed": seed, "epoch": epoch,
            "batch_size": batch_size, "bucket_window": bucket_window, "count": len(rows)}
    if path.exists():
        saved = read_json(path)
        if saved["spec"] != spec:
            raise ValueError(f"Manifest identity mismatch: {path}")
        return saved["batches"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(len(rows)).tolist()
    batches = []
    for start in range(0, len(order), bucket_window):
        window = sorted(order[start:start + bucket_window],
                        key=lambda i: max(len(rows[i]["src_ids"]), len(rows[i]["tgt_ids"])))
        batches.extend(window[j:j + batch_size] for j in range(0, len(window), batch_size))
    rng.shuffle(batches)
    write_json(path, {"spec": spec, "batches": batches,
                      "sample_ids": [[rows[i]["id"] for i in b] for b in batches]})
    return batches


def evaluation_batches(rows, batch_size=16):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["src_ids"]))
    batch = []
    for i in order:
        limit = 1 if max(len(rows[i]["src_ids"]), len(rows[i]["tgt_ids"])) > 128 else batch_size
        if batch and (len(batch) >= limit or len(batch) >= batch_size):
            yield batch
            batch = []
        batch.append(i)
        if limit == 1:
            yield batch
            batch = []
    if batch:
        yield batch


def verify_prepared(dest):
    dest = Path(dest)
    audit = read_json(dest / "audit.json")
    for split, info in audit["splits"].items():
        if sha256(dest / f"{split}.jsonl") != info["sha256"]:
            raise ValueError(f"Data checksum mismatch: {split}")
    if sha256(dest / "tokenizer.model") != audit["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum mismatch")
    return audit
