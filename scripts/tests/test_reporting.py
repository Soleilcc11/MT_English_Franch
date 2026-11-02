import tempfile
import unittest
from pathlib import Path

from enfr.common import read_json, read_jsonl, write_jsonl
from enfr.reporting import bootstrap, error_review


class ReportingTests(unittest.TestCase):
    def row(self, i, bucket="01_<=16"):
        return {"id": str(i), "source": "The red car.", "reference": "La voiture rouge.",
                "hypothesis": "La voiture rouge.", "length_bucket": bucket, "truncated": False}

    def test_identical_paired_bootstrap_has_zero_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pred.jsonl"
            write_jsonl(p, [self.row(i) for i in range(10)])
            r = bootstrap(p, p, Path(tmp) / "bootstrap.json", repetitions=20)
            self.assertEqual(r["delta"], 0)
            self.assertEqual(r["delta_ci95"], [0, 0])

    def test_bootstrap_rejects_unpaired_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            write_jsonl(a, [self.row(1)]); write_jsonl(b, [self.row(2)])
            with self.assertRaises(ValueError):
                bootstrap(a, b, Path(tmp) / "out.json")

    def test_error_review_fills_small_stratum_without_duplicates_or_fake_labels(self):
        buckets = ("01_<=16", "02_17-32", "03_33-64", "04_65-128")
        rows = [self.row(30*b + i, bucket) for b, bucket in enumerate(buckets) for i in range(30)]
        rows.append(self.row(120, "05_>128"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.jsonl"; output = Path(tmp) / "review.jsonl"
            write_jsonl(path, rows)
            selected = error_review({"A0": path, "A1": path, "A2": path}, output)
            self.assertEqual(len(selected), 100)
            self.assertEqual(len({x["id"] for x in selected}), 100)
            self.assertEqual(sum(x["length_bucket"] == "05_>128" for x in selected), 1)
            self.assertTrue(all(x["review_status"] == "unreviewed" for x in selected))
            self.assertEqual(read_json(output.with_suffix(".sampling.json"))["shortages"]["05_>128"], 19)
            with self.assertRaises(FileExistsError):
                error_review({"A0": path}, output)


if __name__ == "__main__":
    unittest.main()
