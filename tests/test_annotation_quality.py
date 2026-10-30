"""Tests for the human annotation agreement and adjudication helpers."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.annotation_quality import load_annotations, summarize_agreement


class AnnotationQualityTests(unittest.TestCase):
    def _write(self, path, rows):
        path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")

    def test_agreement_perfect(self):
        a=[{"source_id":"q1","scope":"FRAME","tier":"L1_DIRECT"},{"source_id":"q2","scope":"CLIP","tier":"L2_RETRIEVAL"},{"source_id":"q3","scope":"WHOLE_VIDEO","tier":"L3_GRAPH"}]
        b=[dict(x) for x in a]
        report=summarize_agreement(a,b)
        self.assertEqual(report["scope"]["cohen_kappa"],1.0)
        self.assertEqual(report["tier"]["cohen_kappa"],1.0)

    def test_load_rejects_unlabelled_or_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"ann.jsonl"
            self._write(p,[{"source_id":"q1","scope":"","tier":"L1_DIRECT"}])
            with self.assertRaises(ValueError):
                load_annotations(p)

    def test_load_duplicate_source_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"ann.jsonl"
            row={"source_id":"q1","scope":"FRAME","tier":"L1_DIRECT"}
            self._write(p,[row,row])
            with self.assertRaisesRegex(ValueError,"duplicate"):
                load_annotations(p)


if __name__=="__main__":
    unittest.main()
