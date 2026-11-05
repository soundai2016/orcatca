"""Regression checks for two easy-to-miss scientific errors in archived aggregates."""
from pathlib import Path
import csv
import importlib.util
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('audit_results', ROOT/'scripts/audit_results.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
RUN = ROOT/'result/analysis/transfer_review'


class ResultAuditTests(unittest.TestCase):
    def altered_table(self, name, predicate, field, value):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        target = Path(td.name)/'analysis'
        shutil.copytree(RUN, target)
        path = target/(name+'.csv')
        with path.open(newline='') as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fields = reader.fieldnames
        next(r for r in rows if predicate(r))[field] = value
        with path.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return target

    def test_different_cutoff_with_unchanged_counts_is_not_a_matched_comparison(self):
        target = self.altered_table('crossfit_operating', lambda r:r['role']=='target',
                                    'threshold','0.00001')
        with self.assertRaisesRegex(ValueError,'matched cutoff'):
            AUDIT.audit(target)

    def test_missing_dfo_gap_cannot_be_encoded_as_zero(self):
        target = self.altered_table('transfer_gap_summary', lambda r:r['fold']=='fold01',
                                    'transfer_gap','0')
        with self.assertRaisesRegex(ValueError,'DFO is not a zero-gap estimate'):
            AUDIT.audit(target)


if __name__=='__main__':
    unittest.main()
