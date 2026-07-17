import unittest

from scripts.compare_finetuned_models import build_comparison_report


class ComparisonReportTests(unittest.TestCase):
    def test_report_contains_signed_deltas_and_low_light_warning(self):
        old = {
            "test": {"loss": 0.20, "top1": 0.82},
            "low_light": {"loss": 0.40, "top1": 0.60},
        }
        new = {
            "test": {"loss": 0.18, "top1": 0.84},
            "low_light": {"loss": 0.20, "top1": 0.80},
        }

        report = build_comparison_report(old, new)

        self.assertIn("-0.020000", report)
        self.assertIn("+0.020000", report)
        self.assertIn("低照度60组数据已参与训练", report)
        self.assertIn("不能作为独立泛化指标", report)


if __name__ == "__main__":
    unittest.main()
