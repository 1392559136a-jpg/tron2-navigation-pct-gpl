import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from select_processing_profile import (  # noqa: E402
    choose_profile,
    safe_standard_budget,
)


class ProcessingProfileSelectorTest(unittest.TestCase):
    def test_standard_is_kept_with_headroom_inside_budget(self):
        self.assertEqual(choose_profile(600, 700, headroom=1.1), 'standard')

    def test_large_profile_is_selected_before_standard_oom(self):
        self.assertEqual(choose_profile(700, 700, headroom=1.1), 'nx-large')

    def test_budget_uses_most_conservative_limit(self):
        self.assertEqual(
            safe_standard_budget(1000, 600, 800),
            540,
        )


if __name__ == '__main__':
    unittest.main()