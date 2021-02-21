import sys
import unittest

sys.path.append("test")

# === unit tests ===
from test_transition import *
# ==================

# logging
if __name__ == "__main__":
    import logging
    logging.basicConfig(level = logging.WARNING)

# run unittests
if __name__ == "__main__":
    unittest.main()