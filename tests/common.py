import gc
import unittest

class MindNLPTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        # release memory
        gc.collect()
