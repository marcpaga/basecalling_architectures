import unittest
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation import count_signatures

class SignatureTest(unittest.TestCase):

    def test_matches(self):

        ref = ['A','C','T','A','G','C','T','A']
        ali = ['|','|','|','|','|','|','|','|']
        que = ['A','C','T','A','G','C','T','A']
        self.arr1 = np.array([ref, ali, que])

        signatures = count_signatures(self.arr1)
        self.assertEqual(signatures['AC>CT'], 1)
        self.assertEqual(signatures['CT>TA'], 2)
        self.assertEqual(signatures['TA>AG'], 1)
        self.assertEqual(signatures['AG>GC'], 1)
        self.assertEqual(signatures['GC>CT'], 1)
        t = 0
        for v in signatures.values():
            if v == 0:
                t += 1
        self.assertEqual(t, len(signatures) - 5)

    def test_mismatches(self):

        ref = ['A','C','T','A','G','C','T','A']
        ali = ['|','.','.','|','|','.','|','.']
        que = ['A','G','C','A','G','T','T','A']
        self.arr2 = np.array([ref, ali, que])

        signatures = count_signatures(self.arr2)
        self.assertEqual(signatures['AC>GT'], 1)
        self.assertEqual(signatures['CT>CA'], 1)
        self.assertEqual(signatures['TA>AG'], 1)
        self.assertEqual(signatures['AG>GC'], 1)
        self.assertEqual(signatures['GC>TT'], 1)
        self.assertEqual(signatures['CT>TA'], 1)
        t = 0
        for v in signatures.values():
            if v == 0:
                t += 1
        self.assertEqual(t, len(signatures) - 6)