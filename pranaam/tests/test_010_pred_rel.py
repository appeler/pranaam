#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for pranaam.py

"""

import unittest
import pandas as pd
from pranaam import pranaam


class TestPredRel(unittest.TestCase):
    def setUp(self):
        eng_names = [
            {"name": "Shah Rukh Khan", "true_rel": "muslim"},
            {"name": "Amitabh Bachchan", "true_rel": "not-muslim"},
        ]
        self.eng_df = pd.DataFrame(eng_names)

        hin_names = [
            {"name": "शाहरुख खान", "true_rel": "muslim"},
            {"name": "अमिताभ बच्चन", "true_rel": "not-muslim"},
        ]
        self.hin_df = pd.DataFrame(hin_names)

    def tearDown(self):
        pass

    def test_pred_label(self):
        odf = pranaam.pred_rel(self.eng_df["name"])
        self.assertIn("pred_label", odf.columns)
        self.assertTrue(odf.iloc[0]["pred_label"] == self.eng_df.iloc[0]["true_rel"])
        self.assertTrue(odf.iloc[1]["pred_label"] == self.eng_df.iloc[1]["true_rel"])

    def test_pred_prob_muslim(self):
        odf = pranaam.pred_rel(self.eng_df["name"])
        self.assertIn("pred_prob_muslim", odf.columns)

    def test_hindi(self):
        odf = pranaam.pred_rel(self.hin_df["name"], lang="hin")
        self.assertIn("pred_label", odf.columns)
        self.assertTrue(odf.iloc[0]["pred_label"] == self.hin_df.iloc[0]["true_rel"])
        self.assertTrue(odf.iloc[1]["pred_label"] == self.hin_df.iloc[1]["true_rel"])


if __name__ == "__main__":
    unittest.main()
