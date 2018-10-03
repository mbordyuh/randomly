#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `randomly` package."""

import pytest

from randomly import randomly


def test_loading_data():
    import pandas as pd
    import pkg_resources
    data_path = pkg_resources.resource_filename('randomly', 'misc/data/data.tsv')
    df = pd.read_table(data_path, sep='\t', index_col=0)

    assert df.shape == (300, 8570)