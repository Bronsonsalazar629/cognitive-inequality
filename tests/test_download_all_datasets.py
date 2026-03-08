# tests/test_download_all_datasets.py
"""Tests for dataset download orchestrator."""
import pytest


def test_download_addhealth_listed():
    """Add Health should appear in dataset registry with manual download instructions."""
    from src.data.download_all_datasets import DATASET_REGISTRY

    assert 'addhealth' in DATASET_REGISTRY
    entry = DATASET_REGISTRY['addhealth']
    assert entry['status'] == 'manual'
    assert 'ICPSR' in entry['instructions']


def test_download_piaac_listed():
    """PIAAC should appear in dataset registry."""
    from src.data.download_all_datasets import DATASET_REGISTRY

    assert 'piaac' in DATASET_REGISTRY
    entry = DATASET_REGISTRY['piaac']
    assert entry['status'] == 'manual'
    assert 'PIAAC' in entry['instructions'] or 'OECD' in entry['instructions']


def test_download_nsduh_listed():
    """NSDUH should appear in dataset registry."""
    from src.data.download_all_datasets import DATASET_REGISTRY

    assert 'nsduh' in DATASET_REGISTRY
    entry = DATASET_REGISTRY['nsduh']
    assert entry['status'] == 'manual'
    assert 'NSDUH' in entry['instructions'] or 'SAMHSA' in entry['instructions']
