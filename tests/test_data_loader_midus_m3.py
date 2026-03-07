import pytest
from pathlib import Path

SURVEY_SAV = Path('M3_P1_SURVEY_N3294_20251029.sav')
BTACT_SAV  = Path('M3_P3_BTACT_N3291_20210922.sav')

@pytest.mark.skipif(not SURVEY_SAV.exists(), reason="M3 SAV not present")
def test_load_midus_m3_shape():
    from src.data.data_loader_midus_m3 import load_midus_m3
    df = load_midus_m3()
    assert 'M2ID' in df.columns
    assert 'C1PRAGE' in df.columns
    assert 'ses_index_m3' in df.columns
    assert 'cognitive_score_m3' in df.columns
    assert len(df) > 500

@pytest.mark.skipif(not SURVEY_SAV.exists(), reason="M3 SAV not present")
def test_load_midus_m3_no_allnan():
    from src.data.data_loader_midus_m3 import load_midus_m3
    df = load_midus_m3()
    assert df['cognitive_score_m3'].notna().sum() > 400
    assert df['ses_index_m3'].notna().sum() > 400
