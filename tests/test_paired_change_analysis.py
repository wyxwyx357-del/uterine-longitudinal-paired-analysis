import pandas as pd

from src.cohort_selection import build_primary_cohort
from src.paired_change_analysis import paired_wide, summarize_delta


def test_primary_cohort_requires_two_long_exams():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "B"],
            "exam_id": ["A1", "A2", "B1", "B2"],
            "pair_order": [1, 2, 1, 2],
            "video_duration_s": [61, 70, 61, 40],
        }
    )
    out = build_primary_cohort(df)
    assert set(out["patient_id"]) == {"A"}
    assert len(out) == 2


def test_delta_is_exam2_minus_exam1():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "A"],
            "pair_order": [1, 2],
            "vfi": [2.0, 5.5],
        }
    )
    paired = paired_wide(df, ["vfi"])
    assert paired.loc[0, "vfi__delta"] == 3.5

    summary = summarize_delta(paired, ["vfi"])
    assert summary.loc[0, "increase_n"] == 1
    assert summary.loc[0, "decrease_n"] == 0
