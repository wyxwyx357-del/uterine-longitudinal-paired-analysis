import pandas as pd

from src.cohort_selection import build_primary_cohort
from src.paired_change_analysis import consecutive_changes, summarize_delta
from src.pregnancy_discordant_pairs import discordant_patient_summary


def test_primary_cohort_keeps_all_long_repeated_exams():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "A", "A", "B", "B"],
            "exam_id": ["A1", "A2", "A3", "B1", "B2"],
            "exam_order": [1, 2, 3, 1, 2],
            "video_duration_s": [61, 70, 80, 61, 40],
        }
    )
    out = build_primary_cohort(df)
    assert set(out["patient_id"]) == {"A"}
    assert list(out["exam_id"]) == ["A1", "A2", "A3"]


def test_consecutive_changes_use_every_adjacent_exam():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "A", "A"],
            "exam_id": ["A1", "A2", "A3"],
            "exam_order": [1, 2, 3],
            "vfi": [2.0, 5.5, 4.0],
        }
    )
    changes = consecutive_changes(df, ["vfi"])
    assert len(changes) == 2
    assert changes.loc[0, "vfi__delta"] == 3.5
    assert changes.loc[1, "vfi__delta"] == -1.5

    summary = summarize_delta(changes, ["vfi"])
    assert summary.loc[0, "interval_n"] == 2
    assert summary.loc[0, "patient_n"] == 1
    assert summary.loc[0, "increase_n"] == 1
    assert summary.loc[0, "decrease_n"] == 1


def test_discordant_patient_uses_all_success_and_failure_exams():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "A", "A", "A"],
            "exam_id": ["A1", "A2", "A3", "A4"],
            "exam_order": [1, 2, 3, 4],
            "clinical_pregnancy": [0, 1, 0, 1],
            "vfi": [2.0, 5.0, 4.0, 9.0],
        }
    )
    out = discordant_patient_summary(df, ["vfi"])
    assert out.loc[0, "failure_exam_n"] == 2
    assert out.loc[0, "success_exam_n"] == 2
    # failure median = 3, success median = 7
    assert out.loc[0, "vfi__success_minus_failure"] == 4.0
