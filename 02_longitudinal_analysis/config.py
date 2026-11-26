from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_RAW = ROOT / "data_raw"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"

# Current clinical workbook structure.
CLINICAL_EXCEL = DATA_RAW / "同一患者至少两次检查有视频_含时长.xlsx"
CLINICAL_SHEET = "临床数据"

# Optional: one row per exam_id. If this file does not exist, the pipeline
# still runs with clinical variables only.
AUTO_FEATURES_CSV = REPO_ROOT / "01_feature_extraction" / "outputs" / "auto_features.csv"

DURATION_THRESHOLD = 60.0

# candidate: >=2 video-matched examinations and >=1 examination >=60 s
# strict:    >=2 examinations >=60 s
ANALYSIS_COHORT = "strict"  # "strict" or "candidate"

ZERO_TOL = 1e-12
MIN_RMCORR_PATIENTS = 3

MORPH_COLS = [
    "endometrium_thickness_mm",
    "endometrium_volume_ml",
]
FLOW_PRIMARY = ["vfi", "pi"]
FLOW_SECONDARY = ["vi", "fi", "ri", "sd"]
CLIN_MOTILITY = [
    "clinical_total_peristalsis_rate_min",
    "clinical_forward_rate_min",
    "clinical_backward_rate_min",
    "clinical_forward_ratio",
]
AUTO_PRIMARY = ["anterior_amplitude", "posterior_amplitude"]
AUTO_EXPLORATORY = [
    "r2_frequency_cpm",
    "anterior_er",
    "posterior_er",
    "mse",
    "m1_prop_rate_min_per_min",
]
QC_NUMERIC_COLS = ["video_duration_s", "tracking_valid_fraction"]

PRIMARY_ANALYSIS_COLS = (
    MORPH_COLS
    + FLOW_PRIMARY
    + ["clinical_total_peristalsis_rate_min"]
    + AUTO_PRIMARY
)
SECONDARY_ANALYSIS_COLS = FLOW_SECONDARY + [
    "clinical_forward_rate_min",
    "clinical_backward_rate_min",
    "clinical_forward_ratio",
]
EXPLORATORY_ANALYSIS_COLS = AUTO_EXPLORATORY
ALL_CONTINUOUS_COLS = list(
    dict.fromkeys(
        PRIMARY_ANALYSIS_COLS
        + SECONDARY_ANALYSIS_COLS
        + EXPLORATORY_ANALYSIS_COLS
    )
)

KEY_PAIRS = [
    ("endometrium_thickness_mm", "vfi"),
    ("endometrium_thickness_mm", "pi"),
    ("endometrium_volume_ml", "vfi"),
    ("vfi", "clinical_total_peristalsis_rate_min"),
    ("vfi", "anterior_amplitude"),
    ("vfi", "posterior_amplitude"),
    ("pi", "anterior_amplitude"),
    ("endometrium_thickness_mm", "anterior_amplitude"),
]

CLINICAL_COLUMN_ALIASES = {
    "exam_id": ["检查ID"],
    "patient_group_source": ["患者分组"],
    "source_exam_order": ["患者内检查序号"],
    "exam_datetime": ["检查时间"],
    "transfer_date": ["对应移植日期", "移植日期"],
    "cycle_id": ["移植周期ID"],
    "clinical_pregnancy_raw": ["临床妊娠"],
    "biochemical_pregnancy_raw": ["生化妊娠"],
    "endometrium_thickness_mm": ["子宫内膜厚度(mm)"],
    "endometrium_type": ["分型"],
    "clinical_forward_rate_min": ["蠕动｜正向"],
    "clinical_backward_rate_min": ["蠕动｜反向"],
    "endometrium_volume_ml": ["容积(ml)"],
    "vi": ["VI"],
    "fi": ["FI"],
    "vfi": ["VFI"],
    "sd": ["子宫内膜血流参数｜S/D"],
    "pi": ["子宫内膜血流参数｜PI"],
    "ri": ["子宫内膜血流参数｜RI"],
    "video_duration_s": ["匹配视频预计时长（秒）", "预计时长秒"],
    "treatment_mode": ["治疗方式"],
    "protocol": ["治疗方案"],
    "cycle_type": ["周期类型"],
    "embryo_info": ["胚胎情况"],
    "embryo_num": ["移植胚胎数"],
    "age": ["女方年龄", "年龄"],
    "bmi": ["女方BMI"],
    "cycle_link_confidence": ["周期关联可信度"],
    "cycle_audit_status": ["临床周期独立核对状态"],
}

AUTO_COLUMN_ALIASES = {
    "exam_id": ["exam_id", "检查ID", "record_id", "case_id"],
    "anterior_amplitude": [
        "anterior_amplitude",
        "amp_anterior",
        "anterior_amp",
    ],
    "posterior_amplitude": [
        "posterior_amplitude",
        "amp_posterior",
        "posterior_amp",
    ],
    "r2_frequency_cpm": [
        "r2_frequency_cpm",
        "R2_frequency_cpm",
        "R2",
    ],
    "m1_prop_rate_min_per_min": [
        "m1_prop_rate_min_per_min",
        "M1",
    ],
    "anterior_er": [
        "anterior_er",
        "anterior_ER_mean",
        "ER_anterior",
    ],
    "posterior_er": [
        "posterior_er",
        "posterior_ER_mean",
        "ER_posterior",
    ],
    "mse": [
        "mse",
        "MSE",
        "record_MSE_time_equal",
    ],
    "tracking_valid_fraction": [
        "tracking_valid_fraction",
        "valid_track_ratio",
        "valid_fraction",
    ],
    "video_qc": ["video_qc", "qc_flag", "QC"],
    "pose_source": ["pose_source"],
}

REVIEW_COLS = [
    "patient_id",
    "exam_id",
    "exam_order",
    "exam_datetime",
    "clinical_pregnancy",
    "days_to_transfer",
    "treatment_mode",
    "protocol",
    "cycle_type",
    "embryo_info",
    "embryo_num",
    "video_duration_s",
    "tracking_valid_fraction",
    "video_qc",
    "pose_source",
]

def ensure_output_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

def cohort_mask(df):
    if ANALYSIS_COHORT == "strict":
        return df["in_strict_set"].fillna(False) & df["long_video_eligible"].fillna(False)
    if ANALYSIS_COHORT == "candidate":
        return df["in_candidate_set"].fillna(False) & df["video_matched"].fillna(False)
    raise ValueError("ANALYSIS_COHORT must be 'strict' or 'candidate'")
