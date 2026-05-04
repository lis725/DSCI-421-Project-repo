from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PATTERN_ROOT = PROJECT_ROOT / "pattern_res"
CKPT_DIR = PROJECT_ROOT / "ckpt"
SUBMIT_OUT = PROJECT_ROOT / "submit_out"
EVAL_OUT = PROJECT_ROOT / "eval_out"

BEST_FUSED_CKPT = CKPT_DIR / "best_fused_params.pt"

