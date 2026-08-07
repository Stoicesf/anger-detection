"""One-shot import/path rewrite after layout migration."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

targets = list((ROOT / "scripts").rglob("*.py"))
targets += list((ROOT / "apps").rglob("*.py"))
targets += list((ROOT / "tests").rglob("*.py"))

replacements = [
    ("from common import", "from anger_detection.common import"),
    ("from losses import", "from anger_detection.losses import"),
    ("from model_anger import", "from anger_detection.models.model_anger import"),
    ("from model import", "from anger_detection.models.model_tf import"),
    (
        "from train_torch import DSCNN, set_seed",
        "from anger_detection.models import DSCNN, set_seed",
    ),
    ("from train_torch import set_seed", "from anger_detection.models import set_seed"),
    ("from decision import", "from anger_detection.decision import"),
    ("from live_monitor import", "from anger_detection.decision import"),
    (
        "from prepare_data import log_mel_full",
        "from anger_detection.features import log_mel_full",
    ),
    ("from prepare_data import", "from anger_detection.features import"),
]

path_fix = [
    ('ROOT / "dataset" / "hard_negative"', 'CURATED_DIR / "hard_negative"'),
    ("ROOT / 'dataset' / 'hard_negative'", "CURATED_DIR / 'hard_negative'"),
    ('ROOT / "dataset" / "home_anger"', 'CURATED_DIR / "home_anger"'),
    ("ROOT / 'dataset' / 'home_anger'", "CURATED_DIR / 'home_anger'"),
]


def strip_sys_path(text: str) -> str:
    text = re.sub(
        r"\nROOT = Path\(__file__\)\.resolve\(\)\.parents\[\d+\]\n"
        r"(DEMO_DIR = Path\(__file__\)\.resolve\(\)\.parent\n)?"
        r"(sys\.path\.insert\(0, str\([^)]+\)\)\n)+",
        "\n",
        text,
    )
    text = re.sub(
        r"\nsys\.path\.insert\(0, str\(ROOT / \"(?:scripts|demo)\"\)\)\n",
        "\n",
        text,
    )
    text = re.sub(
        r"\nsys\.path\.insert\(0, str\(ROOT / '(?:scripts|demo)'\)\)\n",
        "\n",
        text,
    )
    # leftover ROOT = ... without sys.path (keep if still used; fix curated separately)
    return text


def ensure_curated_import(text: str) -> str:
    if 'CURATED_DIR / "' not in text and "CURATED_DIR / '" not in text:
        return text
    if "CURATED_DIR" in text and "from anger_detection.common import" in text:
        # inject CURATED_DIR into existing common import if missing
        def add_curated(m: re.Match[str]) -> str:
            block = m.group(0)
            if "CURATED_DIR" in block:
                return block
            return block.replace(
                "from anger_detection.common import (",
                "from anger_detection.common import (\n    CURATED_DIR,",
                1,
            )

        text = re.sub(
            r"from anger_detection\.common import \([^)]*\)",
            add_curated,
            text,
            count=1,
            flags=re.S,
        )
        if "CURATED_DIR" not in text.split("from anger_detection.common")[1][:400]:
            # single-line import style
            text = text.replace(
                "from anger_detection.common import ",
                "from anger_detection.common import CURATED_DIR, ",
                1,
            )
    elif "CURATED_DIR" in text:
        text = "from anger_detection.common import CURATED_DIR, PROJECT_ROOT\n" + text
    return text


def main() -> None:
    for path in targets:
        text = path.read_text(encoding="utf-8")
        orig = text
        for a, b in replacements:
            text = text.replace(a, b)
        for a, b in path_fix:
            text = text.replace(a, b)
        text = strip_sys_path(text)
        text = ensure_curated_import(text)
        # drop unused `import sys` only if sys. no longer referenced
        if "import sys" in text and not re.search(r"\bsys\.", text):
            text = re.sub(r"^import sys\n", "", text, count=1, flags=re.M)
        if text != orig:
            path.write_text(text, encoding="utf-8")
            print("updated", path.relative_to(ROOT))
        else:
            print("unchanged", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
