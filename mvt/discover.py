from pathlib import Path
from typing import List

def list_s1d_files(night_dir: str) -> List[Path]:
    p = Path(night_dir)
    files = sorted(p.glob("ESPRE.*_S1D_*_DRS.fits"))
    return files