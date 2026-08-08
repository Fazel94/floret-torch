"""Copy trained vectors and the built wheel to Google Drive.

A Colab session is ephemeral -- /content is wiped when the runtime recycles.
Anything you want "later" has to leave the VM. This mounts Drive if it is not
already mounted and copies the pieces worth keeping.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def ensure_drive(dest: Path) -> bool:
    """Mount Drive when running on Colab. Returns False if unavailable."""
    if "/drive/" not in str(dest):
        return True                      # a plain local path; nothing to mount
    if Path("/content/drive/MyDrive").exists():
        return True
    try:
        from google.colab import drive
    except ImportError:
        print("not running on Colab and the destination is a Drive path; "
              "pass a local directory instead", file=sys.stderr)
        return False
    drive.mount("/content/drive")
    return Path("/content/drive/MyDrive").exists()


def main() -> int:
    vectors_dir, package_dir, drive_dir = (Path(p) for p in sys.argv[1:4])
    if not ensure_drive(drive_dir):
        return 1
    drive_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    # The .floret table is the portable artifact: it regenerates the spaCy
    # pipeline anywhere via `spacy init vectors --mode floret`.
    for f in sorted(vectors_dir.glob("*.floret")) + sorted(vectors_dir.glob("*.vec")):
        shutil.copy2(f, drive_dir / f.name)
        copied.append(f.name)
    for whl in sorted(package_dir.rglob("*.whl")):
        shutil.copy2(whl, drive_dir / whl.name)
        copied.append(whl.name)
    # The loadable pipeline directory, zipped for a single-file copy.
    for pipe in sorted(p for p in vectors_dir.iterdir()
                       if p.is_dir() and (p / "config.cfg").exists()):
        out = shutil.make_archive(str(drive_dir / pipe.name), "zip",
                                  root_dir=pipe.parent, base_dir=pipe.name)
        copied.append(Path(out).name)

    if not copied:
        print(f"nothing found under {vectors_dir} or {package_dir}",
              file=sys.stderr)
        return 1
    print(f"copied to {drive_dir}:")
    for name in copied:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
