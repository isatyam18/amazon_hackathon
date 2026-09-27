"""
Stage 5: Official Submission Packaging and Format Validation.

Handles:
1. Validating output files using student_resource/utils/validate_submission.py
2. Creating submission.zip with matching_results.tsv and candidate_pairs.tsv
3. Generating audit reports with file sizes, row counts, and checksums
"""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from typing import Dict, List, Optional, Tuple
import pandas as pd


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_official_validator(
    matching_path: str,
    candidate_path: Optional[str] = None,
    test_dir: str = "student_resource/dataset/test",
    validator_script: str = "student_resource/utils/validate_submission.py",
) -> Tuple[bool, List[str]]:
    """Execute the official student_resource/utils/validate_submission.py script.

    Returns:
        success: True if exit code is 0 (all checks passed)
        output_lines: stdout & stderr lines from the validator
    """
    if not os.path.exists(validator_script):
        return False, [f"Validator script not found at: {validator_script}"]
    if not os.path.exists(matching_path):
        return False, [f"Matching results file not found at: {matching_path}"]

    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_path,
        "--test-dir", test_dir,
    ]
    if candidate_path and os.path.exists(candidate_path):
        cmd.extend(["--candidate", candidate_path])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = proc.stdout.strip().splitlines() + proc.stderr.strip().splitlines()
        success = (proc.returncode == 0)
        return success, output
    except Exception as e:
        return False, [f"Validator execution failed: {e}"]


def package_submission_zip(
    matching_tsv_path: str,
    candidate_tsv_path: Optional[str] = None,
    output_zip_path: str = "output/submission.zip",
) -> Dict[str, any]:
    """Create the competition submission zip file containing output TSVs.

    Returns metadata dictionary with paths, sizes, and hashes.
    """
    os.makedirs(os.path.dirname(output_zip_path) or ".", exist_ok=True)

    files_to_pack = [matching_tsv_path]
    if candidate_tsv_path and os.path.exists(candidate_tsv_path):
        files_to_pack.append(candidate_tsv_path)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_to_pack:
            arcname = os.path.basename(fpath)
            zf.write(fpath, arcname=arcname)
            print(f"  Added {arcname} ({os.path.getsize(fpath) / (1024 * 1024):.2f} MB)")

    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    zip_hash = compute_sha256(output_zip_path)

    meta = {
        "zip_path": output_zip_path,
        "zip_size_mb": round(zip_size_mb, 2),
        "zip_sha256": zip_hash,
        "files_included": [
            {
                "filename": os.path.basename(p),
                "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2),
                "sha256": compute_sha256(p),
                "lines": sum(1 for _ in open(p, encoding="utf-8")),
            }
            for p in files_to_pack
        ],
    }

    report_path = output_zip_path.replace(".zip", "_manifest.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully created {output_zip_path} ({zip_size_mb:.2f} MB)")
    return meta
