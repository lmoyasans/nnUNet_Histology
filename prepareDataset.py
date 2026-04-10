from pathlib import Path
import shutil

# Base paths
base = Path("/home/smu/work/workspace")
raw_dir = base / "nnUNet_raw"
preprocessed_dir = base / "nnUNet_preprocessed"
results_dir = base / "nnUNet_results"

# 1. Create required folders if they don't exist
for p in (raw_dir, preprocessed_dir, results_dir):
    p.mkdir(parents=True, exist_ok=True)

def copy_input_folder(input_root: Path, output_dir):
    if not input_root.exists():
        return

    subdirs = [p for p in input_root.iterdir() if p.is_dir()] 
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 top-level folder in {input_root}, "
            f"found {len(subdirs)}: {subdirs}"
        )

    src_folder = subdirs[0]
    dst_folder = output_dir / src_folder.name

    if dst_folder.exists():
        shutil.rmtree(dst_folder)

    shutil.copytree(src_folder, dst_folder)

# Do input_1 (as before)
copy_input_folder(Path("/home/smu/work/inputs/input_1/"), raw_dir)

# Additionally: if input_2 exists, do the same
copy_input_folder(Path("/home/smu/work/inputs/input_2/"), results_dir)