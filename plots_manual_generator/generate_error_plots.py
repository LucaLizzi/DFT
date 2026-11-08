import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from plot_functions import bit_err, err_vs_freq, heat_map


BASE_DIR = Path(__file__).resolve().parent
WAVE_NAME_PATTERN = re.compile(
    r"^amp=(?P<Amplitude>\d+(?:\.\d+)?)"
    r"_f=(?P<Frequency>\d+(?:\.\d+)?)Hz_"
    r"(?P<BinType>on_bin|off_bin\(0\.25\))_"
    r"(?P<Bits>\d+bits|noQuant)$"
)
ARCHITECTURES = ("rsdft", "rsdft_dc", "msdft", "msdft_dc")


def load_results(csv_path):
    source = pd.read_csv(csv_path)
    parsed_names = source["wave name"].str.extract(WAVE_NAME_PATTERN)
    if parsed_names.isna().any().any():
        invalid_names = source.loc[parsed_names.isna().any(axis=1), "wave name"].tolist()
        raise ValueError(f"Unrecognized waveform names: {invalid_names[:5]}")

    parsed_names["Amplitude"] = pd.to_numeric(parsed_names["Amplitude"])
    parsed_names["Frequency"] = pd.to_numeric(parsed_names["Frequency"])
    parsed_names["Bits"] = parsed_names["Bits"].map(
        lambda bits: "Ideal (NoQuant)" if bits == "noQuant" else bits.removesuffix("bits")
    )

    long_frames = []
    for architecture in ARCHITECTURES:
        frame = parsed_names.copy()
        frame["Architecture"] = architecture
        frame["Magnitude error"] = pd.to_numeric(
            source[f"err absolute {architecture}"], errors="raise"
        )
        frame["Complex error"] = pd.to_numeric(
            source[f"err complex {architecture}"], errors="raise"
        )
        long_frames.append(frame)

    results = pd.concat(long_frames, ignore_index=True)
    results["r_ideal"] = np.nan
    results["r_quant"] = np.nan
    return results


def generate_for_folder(input_dir):
    input_csv = input_dir / "error_results.csv"
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir = input_dir.with_name(f"{input_dir.name} (1)")
    results = load_results(input_csv)
    bin_types = results["BinType"].unique()
    fy = {bin_type: None for bin_type in bin_types}

    err_vs_freq(results, fy, output_dir=output_dir)
    bit_err(results, fy, output_dir=output_dir)
    heat_map(results, fy, output_dir=output_dir)

    image_count = len(list(output_dir.glob("*.png")))
    print(f"Generated {image_count} plots in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from image-folder error results.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Process only this source folder. By default, process each images* folder containing error_results.csv.",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        folders = [args.input_dir if args.input_dir.is_absolute() else BASE_DIR / args.input_dir]
    else:
        folders = sorted(
            folder
            for folder in BASE_DIR.iterdir()
            if folder.is_dir()
            and folder.name.startswith("images")
            and (folder / "error_results.csv").is_file()
        )

    if not folders:
        raise FileNotFoundError(f"No images* folders with error_results.csv found in {BASE_DIR}")

    for folder in folders:
        generate_for_folder(folder)
