"""
One-time text normalizer for already extracted Parler post text files.

This script:
1) reads existing .txt files,
2) converts emoji to text aliases with emoji.demojize(language="en"),
3) replaces :alias: with spaced alias tokens,
4) writes normalized output files to a separate directory.
"""

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import emoji
from tqdm import tqdm

# Configuration
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
input_dir = os.path.join(project_root, "rescraped_posts_txt")
output_dir = os.path.join(project_root, "rescraped_posts_txt_demojized")

# Compile once for speed
EMOJI_ALIAS_PATTERN = re.compile(r":([a-zA-Z0-9_]+):")


def normalize_text(text: str) -> str:
    text = emoji.demojize(text, language="en")
    return EMOJI_ALIAS_PATTERN.sub(r" \1 ", text)


def process_single_file(filename: str) -> dict:
    try:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Skip if already normalized
        if os.path.exists(output_path):
            return {"status": "skipped", "filename": filename}

        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        normalized_text = normalize_text(text)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(normalized_text)

        return {
            "status": "processed",
            "filename": filename,
            "input_length": len(text),
            "output_length": len(normalized_text),
        }

    except Exception as exc:
        return {"status": "error", "filename": filename, "error": str(exc)}


def process_batch(filenames: list[str]) -> list[dict]:
    results = []
    for filename in filenames:
        results.append(process_single_file(filename))
    return results


def main() -> None:
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    txt_files = [
        filename
        for filename in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, filename)) and filename.endswith(".txt")
    ]

    num_cores = mp.cpu_count()
    batch_size = max(1, len(txt_files) // (num_cores * 4)) if txt_files else 1
    batches = [txt_files[i : i + batch_size] for i in range(0, len(txt_files), batch_size)]

    print("=" * 80)
    print("🧹 ONE-TIME TXT NORMALIZER")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files to process: {len(txt_files):,}")
    print(f"Using {num_cores} CPU cores")
    print(f"Created {len(batches):,} batches (batch size: ~{batch_size})")

    all_results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for batch_results in tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Normalizing TXT",
            unit="batch",
        ):
            all_results.extend(batch_results)

    processed = [result for result in all_results if result["status"] == "processed"]
    skipped = [result for result in all_results if result["status"] == "skipped"]
    errors = [result for result in all_results if result["status"] == "error"]

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\n" + "=" * 80)
    print("✅ NORMALIZATION COMPLETE")
    print("=" * 80)
    print(f"Processed: {len(processed):,}")
    print(f"Skipped: {len(skipped):,}")
    print(f"Errors: {len(errors):,}")

    if errors:
        print(f"\n⚠️  First {min(10, len(errors))} errors:")
        for error in errors[:10]:
            print(f"  - {error['filename']}: {error['error']}")

    if minutes > 0:
        print(f"\n⏱️  Total time: {minutes}m {seconds:.2f}s")
    else:
        print(f"\n⏱️  Total time: {seconds:.2f}s")

    print(f"\nNormalized files written to: {output_dir}")


if __name__ == "__main__":
    main()
