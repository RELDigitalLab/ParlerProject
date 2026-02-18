"""
Simple HTML to Text converter for Parler data.
Extracts all text from card--body divs and saves to individual text files.
"""

import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

# Configuration
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
html_dir = r"/mnt/f/Mike's Parler Stuff Please don't mess with this/donk_enby"
txt_dir = os.path.join(project_root, "rescraped_posts_txt")

# Create output directory
os.makedirs(txt_dir, exist_ok=True)

# Start timer
start_time = time.time()


def extract_text_from_html(html_content):
    """
    Extract all text from card--body divs.
    Each card--body has one <p> tag with text content.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find all divs with class "card--body"
    card_bodies = soup.find_all("div", class_="card--body")
    
    if not card_bodies:
        return ""  # Return empty string if no matching divs found
    
    # Extract text from <p> tags within each card--body
    all_text = []
    for card_body in card_bodies:
        # Find <p> tag within this card--body
        p_tag = card_body.find("p")
        if p_tag:
            text = p_tag.get_text(separator=" ", strip=True)
            if text:  # Only add non-empty text
                all_text.append(text)
    
    # Join all text sections with line breaks
    return "\n".join(all_text)


def process_single_file(filename):
    """Process a single HTML file and save as text"""
    try:
        html_path = os.path.join(html_dir, filename)
        txt_filename = filename + ".txt"
        txt_path = os.path.join(txt_dir, txt_filename)
        
        # Skip if already processed
        if os.path.exists(txt_path):
            return {'status': 'skipped', 'filename': filename}
        
        # Read HTML and extract text
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            text = extract_text_from_html(html)
        
        # Skip if no text was extracted
        if not text or text.strip() == "":
            return {'status': 'empty', 'filename': filename}
        
        # Save as text file (only if text exists)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return {'status': 'processed', 'filename': filename, 'length': len(text)}
    
    except Exception as e:
        return {'status': 'error', 'filename': filename, 'error': str(e)}


def process_batch(filenames):
    """Process a batch of files (for multiprocessing)"""
    results = []
    for filename in filenames:
        result = process_single_file(filename)
        results.append(result)
    return results


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

# Get all HTML files, excluding those starting with $
html_files = [
    f for f in os.listdir(html_dir) 
    if os.path.isfile(os.path.join(html_dir, f)) and not f.startswith("$")
]

print("=" * 80)
print("📄 HTML TO TEXT CONVERTER")
print("=" * 80)
print(f"Source directory: {html_dir}")
print(f"Output directory: {txt_dir}")
print(f"Total files to process: {len(html_files):,}")

# Use all available CPU cores
num_cores = mp.cpu_count()
print(f"Using {num_cores} CPU cores for parallel processing")

# Create batches for load balancing
batch_size = max(1, len(html_files) // (num_cores * 4))
batches = [html_files[i:i + batch_size] for i in range(0, len(html_files), batch_size)]
print(f"Created {len(batches)} batches (batch size: ~{batch_size})")

print("\n🔄 Processing files...")
print("=" * 80)

# Process files in parallel
if __name__ == '__main__' or True:
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for batch_results in tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Converting HTML to TXT",
            unit="batch"
        ):
            all_results.extend(batch_results)
    
    # Collect statistics
    processed = [r for r in all_results if r['status'] == 'processed']
    skipped = [r for r in all_results if r['status'] == 'skipped']
    empty = [r for r in all_results if r['status'] == 'empty']
    errors = [r for r in all_results if r['status'] == 'error']
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Total files: {len(html_files):,}")
    print(f"Processed (with text): {len(processed):,} files")
    print(f"Skipped (already exist): {len(skipped):,} files")
    print(f"Empty (no text found): {len(empty):,} files")
    print(f"Errors: {len(errors):,} files")
    
    # Show errors if any
    if errors:
        print(f"\n⚠️  ERRORS ({len(errors)} files):")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err['filename']}: {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Stop timer
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    
    print("\n" + "=" * 80)
    print("⏱️  TOTAL EXECUTION TIME")
    print("=" * 80)
    if minutes > 0:
        print(f"Total time: {minutes}m {seconds:.2f}s")
    else:
        print(f"Total time: {seconds:.2f} seconds")
    print("=" * 80)
    print(f"\n✅ Text files saved to: {txt_dir}")
