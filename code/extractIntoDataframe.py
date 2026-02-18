import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pandas as pd
import time

project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
html_dir = "/mnt/f/Mike's Parler Stuff Please don't mess with this/donk_enby"  # Replace d with your drive letter and add folder path
output_csv = os.path.join(project_root, "parler_data_extracted.csv")

# Start timer
start_time = time.time()

# ==============================================================================
# DATA EXTRACTION FUNCTIONS
# ==============================================================================

def extract_data_from_html(html_content, filename):
    """
    Extract structured data from HTML file.
    
    Returns a dictionary with all extracted fields.
    Add more fields here as you discover what data you need!
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Initialize data dictionary with default values
    data = {
        'filename': filename,
        'text': None,
        # Add more columns here as needed:
        # 'username': None,
        # 'date': None,
        # 'likes': None,
        # 'comments': None,
        # 'shares': None,
        # etc.
    }
    
    # Extract text from card--body divs
    card_bodies = soup.find_all("div", class_="card--body")
    if card_bodies:
        all_text = []
        for card_body in card_bodies:
            p_tag = card_body.find("p")
            if p_tag:
                text = p_tag.get_text(separator=" ", strip=True)
                if text:
                    all_text.append(text)
        data['text'] = "\n".join(all_text) if all_text else None
    
    # TODO: Add more extraction logic here
    # Example: Extract username
    # username_tag = soup.find("div", class_="username")
    # if username_tag:
    #     data['username'] = username_tag.get_text(strip=True)
    
    # Example: Extract date
    # date_tag = soup.find("time")
    # if date_tag:
    #     data['date'] = date_tag.get('datetime') or date_tag.get_text(strip=True)
    
    # Example: Extract engagement metrics
    # likes_tag = soup.find("span", class_="likes-count")
    # if likes_tag:
    #     data['likes'] = likes_tag.get_text(strip=True)
    
    return data


def process_single_file(filename):
    """Process a single HTML file and extract data to dictionary"""
    try:
        html_path = os.path.join(html_dir, filename)
        
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            data = extract_data_from_html(html, filename)
        
        return data  # Return dictionary of extracted data
    except Exception as e:
        # Return error record
        return {
            'filename': filename,
            'text': None,
            'error': str(e)
        }


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
# Get all files, excluding those starting with $
html_files = [f for f in os.listdir(html_dir) if os.path.isfile(os.path.join(html_dir, f)) and not f.startswith("$")]

print(f"📊 Total files to process: {len(html_files)}")

# Use all available CPU cores for maximum performance
num_cores = mp.cpu_count()
print(f"🖥️  Using {num_cores} CPU cores for parallel processing")

# Create batches for better load balancing
batch_size = max(1, len(html_files) // (num_cores * 4))  # 4 batches per core
batches = [html_files[i:i + batch_size] for i in range(0, len(html_files), batch_size)]
print(f"📦 Created {len(batches)} batches (batch size: ~{batch_size})")

print(f"\n🔄 Processing files...")

# Use ProcessPoolExecutor for true parallel processing (not limited by GIL)
if __name__ == '__main__' or True:  # Allow running without __main__ check
    all_data = []  # List to store all extracted data dictionaries
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Process batches in parallel with progress bar
        for batch_results in tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Extracting data from HTML",
            unit="batch"
        ):
            all_data.extend(batch_results)
    
    # Create DataFrame from all extracted data
    print("\n📊 Creating DataFrame...")
    df = pd.DataFrame(all_data)
    
    # Summary statistics
    total_files = len(df)
    successful = df['text'].notna().sum()
    failed = df['text'].isna().sum()
    has_errors = 'error' in df.columns
    
    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Successfully extracted: {successful}")
    print(f"Failed/Empty: {failed}")
    if has_errors:
        error_count = df['error'].notna().sum()
        print(f"Errors encountered: {error_count}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("=" * 80)
    
    # Show preview of data
    print("\n📋 DATAFRAME PREVIEW (first 5 rows):")
    print("-" * 80)
    # Show text preview (truncated)
    preview_df = df.copy()
    if 'text' in preview_df.columns:
        preview_df['text'] = preview_df['text'].apply(
            lambda x: (x[:100] + '...') if pd.notna(x) and len(str(x)) > 100 else x
        )
    print(preview_df.head())
    
    # Save to CSV
    print(f"\n💾 Saving DataFrame to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print("✅ CSV file saved successfully!")
    
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
