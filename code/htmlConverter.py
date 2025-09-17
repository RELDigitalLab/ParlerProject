import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Parler folder
html_dir = os.path.join(project_root, "data", "donk_enby", "parler_posts_unzipped")
txt_dir = os.path.join(project_root, "data", "parler_posts_txt") # Parler/data/txt_files
os.makedirs(txt_dir, exist_ok=True) # If it already exists, no error

# Convert HTML files to txt
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find all divs with class "post--card--wrapper"
    post_card_wrappers = soup.find_all("div", class_="post--card--wrapper")
    
    if not post_card_wrappers:
        return ""  # Return empty string if no matching divs found
    
    # Extract text from all matching divs
    all_text = []
    for wrapper in post_card_wrappers:
        # Remove script and style tags from within this div
        for script in wrapper(["script", "style"]):
            script.extract()
        
        # Replace specific images with their alt text inline, remove others
        for img in wrapper.find_all("img"):
            alt_text = img.get("alt", "").strip()
            if alt_text in ["Impressions", "Post Comments", "Post Echoes", "Post Upvotes"]:
                img.replace_with(f"[{alt_text}]:")
            else:
                img.extract()  # Remove the image entirely
        
        # Get text from this div and its children (now includes inline image replacements)
        text = wrapper.get_text(separator=" ")
        cleaned_text = " ".join(text.split())
        
        if cleaned_text.strip():  # Only add non-empty text
            all_text.append(cleaned_text.strip())
    
    return " ".join(all_text)

def process_single_file(filename):
    """Process a single HTML file and convert it to TXT"""
    try:
        html_path = os.path.join(html_dir, filename)
        txt_filename = filename.replace(".html", ".txt")
        txt_path = os.path.join(txt_dir, txt_filename)
        
        # Skip if already processed
        if os.path.exists(txt_path):
            return f"Skipped {filename} (already exists)"
        
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            text = extract_text_from_html(html)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        return f"Processed {filename}"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]

# Multithreaded processing
max_workers = min(32, (os.cpu_count() or 1) + 4)  # Reasonable thread count
print(f"Processing {len(html_files)} files using {max_workers} threads...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Use tqdm to track progress
    list(tqdm(
        executor.map(process_single_file, html_files),
        total=len(html_files),
        desc="Converting HTML to TXT"
    ))

