import os
from tqdm import tqdm
from bs4 import BeautifulSoup
# from bertopic import BERTopic
# from umap import UMAP

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Parler folder
html_dir = os.path.join(project_root, "data", "test_data")
txt_dir = os.path.join(project_root, "data", "txt_files")
os.makedirs(txt_dir, exist_ok=True)

# Dummy files for testing
# os.makedirs(html_dir, exist_ok=True)
# dummy_htmls = {
#     "doc1.html": "<html><head><title>Doc1</title></head><body><h1>Hello World</h1><p>This is a test document.</p></body></html>",
#     "doc2.html": "<html><body><h1>Python</h1><p>Python is a popular programming language for machine learning.</p></body></html>",
#     "doc3.html": "<html><body><p>Natural language processing (NLP) is fun!</p></body></html>"
# }

# for filename, content in dummy_htmls.items():
#     with open(os.path.join(html_dir, filename), "w", encoding="utf-8") as f:
#         f.write(content)

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

# for filename in os.listdir(html_dir):
#     if filename.endswith(".html"):

# Tracks progress in terminal
html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]
for filename in tqdm(html_files, desc="Converting HTML to TXT"):
    with open(os.path.join(html_dir, filename), "r", encoding="utf-8") as f:
        html = f.read()
        text = extract_text_from_html(html)

    txt_filename = filename.replace(".html", ".txt")
    with open(os.path.join(txt_dir, txt_filename), "w", encoding="utf-8") as f:
        f.write(text)

'''
# Commented out for now due to errors, hopefully will work with larger dataset

# Read in txt files
documents = []
for filename in os.listdir(txt_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(txt_dir, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Topic modeling with BERTopic
umap_model = UMAP(n_neighbors=2, n_components=2, min_dist=0.0, metric="cosine") # Only necessary for small datasets
topic_model = BERTopic(umap_model=umap_model, verbose=True)
topics, probs = topic_model.fit_transform(documents)

print("Documents:", documents)
print("Topics assigned:", topics)

'''
