import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer, models
import torch
from tqdm import tqdm
import time

start_time = time.time()

# WSL-native paths for fast file access
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
data_dir = os.path.join(project_root, "rescraped_posts_txt") # changed to rescraped ones
output_path = os.path.join(project_root, "bertopicOutput")
text_files = glob.glob(os.path.join(data_dir, "*.txt"))

# # Windows paths accessible from WSL
# project_root = "/mnt/c/Parler"  # Absolute path to your Parler folder on Windows C: drive
# data_dir = os.path.join(project_root, "data", "parler_posts_txt")  # /mnt/c/Parler/data/parler_posts_txt
# output_path = os.path.join(project_root, "data", "bertopicOutput")  # /mnt/c/Parler/data/bertopicOutput
docs = []

# Load documents from local directory
for file_path in text_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:  # Only add non-empty files
                docs.append(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")    
        continue

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = "cpu"
    print("⚠️  No GPU detected, using CPU")

# ============================================================================
# COMPUTE EMBEDDINGS WITH PROGRESS TRACKING
# ============================================================================
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
word_embedding_model = models.Transformer("vinai/bertweet-large") # slower/more memory intensive than base, should be better for longer posts and greater accuracy
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# Initialize SentenceTransformer with the detected device
print(f"Initializing embedding model on {device.upper()}...")
print(f"✅ Embedding model loaded on {device.upper()}")
print(f"\n Computing embeddings on {len(docs)} documents with progress tracking...")

embeddings = embedding_model.encode(
    docs,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=32 if device == "cuda" else 16,
    device=device
)

# Save embeddings
output_file = os.path.join(output_path, "embeddings.npy")
print(f"\n💾 Saving embeddings to: {output_file}")
np.save(output_file, embeddings)
print("✅ Embeddings saved")

# Stop timer and print total execution time
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print("\n" + "=" * 80)
print("⏱️  TOTAL EXECUTION TIME")
print("=" * 80)
if hours > 0:
    print(f"Total time: {hours}h {minutes}m {seconds:.2f}s ({total_time:.2f} seconds)")
elif minutes > 0:
    print(f"Total time: {minutes}m {seconds:.2f}s ({total_time:.2f} seconds)")
else:
    print(f"Total time: {seconds:.2f} seconds")
print("=" * 80)
