import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time
from datetime import datetime

# Windows paths accessible from WSL
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
data_dir = os.path.join(project_root, "parler_posts_txt")
output_path = os.path.join(project_root, "bertopicOutput")
os.makedirs(output_path, exist_ok=True)

# Check for GPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = "cpu"
    print("⚠️  No GPU detected, using CPU")

# Load documents
print(f"\n📂 Loading documents from: {data_dir}")
text_files = glob.glob(os.path.join(data_dir, "*.txt"))
print(f"Found {len(text_files)} text files")

docs = []
for file_path in text_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                docs.append(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

print(f"Loaded {len(docs)} documents\n")

# Initialize embedding model with GPU
print(f"🤖 Loading embedding model on {device.upper()}...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("✅ Model loaded\n")

# Compute embeddings
print("🔄 Computing embeddings...")
print("=" * 60)
start_time = time.time()

embeddings = embedding_model.encode(
    docs,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=32 if device == "cuda" else 16,
    device=device
)

elapsed = time.time() - start_time
print(f"\n✅ Embeddings computed in {elapsed:.1f}s ({elapsed/len(docs)*1000:.2f}ms per doc)")
print(f"   Shape: {embeddings.shape}")

# Save embeddings
output_file = os.path.join(output_path, "embeddings.npy")
print(f"\n💾 Saving embeddings to: {output_file}")
np.save(output_file, embeddings)
print("✅ Embeddings saved")

# Save metadata
metadata_file = os.path.join(output_path, "embedding_metadata.txt")
with open(metadata_file, 'w', encoding='utf-8') as f:
    f.write(f"Embedding Metadata\n")
    f.write(f"=" * 50 + "\n")
    f.write(f"Model: all-MiniLM-L6-v2\n")
    f.write(f"Device: {device}\n")
    if device == "cuda":
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.write(f"Total documents: {len(docs)}\n")
    f.write(f"Embedding dimensions: {embeddings.shape[1]}\n")
    f.write(f"Data type: float32\n")
    f.write(f"Normalized: Yes\n")
    f.write(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB\n")
    f.write(f"Computation time: {elapsed:.1f}s\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"✅ Metadata saved to: {metadata_file}")
print(f"\n📊 Summary:")
print(f"   Documents processed: {len(docs)}")
print(f"   File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")
print(f"   Processing rate: {len(docs)/elapsed:.1f} docs/second")
print(f"\n🎉 Complete!")