import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
import gc
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Parler folder
data_dir = os.path.join(project_root, "data", "parler_posts_txt") # Parler/data/parler_posts_txt
output_path = os.path.join(project_root, "data", "embeddings") # Output directory
os.makedirs(output_path, exist_ok=True)

text_files = glob.glob(os.path.join(data_dir, "*.txt"))
docs = []

print(f"Found {len(text_files)} text files")

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

print(f"Loaded {len(docs)} documents")

# Checkpoint/resume functionality
def get_last_processed_index(output_path):
    """Check how many documents were already processed"""
    checkpoint_file = os.path.join(output_path, "checkpoint.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(output_path, index):
    """Save current progress"""
    checkpoint_file = os.path.join(output_path, "checkpoint.txt")
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))

# Optimize batch size for large datasets
def get_optimal_batch_size(num_docs):
    """Determine optimal batch size based on dataset size"""
    if num_docs < 1000:
        return 100
    elif num_docs < 10000:
        return 500
    elif num_docs < 100000:
        return 100
    else:
        return 32  # Very small batches for 1M+ documents

batch_size = get_optimal_batch_size(len(docs))
print(f"Using batch size: {batch_size} for {len(docs)} documents")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # English language model

# Check for existing checkpoint
start_idx = get_last_processed_index(output_path)
if start_idx > 0:
    print(f"🔄 Resuming from document {start_idx}")

# Memory-efficient processing for large datasets
try:
    print(f"Processing {len(docs)} documents starting from index {start_idx}")
    
    # Pre-allocate memory-mapped array for efficiency
    output_file = os.path.join(output_path, "embeddings.npy")
    
    # Get embedding dimension from first small batch (if starting fresh)
    if start_idx == 0:
        print("Getting embedding dimensions from sample...")
        sample_embedding = embedding_model.encode([docs[0]])
        embedding_dim = sample_embedding.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
        
        # Create memory-mapped array to write directly to disk
        embeddings_mmap = np.memmap(
            output_file, 
            dtype='float32', 
            mode='w+', 
            shape=(len(docs), embedding_dim)
        )
    else:
        # Resume: load existing file
        print("Loading existing embeddings file...")
        embeddings_mmap = np.memmap(output_file, dtype='float32', mode='r+')
        embedding_dim = embeddings_mmap.shape[1]
        print(f"Resuming with embedding dimension: {embedding_dim}")
    
    # Process in batches with progress tracking
    total_batches = (len(docs) + batch_size - 1) // batch_size
    remaining_docs = len(docs) - start_idx
    
    print(f"Processing {remaining_docs} remaining documents in batches of {batch_size}")
    
    for batch_num in range(start_idx // batch_size, total_batches):
        current_start = batch_num * batch_size
        current_end = min(current_start + batch_size, len(docs))
        
        # Skip if we've already processed this batch
        if current_start < start_idx:
            continue
            
        batch = docs[current_start:current_end]
        
        print(f"Processing batch {batch_num + 1}/{total_batches} (docs {current_start}-{current_end-1})")
        
        try:
            # Generate embeddings for this batch
            batch_embeddings = embedding_model.encode(
                batch, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Helps with memory and downstream tasks
            )
            
            # Write directly to memory-mapped file
            embeddings_mmap[current_start:current_end] = batch_embeddings.astype('float32')
            
            # Force memory cleanup
            del batch_embeddings
            gc.collect()
            
            # Save checkpoint every 10 batches
            if batch_num % 10 == 0:
                embeddings_mmap.flush()  # Ensure data is written to disk
                save_checkpoint(output_path, current_end)
                print(f"  ✅ Checkpoint saved: {current_end}/{len(docs)} documents processed")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️  Memory error at batch {batch_num + 1}")
                print(f"Consider reducing batch_size from {batch_size}")
                print(f"Processed {current_start}/{len(docs)} documents before error")
                save_checkpoint(output_path, current_start)
                break
            else:
                raise e
        except KeyboardInterrupt:
            print(f"\n⏸️  Process interrupted by user")
            print(f"Saving checkpoint at document {current_end}")
            save_checkpoint(output_path, current_end)
            embeddings_mmap.flush()
            print("✅ Progress saved. You can resume later by running the script again.")
            exit(0)
    
    # Ensure all data is written to disk
    embeddings_mmap.flush()
    print(f"✅ All embeddings saved to: {output_file}")
    
    # Remove checkpoint file since we're done
    checkpoint_file = os.path.join(output_path, "checkpoint.txt")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Save comprehensive metadata
    metadata_file = os.path.join(output_path, "embedding_metadata.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: all-MiniLM-L6-v2\n")
        f.write(f"Total documents: {len(docs)}\n")
        f.write(f"Embedding dimensions: {embedding_dim}\n")
        f.write(f"Data type: float32\n")
        f.write(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB\n")
        f.write(f"Batch size used: {batch_size}\n")
        f.write(f"Source directory: {data_dir}\n")
        f.write(f"Memory-mapped: Yes\n")
        f.write(f"Normalized embeddings: Yes\n")
    
    print(f"✅ Metadata saved to: {metadata_file}")
    print(f"📊 Final file size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

except Exception as e:
    print(f"❌ Error during embedding generation: {e}")
    print("💡 Try reducing batch_size or check available memory")
    # Save current progress before exiting
    if 'current_end' in locals():
        save_checkpoint(output_path, current_end)
        print(f"Progress saved at document {current_end}")
except KeyboardInterrupt:
    print(f"\n⏸️  Process interrupted by user")
    if 'current_end' in locals():
        save_checkpoint(output_path, current_end)
        print(f"Progress saved at document {current_end}")
    print("You can resume by running the script again.")