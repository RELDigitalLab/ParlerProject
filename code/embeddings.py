import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
import gc
from tqdm import tqdm
import torch
import time
from datetime import datetime, timedelta

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

# Start timing
start_time = time.time()
start_datetime = datetime.now()
print(f"🕐 Processing started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

def format_duration(seconds):
    """Format duration in human-readable format"""
    return str(timedelta(seconds=int(seconds)))

# GPU Configuration
def setup_gpu():
    """Configure GPU settings for optimal performance"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"🚀 GPU Setup:")
        print(f"   GPU Count: {gpu_count}")
        print(f"   GPU Name: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return True, gpu_memory
    else:
        print("⚠️  No GPU detected. Using CPU (will be slower)")
        return False, 0

gpu_available, gpu_memory = setup_gpu()

# Checkpoint/resume functionality
def get_last_processed_index(output_path):
    """Check how many documents were already processed and restore timing"""
    global start_time
    checkpoint_file = os.path.join(output_path, "checkpoint.txt")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            lines = f.read().strip().split('\n')
            if len(lines) >= 3:
                # Restore timing information
                index = int(lines[0])
                last_checkpoint_time = float(lines[1])
                original_start_time = float(lines[2])
                
                # Calculate time already spent
                time_spent = last_checkpoint_time - original_start_time
                print(f"📊 Resuming: {format_duration(time_spent)} already spent processing")
                
                # Adjust start_time to account for previous processing
                start_time = time.time() - time_spent
                
                return index
            elif len(lines) >= 1:
                return int(lines[0])
    return 0

def save_checkpoint(output_path, index):
    """Save current progress with timing info"""
    checkpoint_file = os.path.join(output_path, "checkpoint.txt")
    current_time = time.time()
    elapsed = current_time - start_time
    
    with open(checkpoint_file, 'w') as f:
        f.write(f"{index}\n")
        f.write(f"{current_time}\n")  # Save current timestamp
        f.write(f"{start_time}\n")   # Save start timestamp
    
    return elapsed

# Optimize batch size for large datasets and GPU
def get_optimal_batch_size(num_docs, gpu_available, gpu_memory):
    """Determine optimal batch size based on dataset size and GPU"""
    if not gpu_available:
        # CPU-only batch sizes
        if num_docs < 1000:
            return 50
        elif num_docs < 10000:
            return 100
        else:
            return 32
    else:
        # GPU batch sizes - larger batches for better GPU utilization
        if gpu_memory >= 8:  # 8GB+ GPU
            if num_docs < 1000:
                return 200
            elif num_docs < 10000:
                return 500
            else:
                return 256  # Large batches for massive datasets
        elif gpu_memory >= 4:  # 4-8GB GPU
            if num_docs < 1000:
                return 100
            elif num_docs < 10000:
                return 200
            else:
                return 128
        else:  # <4GB GPU
            if num_docs < 1000:
                return 50
            else:
                return 64

batch_size = get_optimal_batch_size(len(docs), gpu_available, gpu_memory)
print(f"Using batch size: {batch_size} for {len(docs)} documents ({'GPU' if gpu_available else 'CPU'} mode)")
# Load model with GPU support
device = "cuda" if gpu_available else "cpu"
print(f"Loading model on device: {device}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# GPU memory optimization
if gpu_available:
    print("🔧 Optimizing GPU settings...")
    # Enable mixed precision for faster processing
    if hasattr(embedding_model, 'to'):
        embedding_model = embedding_model.to(device)
    
    # Set memory management
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.cuda.empty_cache()
    
    print(f"✅ Model loaded on GPU with optimizations")

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
            # Generate embeddings for this batch with GPU optimization
            batch_embeddings = embedding_model.encode(
                batch, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=min(batch_size, len(batch)),  # Ensure we don't exceed batch
                device=device  # Explicitly specify device
            )
            
            # Write directly to memory-mapped file
            embeddings_mmap[current_start:current_end] = batch_embeddings.astype('float32')
            
            # GPU memory management
            del batch_embeddings
            gc.collect()
            if gpu_available:
                torch.cuda.empty_cache()  # Clear GPU memory
            
            # Save checkpoint every 10 batches
            if batch_num % 10 == 0:
                embeddings_mmap.flush()  # Ensure data is written to disk
                elapsed = save_checkpoint(output_path, current_end)
                
                # Calculate processing rate and ETA
                docs_processed = current_end - start_idx
                rate = docs_processed / elapsed if elapsed > 0 else 0
                remaining_docs = len(docs) - current_end
                eta_seconds = remaining_docs / rate if rate > 0 else 0
                
                # GPU memory status
                if gpu_available:
                    gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
                    print(f"  ✅ Checkpoint: {current_end}/{len(docs)} docs | {format_duration(elapsed)} elapsed | {rate:.1f} docs/sec | ETA: {format_duration(eta_seconds)} | GPU: {gpu_used:.1f}GB/{gpu_cached:.1f}GB")
                else:
                    print(f"  ✅ Checkpoint: {current_end}/{len(docs)} docs | {format_duration(elapsed)} elapsed | {rate:.1f} docs/sec | ETA: {format_duration(eta_seconds)}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"\n⚠️  Memory error at batch {batch_num + 1}")
                if gpu_available:
                    print(f"GPU memory issue - consider reducing batch_size from {batch_size}")
                    torch.cuda.empty_cache()
                else:
                    print(f"RAM issue - consider reducing batch_size from {batch_size}")
                print(f"Processed {current_start}/{len(docs)} documents before error")
                save_checkpoint(output_path, current_start)
                break
            else:
                raise e
        except KeyboardInterrupt:
            print(f"\n⏸️  Process interrupted by user")
            elapsed = save_checkpoint(output_path, current_end)
            embeddings_mmap.flush()
            print(f"⏱️  Time elapsed: {format_duration(elapsed)}")
            print(f"📊 Processing rate: {(current_end - start_idx) / elapsed:.1f} docs/sec")
            print("✅ Progress saved. You can resume later by running the script again.")
            exit(0)
    
    # Calculate final timing
    end_time = time.time()
    total_duration = end_time - start_time
    end_datetime = datetime.now()
    
    print(f"\n🎉 Processing completed!")
    print(f"⏱️  Total time: {format_duration(total_duration)}")
    print(f"📊 Average rate: {len(docs) / total_duration:.1f} documents/second")
    print(f"🕐 Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
        f.write(f"Device used: {device}\n")
        if gpu_available:
            f.write(f"GPU name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU memory: {gpu_memory:.1f} GB\n")
        f.write(f"Processing mode: {'GPU-accelerated' if gpu_available else 'CPU-only'}\n")
        # Timing information
        f.write(f"Processing started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total duration: {format_duration(total_duration)}\n")
        f.write(f"Average processing rate: {len(docs) / total_duration:.2f} documents/second\n")
        f.write(f"Total duration (seconds): {total_duration:.2f}\n")
    
    print(f"✅ Metadata saved to: {metadata_file}")
    print(f"📊 Final file size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

except Exception as e:
    # Calculate elapsed time for error reporting
    error_time = time.time()
    elapsed_error = error_time - start_time
    print(f"❌ Error after {format_duration(elapsed_error)}: {e}")
    print("💡 Try reducing batch_size or check available memory")
    # Save current progress before exiting
    if 'current_end' in locals():
        save_checkpoint(output_path, current_end)
        print(f"Progress saved at document {current_end}")
except KeyboardInterrupt:
    # Calculate elapsed time for interruption
    interrupt_time = time.time()
    elapsed_interrupt = interrupt_time - start_time
    print(f"\n⏸️  Process interrupted after {format_duration(elapsed_interrupt)}")
    if 'current_end' in locals():
        save_checkpoint(output_path, current_end)
        print(f"Progress saved at document {current_end}")
    print("You can resume by running the script again.")