import os
import glob
import shutil
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import time
from collections import Counter
from bertopic.vectorizers import ClassTfidfTransformer

start_time = time.time()

# Instructions to run
# 1. Run 'conda activate parlerEnv' to activate the environment with required packages
# 2. Execute this script: 'python [path]/bertopicTest.py'

# Attempt to use GPU-accelerated UMAP and HDBSCAN (cuML)
try:
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    from cuml.manifold import UMAP as cumlUMAP
    print("✅ Using cuML (GPU acceleration)")
    use_gpu_clustering = True
except ImportError:
    from hdbscan import HDBSCAN
    from umap import UMAP
    print("ℹ️  Using CPU clustering (cuML not available on Windows)")
    use_gpu_clustering = False

# WSL-native paths for fast file access
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
data_dir = os.path.join(project_root, "parler_posts_txt")
output_path = os.path.join(project_root, "bertopicOutput")
text_files = glob.glob(os.path.join(data_dir, "*.txt"))
embedding_path = os.path.join(project_root, "embeddings.npy") # None


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

# Define custom stop words to filter out Parler-specific metadata
custom_stop_words = [
    # Parler metadata terms
    "impressions", "post", "comments", "echoed", "upvotes", "echoes", "echo",
    # Time-related individual words
    "days", "hours", "minutes", "weeks", "months", "years", 
    "day", "hour", "minute", "week", "month", "year", "ago",
    # Other common metadata terms
    "parler", "user", "profile", "share", "like", "follow", "video", "tag", "support", "browser", "hidden", "private"
]

# Get the built-in English stop words and combine with custom ones
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# Create a CountVectorizer with combined stop words
vectorizer_model = CountVectorizer(
    stop_words=all_stop_words,  # Use combined stop words list
    ngram_range=(1, 2),         # Use both unigrams and bigrams
    min_df=10,                   # Ignore terms that appear in less than 10 documents
    max_features=5000           # Limit to top 5000 features
)

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
# Option 1: Load pre-computed embeddings from file
# Set embedding_path to your .npy file path, or set to None to compute embeddings
# Embeddings will be automatically saved to: [output_path]/embeddings.npy

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
# GPU-ACCELERATED DIMENSIONALITY REDUCTION AND CLUSTERING
# ============================================================================
print("\n📊 Configuring GPU-accelerated components...")

# Try to use GPU-accelerated UMAP and HDBSCAN (cuML)
try:
    if device == "cuda":
        print("Setting up GPU-accelerated UMAP and HDBSCAN (cuML)...")
        umap_model = cumlUMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            verbose=True,
            low_memory=True
        )
        hdbscan_model = cumlHDBSCAN(
            min_cluster_size=1000,
            min_samples=5,
            metric='euclidean',
            prediction_data=False,
            verbose=True
        )
        print("✅ GPU-accelerated UMAP and HDBSCAN configured")
    else:
        raise ImportError("CPU mode - using standard implementations")
except (ImportError, Exception) as e:
    print(f"⚠️  cuML not available ({e}), using CPU-based UMAP and HDBSCAN")
    from umap import UMAP
    from hdbscan import HDBSCAN
    
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        verbose=True
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=10,
        metric='euclidean',
        prediction_data=False
    )

# ============================================================================
# LOAD OR COMPUTE EMBEDDINGS WITH PROGRESS TRACKING
# ============================================================================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
if embedding_path is not None and os.path.exists(embedding_path):
    print(f"\n📂 Loading pre-computed embeddings from: {embedding_path}")
    embeddings = np.load(embedding_path, allow_pickle=True)
    print(f"✅ Loaded embeddings with shape: {embeddings.shape}")
    # Verify dimensions match
    if len(embeddings) != len(docs):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(docs)} documents")
    
    # Create BERTopic model with GPU-accelerated components
    print(f"\n🤖 Initializing BERTopic with GPU-accelerated components...")
    # ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True) # Try uncommenting for better stop words
    # topic_model = BERTopic(ctfidf_model=ctfidf_model )

    # Add embedding_model=embedding_model if no embeddings
    topic_model = BERTopic(
        verbose=True,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False, # perfomance impact, if true will calculate probability for ALL topics per doc (instead of just the assigned one)
        low_memory=True # Might help speed up
    )
    
    print(f"\n🔄 Running topic modeling on {len(docs)} documents...")
    print("Progress:")
    topics, probs = topic_model.fit_transform(docs, embeddings)
    # Reduce outliers and update model
    # topics = topic_model.reduce_outliers(docs, topics)
    # topic_model.update_topics(docs, topics=topics)
    print(f"✅ Using pre-loaded embeddings with shape: {embeddings.shape}")
    
else:
    print("\n💡 No pre-computed embeddings found.")
    


# Save model to file for later reuse
model_path = os.path.join(output_path, "bertopic_model")
try: 
    topic_model.save(
        model_path,
        serialization="pickle", # Pickle can execute code, public sharing should be in safetensors format (safetensors requires type conversions)
        save_ctfidf=True,
        save_embedding_model=embedding_model
    )
    print(f"✅ Model saved as '{model_path}' (can be loaded later with BERTopic.load())")
except Exception as e:
    print(f"❌ Error saving model: {str(e)}")

print(f"✅ Topic modeling complete!")

print("\nTopic Information:")
print(topic_model.get_topic_info())

# words used to fit in Topic 0
print("\nWords in Topic 0:")
print(topic_model.get_topic(0))

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)
print(f"Output directory: {output_path}")
output_file = os.path.join(output_path, "topicModel.txt") # Text output file

with open(output_file, 'w', encoding='utf-8') as fileObj:
    # Get complete topic information
    topic_info = topic_model.get_topic_info()

    # ---- Topic Counts ----
    num_topics = len(topic_info[topic_info.Topic != -1])

    topic_counts = Counter(topics)
    outlier_pct = (topic_counts.get(-1, 0) / len(topics)) * 100

    largest_topic_pct = (
        topic_info[topic_info.Topic != -1]['Count'].max() / len(topics)
    ) * 100

    median_topic_size = topic_info[topic_info.Topic != -1]['Count'].median()

    # ---- Lexical Concentration Metrics ----
    def top10_weight_concentration(topic_words):
        return sum(score for _, score in topic_words[:10])

    def top_ratio(topic_words):
        return topic_words[0][1] / sum(score for _, score in topic_words[:10])

    concentrations = []
    ratios = []

    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        if words and len(words) >= 10:
            concentrations.append(top10_weight_concentration(words))
            ratios.append(top_ratio(words))

    avg_top10_concentration = np.mean(concentrations)
    avg_top_ratio = np.mean(ratios)

    
    # Log header information
    print("=" * 80, file=fileObj)
    print("BERTOPIC COMPLETE ANALYSIS RESULTS", file=fileObj)
    print("=" * 80, file=fileObj)
    print(f"Total Documents Processed: {len(docs)}", file=fileObj)
    print(f"Total Topics Found: {len(topic_info)}", file=fileObj)
    print(f"Date/Time: {pd.Timestamp.now()}", file=fileObj)
    print(f"Device Available: {device.upper()}", file=fileObj)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", file=fileObj)
        print(f"CUDA Version: {torch.version.cuda}", file=fileObj)
    if embedding_path is not None and os.path.exists(embedding_path):
        print(f"Embeddings: Loaded from file ({embedding_path})", file=fileObj)
    else:
        print(f"Embeddings: Computed using SentenceTransformer on {device.upper()}", file=fileObj)
    print(f"UMAP: {type(umap_model).__module__}.{type(umap_model).__name__}", file=fileObj)
    print(f"HDBSCAN: {type(hdbscan_model).__module__}.{type(hdbscan_model).__name__}", file=fileObj)
    print("=" * 80, file=fileObj)

    print("\nMODEL QUALITY METRICS", file=fileObj)
    print("-" * 50, file=fileObj)
    print(f"Number of Topics (excluding outliers): {num_topics}", file=fileObj)
    print(f"Outlier Percentage: {outlier_pct:.2f}%", file=fileObj)
    print(f"Largest Topic Percentage: {largest_topic_pct:.2f}%", file=fileObj)
    print(f"Median Topic Size: {median_topic_size}", file=fileObj)
    print(f"Avg Top-10 Weight Concentration: {avg_top10_concentration:.4f}", file=fileObj)
    print(f"Avg Top1/Top10 Ratio: {avg_top_ratio:.4f}", file=fileObj)

    
    # Log COMPLETE topic info (not just top 10)
    print("\nCOMPLETE TOPIC INFORMATION:", file=fileObj)
    print("-" * 50, file=fileObj)

    # Drop the Representative_Docs column if it exists
    topic_info_clean = topic_info.copy()
    if 'Representative_Docs' in topic_info_clean.columns:
        topic_info_clean = topic_info_clean.drop('Representative_Docs', axis=1)
    
    print(topic_info_clean.to_string(index=False), file=fileObj) 
    
    # Log detailed words for ALL topics
    print("\n\nDETAILED TOPIC WORDS:", file=fileObj)
    print("-" * 50, file=fileObj)
    
    for topic_id in sorted(topic_info['Topic'].tolist()):
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:  # Only process if topic has words
            print(f"\nTopic {topic_id}:", file=fileObj)
            print(f"Count: {topic_info[topic_info['Topic'] == topic_id]['Count'].iloc[0]} documents", file=fileObj)
            print("Words (word, relevance_score):", file=fileObj)
            for word, score in topic_words:
                print(f"  {word}: {score:.4f}", file=fileObj)
        else:
            print(f"\nTopic {topic_id}: No words found (likely outlier topic)", file=fileObj)
    
    # Log all topics using get_topics() method
    print("\n\nALL TOPICS SUMMARY (using get_topics):", file=fileObj)
    print("-" * 50, file=fileObj)
    all_topics = topic_model.get_topics()
    print(f"Total topics retrieved: {len(all_topics)}", file=fileObj)
    
    for topic_id, topic_words in all_topics.items():
        if topic_id == -1:
            print(f"\nTopic {topic_id} (Outlier Topic):", file=fileObj)
            print(f"  Contains {len(topic_words)} word associations", file=fileObj)
        else:
            print(f"\nTopic {topic_id}:", file=fileObj)
            print(f"  Top 15 words: {', '.join([word for word, score in topic_words[:15]])}", file=fileObj)
            print(f"  Total words in topic: {len(topic_words)}", file=fileObj)
            if len(topic_words) > 0:
                print(f"  Highest score: {topic_words[0][1]:.4f}", file=fileObj)
                print(f"  Lowest score: {topic_words[-1][1]:.4f}", file=fileObj)
    
    # Log document-topic assignments
    print("\n\nDOCUMENT-TOPIC ASSIGNMENTS:", file=fileObj)
    print("-" * 50, file=fileObj)
    from collections import Counter
    topic_counts = Counter(topics)
    
    for topic_id, count in sorted(topic_counts.items()):
        percentage = (count / len(topics)) * 100
        print(f"Topic {topic_id}: {count} documents ({percentage:.1f}%)", file=fileObj)
    
    print("\n" + "=" * 80, file=fileObj)
    print("END OF BERTOPIC ANALYSIS", file=fileObj)
    print("=" * 80, file=fileObj)

# Save visualizations to HTML files for later access
print("\nGenerating and saving visualizations...")

try:
    # Get unique topics count to check if visualization is possible
    unique_topics = len(set(topics)) - (1 if -1 in topics else 0)
    
    if unique_topics >= 2 and len(docs) >= 10:
        # 2D topic visualization
        print("Creating topic visualization...")
        topic_viz = topic_model.visualize_topics()
        topic_viz_path = os.path.join(output_path, "bertopic_topics_visualization.html")
        topic_viz.write_html(topic_viz_path)
        print(f"✅ Topic visualization saved as '{topic_viz_path}'")
        
        # # 2D document visualization
        # print("Creating document visualization...")
        # doc_viz = topic_model.visualize_documents(docs, topics=topics, embeddings=embeddings, hide_annotations=True)
        # doc_viz_path = os.path.join(output_path, "bertopic_documents_visualization.html")
        # doc_viz.write_html(doc_viz_path)
        # print(f"✅ Document visualization saved as '{doc_viz_path}'")
        
        # # Topic hierarchy (if enough topics)
        # if unique_topics >= 3:
        #     print("Creating topic hierarchy...")
        #     hierarchy_viz = topic_model.visualize_hierarchy()
        #     hierarchy_viz_path = os.path.join(output_path, "bertopic_hierarchy_visualization.html")
        #     hierarchy_viz.write_html(hierarchy_viz_path)
        #     print(f"✅ Hierarchy visualization saved as '{hierarchy_viz_path}'")
        
        # # Heatmap of topic similarities
        # if unique_topics >= 2:
        #     print("Creating topic heatmap...")
        #     heatmap_viz = topic_model.visualize_heatmap()
        #     heatmap_viz_path = os.path.join(output_path, "bertopic_heatmap_visualization.html")
        #     heatmap_viz.write_html(heatmap_viz_path)
        #     print(f"✅ Heatmap visualization saved as '{heatmap_viz_path}'")
        
        # # Barchart of top words per topic
        # print("Creating topic barchart...")
        # barchart_viz = topic_model.visualize_barchart(top_n_topics=min(10, unique_topics))
        # barchart_viz_path = os.path.join(output_path, "bertopic_barchart_visualization.html")
        # barchart_viz.write_html(barchart_viz_path)
        # print(f"✅ Barchart visualization saved as '{barchart_viz_path}'")
        
        # print(f"\n🎉 All visualizations saved to {output_path}:")
        # print(f"   - bertopic_topics_visualization.html (2D topic map)")
        # print(f"   - bertopic_documents_visualization.html (2D document map)")
        # if unique_topics >= 3:
        #     print(f"   - bertopic_hierarchy_visualization.html (topic hierarchy)")
        # if unique_topics >= 2:
        #     print(f"   - bertopic_heatmap_visualization.html (topic similarity heatmap)")
        # print(f"   - bertopic_barchart_visualization.html (top words per topic)")
        
    else:
        print(f"⚠️  Skipping visualizations - insufficient data:")
        print(f"   Topics found: {unique_topics} (need ≥2)")
        print(f"   Documents: {len(docs)} (need ≥10)")
        
except Exception as e:
    print(f"❌ Visualization failed: {e}")
    print("This often happens with small datasets or when topics are too similar.")

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
