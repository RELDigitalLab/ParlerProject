"""
Run BERTopic modeling on outlier documents from a saved model.
This helps discover sub-topics within documents that were classified as outliers.
"""

import os
import random
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import glob
import time
from collections import Counter

# Configuration
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
original_model_path = os.path.join(project_root, "bertopicOutput", "baselineWithVisualizations", "bertopic_model")
output_path = os.path.join(project_root, "bertopicOutput", "outlier_analysis")

# Start timer
start_time = time.time()

print("=" * 80)
print("📊 BERTOPIC OUTLIER RE-MODELING")
print("=" * 80)

# Load the original saved model
print(f"\n📂 Loading original BERTopic model from: {original_model_path}")
original_topic_model = BERTopic.load(original_model_path)
print("✅ Model loaded successfully")

# Get topic assignments from original model
topics = original_topic_model.topics_
print(f"\n📈 Total documents in original model: {len(topics)}")

# Find outlier documents (topic == -1)
outlier_indices = [i for i, topic in enumerate(topics) if topic == -1]
num_outliers = len(outlier_indices)
outlier_percentage = (num_outliers / len(topics)) * 100

print(f"🔍 Outlier documents: {num_outliers} ({outlier_percentage:.2f}%)")

if num_outliers == 0:
    print("\n✅ No outliers found in this model!")
    exit(0)

# Reload original documents
print("\n📂 Loading original documents...")
data_dir = os.path.join(project_root, "parler_posts_txt")
text_files = glob.glob(os.path.join(data_dir, "*.txt"))

docs = []
for file_path in text_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                docs.append(content)
    except Exception as e:
        continue

print(f"✅ Loaded {len(docs)} documents")

# Verify document count matches
if len(docs) != len(topics):
    print(f"\n⚠️  WARNING: Document count mismatch!")
    print(f"   Loaded: {len(docs)} documents")
    print(f"   Model has: {len(topics)} topic assignments")
    min_len = min(len(docs), len(topics))
    docs = docs[:min_len]
    outlier_indices = [i for i in outlier_indices if i < min_len]
    num_outliers = len(outlier_indices)

# Extract only outlier documents
outlier_docs = [docs[i] for i in outlier_indices]
print(f"\n✅ Extracted {len(outlier_docs)} outlier documents for re-modeling")


# ============================================================================
# RUN TOPIC MODELING ON OUTLIERS
# ============================================================================
print("\n" + "=" * 80)
print("🤖 RUNNING BERTOPIC ON OUTLIER DOCUMENTS")
print("=" * 80)

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️  No GPU detected, using CPU")

# Define custom stop words (same as bertopicTest.py)
custom_stop_words = [
    "impressions", "post", "comments", "echoed", "upvotes", "echoes", "echo",
    "days", "hours", "minutes", "weeks", "months", "years", 
    "day", "hour", "minute", "week", "month", "year", "ago",
    "parler", "user", "profile", "share", "like", "follow", "video", "tag", "support", "browser", "hidden", "private"
]
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# Create vectorizer
vectorizer_model = CountVectorizer(
    stop_words=all_stop_words,
    ngram_range=(1, 2),
    min_df=5,  # Lower threshold for smaller dataset
    max_features=5000
)

# Setup UMAP and HDBSCAN
try:
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    from cuml.manifold import UMAP as cumlUMAP
    
    if device == "cuda":
        print("Setting up GPU-accelerated UMAP and HDBSCAN...")
        umap_model = cumlUMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            verbose=True,
            low_memory=True
        )
        hdbscan_model = cumlHDBSCAN(
            min_cluster_size=50,  # Smaller for outlier subset
            min_samples=5,
            metric='euclidean',
            prediction_data=False,
            verbose=True
        )
        print("✅ GPU-accelerated components configured")
    else:
        raise ImportError("Using CPU")
except (ImportError, Exception) as e:
    print(f"⚠️  Using CPU-based UMAP and HDBSCAN")
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
        min_cluster_size=50,  # Smaller for outlier subset
        min_samples=10,
        metric='euclidean',
        prediction_data=False
    )

# Initialize embedding model
print(f"\n🤖 Initializing embedding model on {device.upper()}...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Create BERTopic model for outliers
outlier_topic_model = BERTopic(
    verbose=True,
    vectorizer_model=vectorizer_model,
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,
    low_memory=True
)

# Fit the model on outlier documents
print(f"\n🔄 Running topic modeling on {len(outlier_docs)} outlier documents...")
outlier_topics, outlier_probs = outlier_topic_model.fit_transform(outlier_docs)

print("✅ Outlier topic modeling complete!")

# ============================================================================
# SAVE RESULTS
# ============================================================================
os.makedirs(output_path, exist_ok=True)

# Save the outlier model
outlier_model_path = os.path.join(output_path, "outlier_bertopic_model")
try:
    outlier_topic_model.save(
        outlier_model_path,
        serialization="pickle",
        save_ctfidf=True,
        save_embedding_model=embedding_model
    )
    print(f"\n✅ Outlier model saved to: {outlier_model_path}")
except Exception as e:
    print(f"❌ Error saving model: {str(e)}")

# ============================================================================
# ANALYZE AND SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("📊 OUTLIER TOPIC MODELING RESULTS")
print("=" * 80)

topic_info = outlier_topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info)

# Save detailed results to text file
output_file = os.path.join(output_path, "outlier_topics.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    print("=" * 80, file=f)
    print("BERTOPIC OUTLIER ANALYSIS RESULTS", file=f)
    print("=" * 80, file=f)
    print(f"Original Model Path: {original_model_path}", file=f)
    print(f"Original Total Documents: {len(topics)}", file=f)
    print(f"Original Outliers: {num_outliers} ({outlier_percentage:.2f}%)", file=f)
    print(f"Documents Remodeled: {len(outlier_docs)}", file=f)
    print(f"New Topics Found: {len(topic_info)}", file=f)
    print(f"Date/Time: {pd.Timestamp.now()}", file=f)
    print(f"Device: {device.upper()}", file=f)
    print("=" * 80, file=f)
    
    # Topic statistics
    num_new_topics = len(topic_info[topic_info.Topic != -1])
    new_outlier_count = sum(1 for t in outlier_topics if t == -1)
    new_outlier_pct = (new_outlier_count / len(outlier_topics)) * 100
    
    print("\nNEW TOPIC STATISTICS:", file=f)
    print(f"Topics found in outliers: {num_new_topics}", file=f)
    print(f"Still outliers after remodeling: {new_outlier_count} ({new_outlier_pct:.2f}%)", file=f)
    print(f"Now assigned to topics: {len(outlier_topics) - new_outlier_count} ({100-new_outlier_pct:.2f}%)", file=f)
    
    # Topic info table
    print("\n\nTOPIC INFORMATION:", file=f)
    print("-" * 50, file=f)
    topic_info_clean = topic_info.copy()
    if 'Representative_Docs' in topic_info_clean.columns:
        topic_info_clean = topic_info_clean.drop('Representative_Docs', axis=1)
    print(topic_info_clean.to_string(index=False), file=f)
    
    # Detailed words for each topic
    print("\n\nDETAILED TOPIC WORDS:", file=f)
    print("-" * 50, file=f)
    for topic_id in sorted(topic_info['Topic'].tolist()):
        topic_words = outlier_topic_model.get_topic(topic_id)
        if topic_words:
            print(f"\nTopic {topic_id}:", file=f)
            count = topic_info[topic_info['Topic'] == topic_id]['Count'].iloc[0]
            print(f"Count: {count} documents", file=f)
            print("Top words:", file=f)
            for word, score in topic_words[:20]:
                print(f"  {word}: {score:.4f}", file=f)
        else:
            print(f"\nTopic {topic_id}: No words (outlier)", file=f)
    
    # Document assignments
    print("\n\nDOCUMENT-TOPIC ASSIGNMENTS:", file=f)
    print("-" * 50, file=f)
    topic_counts = Counter(outlier_topics)
    for topic_id, count in sorted(topic_counts.items()):
        percentage = (count / len(outlier_topics)) * 100
        print(f"Topic {topic_id}: {count} documents ({percentage:.1f}%)", file=f)

print(f"\n✅ Results saved to: {output_file}")

# Generate visualizations if possible
print("\n" + "=" * 80)
print("📊 GENERATING VISUALIZATIONS")
print("=" * 80)

try:
    unique_topics = len(set(outlier_topics)) - (1 if -1 in outlier_topics else 0)
    
    if unique_topics >= 2 and len(outlier_docs) >= 10:
        # Topic visualization
        print("Creating topic visualization...")
        topic_viz = outlier_topic_model.visualize_topics()
        topic_viz_path = os.path.join(output_path, "outlier_topics_visualization.html")
        topic_viz.write_html(topic_viz_path)
        print(f"✅ Saved: {topic_viz_path}")
        
        # Barchart
        print("Creating topic barchart...")
        barchart_viz = outlier_topic_model.visualize_barchart(top_n_topics=min(10, unique_topics))
        barchart_path = os.path.join(output_path, "outlier_barchart_visualization.html")
        barchart_viz.write_html(barchart_path)
        print(f"✅ Saved: {barchart_path}")
        
        print(f"\n🎉 Visualizations saved to: {output_path}")
    else:
        print(f"⚠️  Insufficient topics for visualization (found {unique_topics}, need ≥2)")
        
except Exception as e:
    print(f"❌ Visualization failed: {e}")

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
