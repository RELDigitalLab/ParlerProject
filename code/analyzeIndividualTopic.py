"""
Run BERTopic modeling on a single topic's documents from a saved model.
This helps discover sub-topics within the given topic.
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

# SET ARGUMENTS
subfolder_name = 'Rescraped03'  # Subfolder for this analysis
topic_number = 5

# Configuration
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
original_model_path = os.path.join(project_root, "bertopicOutput", subfolder_name, "bertopic_model")
output_path = os.path.join(project_root, "bertopicOutput", f"{subfolder_name}_topic{topic_number}")

# Start timer
start_time = time.time()

print("=" * 80)
print("📊 BERTOPIC INDIVIDUAL TOPIC RE-MODELING")
print("=" * 80)

# Load the original saved model
print(f"\n📂 Loading original BERTopic model from: {original_model_path}")
original_topic_model = BERTopic.load(original_model_path)
print("✅ Model loaded successfully")

# Get topic assignments from original model
topics = original_topic_model.topics_
print(f"\n📈 Total documents in original model: {len(topics)}")

# Find documents for the specified topic
doc_indices = [i for i, topic in enumerate(topics) if topic == topic_number]
num_docs = len(doc_indices)
doc_percentage = (num_docs / len(topics)) * 100

print(f"🔍 Documents in topic {topic_number}: {num_docs} ({doc_percentage:.2f}%)")

if num_docs == 0:
    print(f"\n✅ No documents found for topic {topic_number}!")
    exit(0)

# Reload original documents
print("\n📂 Loading original documents...")
data_dir = os.path.join(project_root, "rescraped_posts_txt")
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
    doc_indices = [i for i in doc_indices if i < min_len]
    num_docs = len(doc_indices)

# Extract only individual topic documents
topic_docs = [docs[i] for i in doc_indices]
print(f"\n✅ Extracted {len(topic_docs)} documents for re-modeling")


# ============================================================================
# RUN TOPIC MODELING ON TOPIC
# ============================================================================
print("\n" + "=" * 80)
print(f"🤖 RUNNING BERTOPIC ON TOPIC {topic_number} DOCUMENTS")
print("=" * 80)

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️  No GPU detected, using CPU")

all_stop_words = list(ENGLISH_STOP_WORDS) #+ custom_stop_words

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
except (ImportError, Exception) as e:
    print("Failed to import cuml UMAP/HDBSCAN. GPU acceleration required.\nError:", e)
    raise

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
        min_cluster_size=100,
        min_samples=5,
        metric='euclidean',
        prediction_data=False,
        verbose=True
    )
    print("✅ GPU-accelerated components configured")
else:
    raise RuntimeError("GPU device required for cuml UMAP/HDBSCAN. Exiting.")

# Initialize embedding model
print(f"\n🤖 Initializing embedding model on {device.upper()}...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Create BERTopic model for individual topic
individual_topic_model = BERTopic(
    verbose=True,
    vectorizer_model=vectorizer_model,
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=False,
    low_memory=True
)

# Fit the model on outlier documents
print(f"\n🔄 Running topic modeling on {len(topic_docs)} outlier documents...")
individual_topics, individual_probs = individual_topic_model.fit_transform(topic_docs)
individual_topics = individual_topic_model.reduce_outliers(topic_docs, individual_topics)
individual_topic_model.update_topics(topic_docs, topics=individual_topics)

print("✅ Individual topic modeling complete!")

# ============================================================================
# SAVE RESULTS
# ============================================================================
os.makedirs(output_path, exist_ok=True)

# Save the outlier model
individual_model_path = os.path.join(output_path, "bertopic_model")
try:
    individual_topic_model.save(
        individual_model_path,
        serialization="pickle",
        save_ctfidf=True,
        save_embedding_model=embedding_model
    )
    print(f"\n✅ Individual model saved to: {individual_model_path}")
except Exception as e:
    print(f"❌ Error saving model: {str(e)}")

# ============================================================================
# ANALYZE AND SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("📊 INDIVIDUAL TOPIC TOPIC MODELING RESULTS")
print("=" * 80)

topic_info = individual_topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info)

# Save detailed results to text file
output_file = os.path.join(output_path, f"topic_{topic_number}_topics.txt")
with open(output_file, 'w', encoding='utf-8') as f:
    print("=" * 80, file=f)
    print(f"BERTOPIC TOPIC {topic_number} ANALYSIS RESULTS", file=f)
    print("=" * 80, file=f)
    print(f"Original Model Path: {original_model_path}", file=f)
    print(f"Original Total Documents: {len(topics)}", file=f)
    print(f"Original Docs in Topic {topic_number}: {num_docs} ({doc_percentage:.2f}%)", file=f)
    print(f"Documents Remodeled: {len(topic_docs)}", file=f)
    print(f"Date/Time: {pd.Timestamp.now()}", file=f)
    print(f"Device: {device.upper()}", file=f)
    print("=" * 80, file=f)
    
    # Topic statistics
    num_new_topics = len(topic_info[topic_info.Topic != -1])    
    print("\nNEW TOPIC STATISTICS:", file=f)
    print(f"Topics found in Topic {topic_number}: {num_new_topics}", file=f)
    
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
        topic_words = individual_topic_model.get_topic(topic_id)
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
    topic_counts = Counter(individual_topics)
    for topic_id, count in sorted(topic_counts.items()):
        percentage = (count / len(individual_topics)) * 100
        print(f"Topic {topic_id}: {count} documents ({percentage:.1f}%)", file=f)

print(f"\n✅ Results saved to: {output_file}")

# Generate visualizations if possible
print("\n" + "=" * 80)
print("📊 GENERATING VISUALIZATIONS")
print("=" * 80)

try:
    unique_topics = len(set(individual_topics)) - (1 if -1 in individual_topics else 0)
    
    if unique_topics >= 2 and len(topic_docs) >= 10:
        # Topic visualization
        print("Creating topic visualization...")
        topic_viz = individual_topic_model.visualize_topics()
        topic_viz_path = os.path.join(output_path, f"topic_{topic_number}_topics_visualization.html")
        topic_viz.write_html(topic_viz_path)
        print(f"✅ Saved: {topic_viz_path}")
        
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
