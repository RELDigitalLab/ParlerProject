import os
import glob
import shutil
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import time
from collections import Counter
from cuml.cluster import HDBSCAN as cumlHDBSCAN
from cuml.manifold import UMAP as cumlUMAP

start_time = time.time()

# Instructions to run
# 1. Run 'conda activate parlerEnv' to activate the environment with required packages
# 2. Execute this script: 'python [path]/bertopicTest.py'

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ No GPU detected. Exiting.")
    exit(1)

# ============================================================================
# ARGUMENTS: Major configuration options
# ============================================================================
# Set file paths for loading documents and embeddings (WSL-native paths)
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
embedding_path = os.path.join(project_root, "embeddings.npy") # Location of embeddings; pre-computed embeddings are required
data_dir = os.path.join(project_root, "rescraped_posts_txt") # Directory containing text files of documents to analyze
output_path = os.path.join(project_root, "bertopicOutput") # Directory to save model and results
output_file = os.path.join(output_path, "topicModel.txt") # Text output file
saved_model_name = 'bertopic_model' # Filename for saving the BERTopic model


# Set options for the topic modeling process
outlier_reduction = True # Whether to perform outlier reduction after initial topic assignment
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device) # Should be set to match the model used for pre-computed embeddings
umap_model = cumlUMAP(
    n_neighbors=100,
    n_components=5,
    min_dist=0.05,
    metric='cosine',
    verbose=True,
    low_memory=True,
    random_state=42 # A constant number means the same random initialization for reproducibility
)
hdbscan_model = cumlHDBSCAN(
    min_cluster_size=400,
    min_samples=None,  # default based on cluster size
    metric='euclidean',
    prediction_data=False,
    verbose=True
)


docs = []

# Load documents from local directory
text_files = glob.glob(os.path.join(data_dir, "*.txt"))
for file_path in text_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:  # Only add non-empty files
                docs.append(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")    
        continue

# Create a CountVectorizer English stop words
vectorizer_model = CountVectorizer(
    stop_words='english',       # Use built-in English stop words
    ngram_range=(1, 2),         # Use both unigrams and bigrams
    min_df=10,                  # Ignore terms that appear in less than 10 documents
    max_features=5000           # Limit to top 5000 features
)

# ============================================================================
# RUN TOPIC MODELING
# ============================================================================

if embedding_path is not None and os.path.exists(embedding_path):
    print(f"\n📂 Loading pre-computed embeddings from: {embedding_path}")
    embeddings = np.load(embedding_path, allow_pickle=True)
    print(f"✅ Loaded embeddings with shape: {embeddings.shape}")
    # Verify dimensions match
    if len(embeddings) != len(docs):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(docs)} documents")
    
    # Create BERTopic model with GPU-accelerated components
    print(f"\n🤖 Initializing BERTopic with GPU-accelerated components...")

    topic_model = BERTopic(
        verbose=True,
        embedding_model=embedding_model, # Required for dimensionality reduction and clustering, even if embeddings are pre-computed
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False, # perfomance impact, if true will calculate probability for ALL topics per doc (instead of just the assigned one)
        low_memory=True # Might help speed up
    )
    
    print(f"\n🔄 Running topic modeling on {len(docs)} documents...")
    print("Progress:")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    # If enabled, reduce outliers and update model
    if outlier_reduction:
        print("\n🔄 Performing outlier reduction...")
        topics = topic_model.reduce_outliers(docs, topics)
        topic_model.update_topics(docs, topics=topics)
else:
    print("\n💡 No pre-computed embeddings found. Exiting.")
    exit(1)
    


# Save model to file for later reuse
os.makedirs(output_path, exist_ok=True) # create output directory if it doesn't exist
model_path = os.path.join(output_path, saved_model_name)
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
print(f"Output directory: {output_path}")

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
