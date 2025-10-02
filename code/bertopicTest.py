import os
import glob
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
# from datasets import load_dataset
import pandas as pd

# display options for printing DataFrames (datatype most BERTopic funcs return)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Parler folder
data_dir = os.path.join(project_root, "data", "parler_posts_txt") # Parler/data/parler_posts_txt
output_path = os.path.join(project_root, "data", "bertopicOutput") # Parler/data/bertopicOutput
text_files = glob.glob(os.path.join(data_dir, "*.txt"))
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
# Combine English stop words with our custom stop words
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words

# Create a CountVectorizer with combined stop words
vectorizer_model = CountVectorizer(
    stop_words=all_stop_words,  # Use combined stop words list
    ngram_range=(1, 2),         # Use both unigrams and bigrams
    min_df=2,                   # Ignore terms that appear in less than 2 documents
    max_features=5000           # Limit to top 5000 features
)

# Use pre-embedded data here
topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(docs) # Custom stop_words now configured to filter out Parler metadata terms

print("\nTop 10 Topics:")
print(topic_model.get_topic_info().head(10))

# words used to fit in Topic 0
print("\nWords in Topic 0:")
print(topic_model.get_topic(0))

output_file = os.path.join(output_path, "topicModel.txt") # Text output file

with open(output_file, 'w', encoding='utf-8') as fileObj:
    # Get complete topic information
    topic_info = topic_model.get_topic_info()
    
    # Log header information
    print("=" * 80, file=fileObj)
    print("BERTOPIC COMPLETE ANALYSIS RESULTS", file=fileObj)
    print("=" * 80, file=fileObj)
    print(f"Total Documents Processed: {len(docs)}", file=fileObj)
    print(f"Total Topics Found: {len(topic_info)}", file=fileObj)
    print(f"Date/Time: {pd.Timestamp.now()}", file=fileObj)
    print("=" * 80, file=fileObj)
    
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

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)
print(f"Output directory: {output_path}")

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
        
        # 2D document visualization
        print("Creating document visualization...")
        doc_viz = topic_model.visualize_documents(docs, topics=topics, hide_annotations=True)
        doc_viz_path = os.path.join(output_path, "bertopic_documents_visualization.html")
        doc_viz.write_html(doc_viz_path)
        print(f"✅ Document visualization saved as '{doc_viz_path}'")
        
        # Topic hierarchy (if enough topics)
        if unique_topics >= 3:
            print("Creating topic hierarchy...")
            hierarchy_viz = topic_model.visualize_hierarchy()
            hierarchy_viz_path = os.path.join(output_path, "bertopic_hierarchy_visualization.html")
            hierarchy_viz.write_html(hierarchy_viz_path)
            print(f"✅ Hierarchy visualization saved as '{hierarchy_viz_path}'")
        
        # Heatmap of topic similarities
        if unique_topics >= 2:
            print("Creating topic heatmap...")
            heatmap_viz = topic_model.visualize_heatmap()
            heatmap_viz_path = os.path.join(output_path, "bertopic_heatmap_visualization.html")
            heatmap_viz.write_html(heatmap_viz_path)
            print(f"✅ Heatmap visualization saved as '{heatmap_viz_path}'")
        
        # Barchart of top words per topic
        print("Creating topic barchart...")
        barchart_viz = topic_model.visualize_barchart(top_n_topics=min(10, unique_topics))
        barchart_viz_path = os.path.join(output_path, "bertopic_barchart_visualization.html")
        barchart_viz.write_html(barchart_viz_path)
        print(f"✅ Barchart visualization saved as '{barchart_viz_path}'")
        
        print(f"\n🎉 All visualizations saved to {output_path}:")
        print(f"   - bertopic_topics_visualization.html (2D topic map)")
        print(f"   - bertopic_documents_visualization.html (2D document map)")
        if unique_topics >= 3:
            print(f"   - bertopic_hierarchy_visualization.html (topic hierarchy)")
        if unique_topics >= 2:
            print(f"   - bertopic_heatmap_visualization.html (topic similarity heatmap)")
        print(f"   - bertopic_barchart_visualization.html (top words per topic)")
        
    else:
        print(f"⚠️  Skipping visualizations - insufficient data:")
        print(f"   Topics found: {unique_topics} (need ≥2)")
        print(f"   Documents: {len(docs)} (need ≥10)")
        
except Exception as e:
    print(f"❌ Visualization failed: {e}")
    print("This often happens with small datasets or when topics are too similar.")

# Save model to file for later reuse
print("\nSaving BERTopic model...")
model_path = os.path.join(output_path, "bertopic_model")
topic_model.save(model_path, serialization="pickle")
print(f"✅ Model saved as '{model_path}' (can be loaded later with BERTopic.load())")
