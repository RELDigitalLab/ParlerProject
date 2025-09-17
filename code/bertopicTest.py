import os
import glob
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
# from datasets import load_dataset
import pandas as pd

# display options for printing DataFrames (datatype most BERTopic funcs return)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Parler folder
data_dir = os.path.join(project_root, "data", "test_data") # Parler/data/test_data
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
    "[impressions]:", "[post", "comments]:", "echoes]:", "upvotes]:"
]

# Create a CountVectorizer with custom stop words
vectorizer_model = CountVectorizer(
    stop_words="english",  # Include standard English stop words
    ngram_range=(1, 2),    # Use both unigrams and bigrams
    min_df=2,              # Ignore terms that appear in less than 2 documents
    max_features=5000      # Limit to top 5000 features
)

# Add custom stop words to the vectorizer's stop words
if vectorizer_model.stop_words:
    vectorizer_model.stop_words = vectorizer_model.stop_words.union(set(custom_stop_words))
else:
    vectorizer_model.stop_words = set(custom_stop_words)

topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(docs) # Custom stop_words now configured to filter out Parler metadata terms

print("\nTop 10 Topics:")
print(topic_model.get_topic_info().head(10))

# words used to fit in Topic 0
print("\nWords in Topic 0:")
print(topic_model.get_topic(0))

with open ('bertopicTestOutput.txt', 'w') as fileObj:
    print("Top 10 Topics:", file=fileObj)
    print(topic_model.get_topic_info().head(10), file=fileObj)

    print("\nWords in Topic 0:", file=fileObj)
    print(topic_model.get_topic(0), file=fileObj)

# 2d map of topics
topic_model.visualize_topics().show()

# 2d map of individual documents and their relation to topics
topic_model.visualize_documents(docs, topics=topics, hide_annotations=True).show()

# save model to file
# topic_model.save("testing_bertopic_model.pkl", serialization="pickle")
