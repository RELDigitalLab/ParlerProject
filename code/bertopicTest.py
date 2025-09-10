from bertopic import BERTopic
from datasets import load_dataset
import pandas as pd

# display options for printing DataFrames (datatype most BERTopic funcs return)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# Load dataset of news headlines
dataset = load_dataset("ag_news", split="test[:2000]")  # first 2000 rows of test set
docs = list(dataset["text"]) # Convert to list; pandas DataFrames can also be used to maintain titles

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs)

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
