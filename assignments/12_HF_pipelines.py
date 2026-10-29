"""A text-classification pipeline and an NER pipeline, ready to experiment with.

    uv run assignments/12_HF_pipelines.py

Change TEXTS, swap in a different model from the Hub, and see what happens.

Read 12_HF_tutorials.md first.
"""

from transformers import pipeline

# Both models are named explicitly. Without a name, pipeline() picks a default,
# and the default NER model is a 1.3 GB BERT-large. These two are about 270 MB
# and 430 MB, download themselves the first time, and are cached in
# ~/.cache/huggingface afterward.
CLASSIFIER_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
NER_MODEL = "dslim/bert-base-NER"

TEXTS = [
    "I've been waiting for a HuggingFace course my whole life.",
    "My name is Sylvain and I work at Hugging Face in Brooklyn.",
]

classifier = pipeline("text-classification", model=CLASSIFIER_MODEL)
# The course writes grouped_entities=True. That argument no longer exists;
# aggregation_strategy is its replacement, and it glues word pieces back into
# whole entities. "first" labels a word by its first piece. ("simple" leaves
# this model's output in fragments like S ##yl ##va ##in -- try it and see.)
ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="first")

for text in TEXTS:
    print(text)
    for result in classifier(text):
        print(f"  {result['label']:<10} {result['score']:.4f}")
    for entity in ner(text):
        print(f"  {entity['entity_group']:<10} {entity['score']:.4f}  {entity['word']}")
    print()
