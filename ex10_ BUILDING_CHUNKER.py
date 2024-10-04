import spacy
import os

# Function to download the SpaCy model if not already installed
def download_spacy_model(model_name='en_core_web_sm'):
    try:
        # Attempt to load the SpaCy model
        nlp = spacy.load(model_name)
        print(f"Loaded model '{model_name}' successfully.")
    except OSError:
        # If model not found, download it
        print(f"Model '{model_name}' not found. Downloading...")
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)
        print(f"Downloaded and loaded model '{model_name}' successfully.")
    return nlp

# Download and load the SpaCy English model
nlp = download_spacy_model()

# Sample text for named entity recognition
text = "Apple Inc. is looking at buying U.K. startup for $1 billion. Steve Jobs and Tim Cook will lead the project."

# Process the text through SpaCy
doc = nlp(text)

# Print the named entities
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# If you want to further extract entities, you can save them in a list:
extracted_entities = [(ent.text, ent.label_) for ent in doc.ents]

print("\nExtracted Entities:")
for entity, label in extracted_entities:
    print(f"{entity}: {label}")
