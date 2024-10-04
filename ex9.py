import spacy

# Load the SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Sample text for chunking
text = "The quick brown fox jumps over the lazy dog. He quickly eats an apple."

# Process the text through SpaCy
doc = nlp(text)

# Print the noun phrases (NP) and verb phrases (VP)
print("Extracted Phrases:")
for sent in doc.sents:
    noun_phrases = [chunk.text for chunk in sent.noun_chunks]
    verb_phrases = [token.text for token in sent if token.pos_ == "VERB"]

    # Print noun phrases
    print("Noun Phrases:", noun_phrases)

    # Print verb phrases
    print("Verb Phrases:", verb_phrases)
