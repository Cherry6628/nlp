import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Download necessary resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Example words
words = ["playing", "played", "plays", "player", "running", "ran"]

# Add-Delete table (simplified example)
# This is a conceptual table for educational purposes
add_delete_table = {
    "ing": ("delete", 3),  # Remove 'ing'
    "ed": ("delete", 2),  # Remove 'ed'
    "s": ("delete", 1),  # Remove 's'
    "er": ("delete", 2),  # Remove 'er'
}


def morphological_analysis(word):
    # Apply stemming
    stem = stemmer.stem(word)

    # Apply lemmatization (considering the word as a verb here for simplicity)
    lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)

    # Use the Add-Delete table to analyze the word
    for suffix, (action, count) in add_delete_table.items():
        if word.endswith(suffix):
            if action == "delete":
                root = word[:-count]
                return f"Word: {word}, Root: {root}, Suffix: {suffix}, Stem: {stem}, Lemma: {lemma}"

    return f"Word: {word}, Root: {lemma}, Stem: {stem}, Lemma: {lemma}"


# Perform morphological analysis on each word
for word in words:
    result = morphological_analysis(word)
    print(result)