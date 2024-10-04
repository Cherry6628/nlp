'''N-Grams Smoothing'''
import nltk
from nltk import bigrams, word_tokenize
from collections import defaultdict, Counter

# Sample corpus
corpus = """
I am learning to code. I am developing a language model. This language model is for text prediction.  
The model predicts the next word based on previous words. It is used in mobile keyboards.
"""

# Step 1: Tokenize the corpus
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Step 2: Generate bigrams and calculate frequencies
bigram_counts = defaultdict(Counter)
unigram_counts = Counter(tokens)
vocabulary_size = len(unigram_counts)

for w1, w2 in bigrams(tokens):
    bigram_counts[w1][w2] += 1


# Step 3: Apply Add-One Smoothing
def add_one_smoothing(bigram_counts, unigram_counts, vocabulary_size):
    smoothed_probabilities = defaultdict(dict)

    for w1 in unigram_counts:
        for w2 in unigram_counts:
            count_bigram = bigram_counts[w1][w2]  # Count of the bigram (w1, w2)
            count_unigram = unigram_counts[w1]  # Count of the unigram w1
            # Apply Add-One smoothing formula
            smoothed_probabilities[w1][w2] = (count_bigram + 1) / (count_unigram + vocabulary_size)

    return smoothed_probabilities


smoothed_probabilities = add_one_smoothing(bigram_counts, unigram_counts, vocabulary_size)


# Step 4: Display some smoothed bigram probabilities
def display_bigram_probabilities(smoothed_probabilities, word):
    print(f"Smoothed probabilities for bigrams starting with '{word}':")
    for next_word, prob in smoothed_probabilities[word].items():
        print(f"P({next_word} | {word}) = {prob:.6f}")


# Example: Display probabilities for a given word
display_bigram_probabilities(smoothed_probabilities, 'the')