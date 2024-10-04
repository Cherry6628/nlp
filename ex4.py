import nltk
from nltk import trigrams, word_tokenize
from collections import defaultdict, Counter
import math

# Sample corpus (simplified for this example)
corpus = """
I am sitting in the car. The car is parked. I am going to the market.  
The market is crowded. The car is red. The market is near my house.
"""

# Step 1: Tokenize the corpus
nltk.download('punkt')
tokens = word_tokenize(corpus.lower())

# Step 2: Generate trigrams and calculate frequencies
trigram_model = defaultdict(Counter)

for w1, w2, w3 in trigrams(tokens, pad_right=True, pad_left=True):
    trigram_model[(w1, w2)][w3] += 1

# Step 3: Convert frequencies to probabilities
trigram_probabilities = defaultdict(dict)

for context, counter in trigram_model.items():
    total_count = sum(counter.values())
    for word, count in counter.items():
        trigram_probabilities[context][word] = count / total_count


# Step 4: Calculate the probability of a sentence
def calculate_sentence_probability(sentence):
    tokens = word_tokenize(sentence.lower())
    sentence_prob = 0.0

    for i in range(len(tokens) - 2):
        context = (tokens[i], tokens[i + 1])
        next_word = tokens[i + 2]
        prob = trigram_probabilities.get(context, {}).get(next_word, 1e-6)  # Smoothing for unseen words
        sentence_prob += math.log(prob)

    return math.exp(sentence_prob)


# Example: Calculate probability of a sentence
sentence1 = "the car is parked"
sentence2 = "the market is red"

prob_sentence1 = calculate_sentence_probability(sentence1)
prob_sentence2 = calculate_sentence_probability(sentence2)

print(f"Probability of the sentence '{sentence1}': {prob_sentence1:.6f}")
print(f"Probability of the sentence '{sentence2}': {prob_sentence2:.6f}")