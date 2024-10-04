import nltk
from nltk import bigrams
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from collections import defaultdict, Counter
import random

# Sample text data
text = """
The cat plays with the ball. The dog chased the cat. The cat is playing in the garden.  
The boy is playing football. The boy plays with the toy. The dog is running.
"""

# Step 1: Ensure 'punkt' is downloaded
try:
    nltk.download('punkt', quiet=True)

    # Step 2: Tokenize the text using word_tokenize (primary tokenizer)
    tokens = word_tokenize(text.lower())

except LookupError:
    print("Error using 'punkt' tokenizer, switching to 'TreebankWordTokenizer' as fallback.")

    # Step 3: Fallback to TreebankWordTokenizer in case 'punkt' isn't working
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text.lower())

# Step 4: Generate bigrams and calculate frequencies
bigram_model = defaultdict(Counter)
for w1, w2 in bigrams(tokens):
    bigram_model[w1][w2] += 1


# Step 5: Generate the next word based on current context
def predict_next_word(context_word):
    possible_words = bigram_model[context_word]

    # If no possible words exist, return None
    if not possible_words:
        return None

    total_count = sum(possible_words.values())

    # Generate a random number to decide the next word
    rand_choice = random.uniform(0, total_count)
    cumulative = 0
    for word, count in possible_words.items():
        cumulative += count
        if cumulative > rand_choice:
            return word
    return None


# Example: Predict the next word after "the"
context_word = "the"
next_word = predict_next_word(context_word)

# Check if a word was predicted or not
if next_word:
    print(f"After '{context_word}', the predicted word is '{next_word}'.")
else:
    print(f"No prediction available for '{context_word}'.")


# Step 6: Generate word forms from root and suffix
def generate_word(root, suffix_info):
    suffix_map = {
        ('n', 'pl'): 's',  # Plural noun suffix
        ('v', 'sg', '3', 'pr'): 's',  # Verb suffix for 3rd person singular present
    }

    suffix = suffix_map.get(tuple(suffix_info), "")
    generated_word = root + suffix
    return generated_word


# Example: Generate word forms
root1 = "boy"
suffix_info1 = ['n', 'pl']  # Example: Plural form
word1 = generate_word(root1, suffix_info1)
print(f"Generated word: {word1} (Root: {root1}, Suffix Info: {suffix_info1})")

root2 = "play"
suffix_info2 = ['v', 'sg', '3', 'pr']  # Example: 3rd person singular present
word2 = generate_word(root2, suffix_info2)
print(f"Generated word: {word2} (Root: {root2}, Suffix Info: {suffix_info2})")
