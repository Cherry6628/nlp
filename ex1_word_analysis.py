import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download necessary resources
nltk.download('wordnet')
# nltk.download('omw-1.4')

# Sample words for morphological analysis
words = ['play', 'plays', 'played', 'playing']

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


def get_word_morphology(word):
    # Lemmatize the word to get its root form
    root = lemmatizer.lemmatize(word, pos=wordnet.VERB)

    # Identify affixes by comparing the word with its root
    if word.startswith(root):
        prefix = word[:len(root)]
        suffix = word[len(root):]
    else:
        prefix = ''
        suffix = word

    return root, prefix, suffix


# Perform morphological analysis on each word
for word in words:
    root, prefix, suffix = get_word_morphology(word)
    print(f"Word: {word}, Root: {root}, Suffix: {suffix}")
from sys import prefix

# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
#
# nltk.download('wordnet')
# nltk.download("omw-1.4")
#
# word = ['walk', 'walks', 'walking', 'walked']
#
# lemmatizer = WordNetLemmatizer()
#
# def word_finder(word):
#     root = lemmatizer.lemmatize(word, pos=wordnet.VERB)
#     if word.startswith(root):
#         prefix = word[:len(root)]
#         suffix = word[len(root):]
#
#     else:
#         prefix = ""
#         suffix = word
#     return root,prefix,suffix
#
# for word in word:
#     root, prefix, suffix = word_finder(word)
#     print(f"Word : {word} :: prefix : {prefix} :: suffix : {suffix}")
#
#
#
#
#
#




















