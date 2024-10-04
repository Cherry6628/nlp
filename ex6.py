from nltk.corpus import brown
from nltk.tag import hmm
from nltk import word_tokenize, download

# Step 1: Load the data
download('brown')
download('universal_tagset')
# Use the Brown corpus with universal tagset for simplicity
train_data = brown.tagged_sents(tagset='universal')

# Step 2: Train an HMM POS tagger
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Step 3: Test the tagger on a new sentence
test_sentence = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(test_sentence)
tags = hmm_tagger.tag(tokens)

# Output the tagged sentence
print("Tagged Sentence:")
for word, tag in tags:
    print(f"{word}: {tag}")

# Step 4: Display Emission and Transition Probabilities

# Emission Probabilities
print("\nEmission Probabilities:")
for state in hmm_tagger._states:
    print(f"\nState: {state}")
    # Get emission probabilities for each symbol (word) for the current state
    for symbol in hmm_tagger._symbols:
        # Calculate the emission probability
        prob = hmm_tagger._output_logprob(state, symbol)
        print(f"{symbol}: {prob}")

# Transition Probabilities
print("\nTransition Probabilities:")
for state in hmm_tagger._states:
    transitions = hmm_tagger._transitions[state]
    print(state, ":", {next_state: transitions.prob(next_state) for next_state in hmm_tagger._states})