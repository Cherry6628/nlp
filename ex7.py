# Define a simple set of POS tags and states (words)
states = ('Noun', 'Verb', 'Adjective', 'Adverb')
observations = ('dog', 'barks', 'loudly')
# Initial probabilities
start_prob = {'Noun': 0.6, 'Verb': 0.2, 'Adjective': 0.1, 'Adverb': 0.1}
# Transition probabilities (a)
transition_prob = {
    'Noun': {'Noun': 0.1, 'Verb': 0.6, 'Adjective': 0.2, 'Adverb': 0.1},
    'Verb': {'Noun': 0.4, 'Verb': 0.2, 'Adjective': 0.3, 'Adverb': 0.1},
    'Adjective': {'Noun': 0.2, 'Verb': 0.2, 'Adjective': 0.2, 'Adverb': 0.4},
    'Adverb': {'Noun': 0.3, 'Verb': 0.4, 'Adjective': 0.2, 'Adverb': 0.1}
}
# Emission probabilities (b)
emission_prob = {
    'Noun': {'dog': 0.8, 'barks': 0.1, 'loudly': 0.1},
    'Verb': {'dog': 0.1, 'barks': 0.7, 'loudly': 0.2},
    'Adjective': {'dog': 0.2, 'barks': 0.1, 'loudly': 0.7},
    'Adverb': {'dog': 0.1, 'barks': 0.2, 'loudly': 0.7}
}
# Step 1: Initialize the Viterbi table and the backpointer
V = [{}]
backpointer = [{}]

# Initialization step
for state in states:
    V[0][state] = start_prob[state] * emission_prob[state][observations[0]]
    backpointer[0][state] = None

# Step 2: Recursion step
for t in range(1, len(observations)):
    V.append({})
    backpointer.append({})
    for state in states:
        max_tr_prob = max(V[t-1][prev_state] * transition_prob[prev_state][state] for prev_state in states)
        for prev_state in states:
            if V[t-1][prev_state] * transition_prob[prev_state][state] == max_tr_prob:
                V[t][state] = max_tr_prob * emission_prob[state][observations[t]]
                backpointer[t][state] = prev_state
                break

# Step 3: Termination step
opt = []
# The highest probability
max_prob = max(value for value in V[-1].values())
previous = None
# Get the tag sequence
for state, data in V[-1].items():
    if data == max_prob:
        opt.append(state)
        previous = state
        break

# Follow the backpointer
for t in range(len(observations) - 2, -1, -1):
    opt.insert(0, backpointer[t + 1][previous])
    previous = backpointer[t + 1][previous]

# Output the final results
print("The sentence is: ", " ".join(observations))
print("The best POS tag sequence is: ", " ".join(opt))
print("The Viterbi table is: ")
for i, v in enumerate(V):
    print(f"Time step {i}: {v}")