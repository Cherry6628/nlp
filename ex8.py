from subprocess import call
call("pip install sklearn-crfsuite scikit-learn")
del call
from sklearn_crfsuite import metrics, CRF
from sklearn.model_selection import train_test_split
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),}
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(), })
    else:
        features['BOS'] = True # Beginning of sentence
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True # End of sentence
    return features
def sent2features(sent): return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent): return [label for token, label in sent]
def sent2tokens(sent): return [token for token, label in sent]
data = [
    [('Aspirin', 'Noun'), ('is', 'Verb'), ('used', 'Verb'), ('to', 'Preposition'), ('reduce', 'Verb'), ('fever', 'Noun')],
    [('The', 'Determiner'), ('patient', 'Noun'), ('was', 'Verb'), ('prescribed', 'Verb'), ('penicillin', 'Noun')],]
X = [sent2features(s) for s in data]
y = [sent2labels(s) for s in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
crf = CRF(algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True)
crf.fit(X_train, y_train)
y_pred = crf.predict(X_test)
labels = list(crf.classes_)
metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))
