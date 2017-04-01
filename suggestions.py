from nltk.corpus import wordnet

def getSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        # print(syn)
        for l in syn.lemmas():
            synonyms.append(l.name())
    return list(set(synonyms))

def getMisspeled(word):
    return []

def suggestions(word):
    return getSynonyms(word)+getMisspeled(word)

if __name__ == '__main__':
    testres = sorted(getSynonyms("big"))
    for w in testres:
        print w