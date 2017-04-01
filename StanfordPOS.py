from nltk.tag.stanford import StanfordPOSTagger
import os

path_to_model = os.path.join ( os.getcwd(), "StanfordNLP/pos/models/english-bidirectional-distsim.tagger")
path_to_jar = os.path.join ( os.getcwd(),"StanfordNLP/pos/stanford-postagger.jar")
POStagger=StanfordPOSTagger(path_to_model, path_to_jar)
POStagger.java_options='-mx4096m'          ### Setting higher memory limit for long sentences
from nltk.tokenize import word_tokenize

def getPOSFocus(sentence):
    sentence = sentence.lower()
    result = []
    tagged = POStagger.tag(word_tokenize(sentence))
    for (word, cat) in tagged:
        if cat.startswith("NN") or cat.startswith("JJ"):
            result.append(word.lower())
    return result

if __name__ == '__main__':
    text = "Where is HEC( Himalayan Explorers Club ) office ?"
    print getPOSFocus(text)