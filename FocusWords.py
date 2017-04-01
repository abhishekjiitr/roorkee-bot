from StanfordPOS import *
from mydictionary import *
from suggestions import *
from nltk.tokenize import word_tokenize

def getFocusWords(sentence):
    wordt=word_tokenize(sentence)
    res = getPOSFocus(sentence)
    ans = []
    for word in res:
        # print("$$$$$$  " +word+" $$$$$$")
        similar_words = suggestions(word)
        similar_words = similar_words + [word]
        # print(similar_words)
        ans = ans + similar_words
    # print "yo"
    for word in wordt:
        if not check(word) and word not in ans:
            ans.append(word)
    ans = [word.lower() for word in ans]
    return ans 

if __name__ == '__main__':
    testSentence = "Do I have to pass all the courses that are assigned to me ?"
    focusW = getFocusWords(testSentence)
    for w in focusW:
        print w