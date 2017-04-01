from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI


class VoteClassifier(ClassifierI):
	def __init__(self,*classifiers):
		self.classifiers=classifiers

	def classify(self,features):
		votes=[]
		for c in self.classifiers:
			v=c.classify(features)
			votes.append(v)

		votes=nltk.FreqDist(votes)

		return list(votes)[0]

	def confidence(self,features):
		votes=[]
		for c in self.classifiers:
			v=c.classify(features)
			votes.append(v)
		choice_votes=votes.count(mode(votes))
		conf=choice_votes/len(votes)
		return conf

f=open('train.txt','r')
train=f.readlines()
f.close()

tr=[]
stop_words=set(stopwords.words("english"))
all_words=[]

for q in train:
	w=word_tokenize(q)
	res=w[0]
	l=[]
	for word in w[3:]:
		if word not in stop_words and  word != '?':
			l.append(word.lower())

	tr.append((l,res))
	all_words=all_words+l

random.shuffle(tr)


all_words=nltk.FreqDist(all_words)


def feature(que):
	#tagged=nltk.pos_tag(que)
	feat={}
	for w in que:
		feat[w]=all_words[w]
	feat["first:"]=que[0]

	return feat

feature_set=[]

print tr[0]

for (q,res) in tr:
	feature_set.append((feature(q),res))

print feature_set[0]

training_set=feature_set[:3500]
test_set=feature_set[3500:]


print(feature(tr[0][0]))

classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Original Accuracy:",nltk.classify.accuracy(classifier,test_set))
#classifier.show_most_informative_features(10)

MClassifier=SklearnClassifier(MultinomialNB())
MClassifier.train(training_set)

print("Multi Accuracy:",nltk.classify.accuracy(MClassifier,test_set))


BClassifier=SklearnClassifier(BernoulliNB())
BClassifier.train(training_set)

print("Bernoulli Accuracy:",nltk.classify.accuracy(BClassifier,test_set))


LogisticRegressionClassifier=SklearnClassifier(LogisticRegression())
LogisticRegressionClassifier.train(training_set)

print("LogisticRegression Accuracy:",nltk.classify.accuracy(LogisticRegressionClassifier,test_set))

SGDClassifier=SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)

print("SGDClassifier Accuracy:",nltk.classify.accuracy(SGDClassifier,test_set))

SVCClassifier=SklearnClassifier(SVC())
SVCClassifier.train(training_set)

print("SVC Accuracy:",nltk.classify.accuracy(SVCClassifier,test_set))

LinearSVCClassifier=SklearnClassifier(LinearSVC())
LinearSVCClassifier.train(training_set)

print("LinearSVC Accuracy:",nltk.classify.accuracy(LinearSVCClassifier,test_set))

voted_classifier=VoteClassifier(BClassifier,SVCClassifier,LogisticRegressionClassifier);

#print(voted_classifier.classify(test_set[0][0]))
print("voted_classifier Accuracy:",nltk.classify.accuracy(voted_classifier,test_set))