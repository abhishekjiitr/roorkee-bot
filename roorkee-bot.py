from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from FocusWords import *
import numpy as np
import pickle, os


lemmatizer = WordNetLemmatizer()

def most_common(lst):
	return max(set(lst), key=lst.count)

class VoteClassifier(ClassifierI):
	def __init__(self,*classifiers):
		self.classifiers=classifiers

	def classify(self,features):
		votes=[]
		for c in self.classifiers:
			v=c.classify(features)
			votes.append(v)
		#otes=nltk.FreqDist(votes)

		return most_common(votes)

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
stop_words=['a', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 'as', 'at', 'be', 'because', 'been', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', 'd', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'down', 'during', 'each', 'for', 'from', 'had', 'hadn', 'has', 'hasn', 'have', 'haven', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'i', 'if', 'in', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 'more', 'most', 'mustn', 'my', 'myself', 'needn', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', 'she', 'should', 'shouldn', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 'we', 'were', 'weren', 'will', 'with', 'won', 'wouldn', 'y', 'yo', 'your', 'yours', 'yourself', 'yourselves',',','(',')','[',']','.','{','}',':']
all_words=[]
for q in train:
	w=word_tokenize(q)
	res=w[0]
	l=[]
	l1=[]

	for word in w[3:]:
			#word=lemmatizer.lemmatize(word)
		if word not in stop_words and  word != '?' and check(word):
			l.append(word.lower())
			l1.append(word.lower()+w[0])


	# i=0
	# lt=[]
	# lt1=[]
	# while i < len(l)-1:
	# 	lt=lt+[l[i]+" "+l[i+1]]
	# 	lt1=lt1+[l[i]+" "+l[i+1]+w[0]]
	# 	i+=1
	# l=l+lt
	# l1=l1+lt1
	tr.append((l,res))
	all_words=all_words+l1
random.shuffle(tr)

def process(q):
	w=word_tokenize(q)
	l=[]
	for word in w:
		if word not in stop_words and  word != '?' and check(word):
			l.append(word.lower())
	# i=0
	# lt=[]
	# while i < len(l)-1:
	# 	lt=lt+[l[i]+" "+l[i+1]]
	# 	i+=1
	# l=l+lt
	return l

#print(tr[0:10])

all_words=nltk.FreqDist(all_words)


def feature(que):
	#tagged=nltk.pos_tag(que)
	feat={}
	#feat["Wh:"]=0
	for w in que:
		temp=max(all_words[w+"DESC"],all_words[w+"ABBR"],all_words[w+"HUM"],all_words[w+"NUM"],all_words[w+"LOC"],all_words[w+"ENTY"])
		if temp==all_words[w+'DESC']:
			feat[w]="DESC"
		elif temp==all_words[w+'ABBR']:
			feat[w]='ABBR'
		elif temp==all_words[w+'HUM']:
			feat[w]='HUM'
		elif temp==all_words[w+'NUM']:
			feat[w]='NUM'
		elif temp==all_words[w+'LOC']:
			feat[w]='LOC'
		elif temp==all_words[w+'ENTY']:
			feat[w]='ENTY'

	# 	if(w in ['how','what','why','where','who']):
	# 		feat["Wh:"]=w
	# feat["first:"]=que[0]

	return feat

feature_set=[]

for (q,res) in tr:
	feature_set.append((feature(q),res))

# print(feature_set[0])

training_set=feature_set[:3500]
test_set=feature_set[3500:]

if not os.path.exists("voted_classifier.p"):
	# classifier=nltk.NaiveBayesClassifier.train(training_set)
	#classifier.show_most_informative_features(10)

	#print("NaiveBayesClassifier Accuracy:",nltk.classify.accuracy(classifier,test_set))

	MClassifier=SklearnClassifier(MultinomialNB())
	MClassifier.train(training_set)

	# print("MultinomialNB Accuracy:",nltk.classify.accuracy(MClassifier,test_set))


	# BClassifier=SklearnClassifier(BernoulliNB())
	# BClassifier.train(training_set)

	# print("BernoulliNB Accuracy:",nltk.classify.accuracy(BClassifier,test_set))


	LogisticRegressionClassifier=SklearnClassifier(LogisticRegression())
	LogisticRegressionClassifier.train(training_set)

	# print("LogisticRegression Accuracy:",nltk.classify.accuracy(LogisticRegressionClassifier,test_set))

	SGDClassifier=SklearnClassifier(SGDClassifier())
	SGDClassifier.train(training_set)

	# print("SGDClassifier Accuracy:",nltk.classify.accuracy(SGDClassifier,test_set))

	# SVCClassifier=SklearnClassifier(SVC())
	# SVCClassifier.train(training_set)

	# print("SVC Accuracy:",nltk.classify.accuracy(SVCClassifier,test_set))

	LinearSVCClassifier=SklearnClassifier(LinearSVC())
	LinearSVCClassifier.train(training_set)

	# print("LinearSVC Accuracy:",nltk.classify.accuracy(LinearSVCClassifier,test_set))

	voted_classifier=VoteClassifier(MClassifier,SGDClassifier,LogisticRegressionClassifier,LinearSVCClassifier);

	#print(voted_classifier.classify(test_set[0][0]))

	print("voted_classifier Accuracy:",nltk.classify.accuracy(voted_classifier,test_set))
	pickle.dump( voted_classifier, open( "voted_classifier.p", "wb" ) )
else:
	voted_classifier = pickle.load( open( "voted_classifier.p", "rb" ) )

def processQuestion(que):
	q1=que.lower()
	w=word_tokenize(q1)
	l=[]
	for word in w:
		if word not in stop_words and  word != '?':
			l.append(word.lower())
	return l

with open('database.txt','r') as f:
	test=f.readlines()
dictionary=[]

for q in test:
	l = processQuestion(q)
	dictionary=dictionary+l

# dictionary2=nltk.FreqDist(dictionary)

d_index = {}
revDic = {}
g_index = 6
for key in dictionary:
	if key not in d_index:
		#print(key)
		d_index[key] = g_index
		revDic[g_index] = key
		g_index += 1


def sent2vec(que):
	temp=que
	que=getFocusWords(que)
	#print(que)
	
	fq=[0]*(g_index)
	for w in que:
		if(w in d_index):
			fq[d_index[w]]=1
			if not check(w):
				fq[d_index[w]] *= 1.5
	temp=process(temp)
	#print(temp)
	temp=feature(temp)
	#print(temp)
	typ=voted_classifier.classify(temp)
	#print(typ)
	if(typ=="ABBR"):
		fq[0]=1
	elif(typ=="ENTY"):
		fq[1]=1
	elif(typ=="HUM"):
		fq[2]=1
	elif(typ=="NUM"):
		fq[3]=1
	elif(typ=="DESC"):
		fq[4]=1
	elif(typ=="LOC"):
		fq[5]=1
	return fq

def dot(v1, v2):
	v1, v2 = map(np.array, (v1, v2) )
	res = v1 * v2
	val = sum(res)
	return val

def getAnswerIndex(q, data):
	maxi = -1
	answers = []
	q = np.array(q)
	for i in range(len(data)):
	   d = data[i]
	   d = np.array(d)
	   val = dot(q, d)
	   if val == maxi:
			answers.append(i)
	   elif val > maxi:
			maxi = val
			answers = [i]
	   #print(i,val)

	return answers if maxi > 1 else []


#print("$$$$$$$$$")

# print(answers)

print("Ask your query:")
# que = "Who is the director of IIT Roorkee"
que = raw_input()

data = []

def loadQuestionVectors():
	global data, test
	if not os.path.exists("sentence_vectors.p"):
		data = []
		test = [q.strip().replace('?', '') for q in test]

		for i in range(len(test)):
			q = test[i]
			temp=sent2vec(q)
			data.append(temp)
			print(i)
			print(len(temp))

		pickle.dump( data, open( "sentence_vectors.p", "wb" ) )
	else:
		data = pickle.load( open( "sentence_vectors.p", "rb" ) )

def addQuestion(que, ans):
	que = que.lower()
	loadQuestionVectors()
	global data
	global d_index
	global g_index
	global revDic
	# global data
	
	new_word_count = 0
	word_set = set(processQuestion(que))
	# print(word_set)
	for word in word_set:
		if word not in d_index:
			new_word_count += 1
	# print(new_word_count)
	# print(data[0])
	data = [v + new_word_count * [0] for v in data]
	# print(len(data[0]))

	qvector = sent2vec(que) + new_word_count * [1]
	# print(len(qvector))
	# print(qvector)
	data.append(qvector)
	pickle.dump( data, open( "sentence_vectors.p", "wb" ) )

	for key in word_set:
		if key not in d_index:
			d_index[key] = g_index
			revDic[g_index] = key
			g_index += 1
	with open('ans.txt','a+') as f:
		f.write(ans+"\n")
	with open('database.txt','a+') as f:
		f.write(que+"\n")

def getAnswer(question):
	# PREPROCESSING
	loadQuestionVectors()
	with open('ans.txt','r') as f:
		answer_list = f.readlines()
	answer_list = [ans.strip() for ans in answer_list]
	#######################################
	qvector = sent2vec(question)
	answers = getAnswerIndex(qvector, data)
	if len(answers) > 1:
		print("Possible Answers: \n")
		ans_vectors = [(dot(sent2vec(answer_list[index]), qvector), index) for index in answers]
		ans_vectors.sort( key = lambda x : x[0], reverse = True)

		# print(ans_vectors)
		final_answers = (map(lambda x : answer_list[ x[1] ], ans_vectors))
		# for i in range(len(answers)):
		# 	text = "Answer " + str(i+1) + ": "
		# 	text += answer_list[answers[i]]+"\n"
		# 	print(text)
		# print(final_answers)
		for i in range(len(final_answers)):
			text = "Answer " + str(i+1) + ": "
			text += final_answers[i]+"\n"
			print(text)
	elif len(answers) == 1:
		print("Answer:\n" + answer_list[answers[0]])
	else:
		print("No suitable answer found.")
	print("")
	print("If you are not satisfied with the answer, you can add this question.\nTo add this question, press y")
	choice = raw_input().strip()
	if choice in ['y', 'Y']:
		ans = raw_input("Please input the corresponding answer:\n")
		addQuestion(question, ans)
		print("Database Updated")
getAnswer(que)	
#addQuestion("Who is the director of IIT Roorkee", "A.K. Ghosh, (Previously in IIT Kanpur)")
# print("")