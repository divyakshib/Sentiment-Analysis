import numpy as np
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize import RegexpTokenizer
with open('angry.txt', 'r') as f:
    angry = f.read()
with open('happy.txt', 'r') as f:
    happy = f.read()
with open('sad.txt', 'r') as f:
    sad = f.read()
with open('kind.txt', 'r') as f:
    kind = f.read()
with open('nervous.txt', 'r') as f:
    nervous = f.read()
with open('shy.txt', 'r') as f:
    shy = f.read()
with open('test.txt', 'r') as f:
    test = f.read()
sw=set(stopwords.words("english"))
def remove_stopwords(words):
    wor=[w for w in words if w not in sw]
    return(wor)
def tokenizer_(text):
    text=text.lower()
    token=RegexpTokenizer("[a-zA-Z]+")
    word=token.tokenize(text)
    word_list=remove_stopwords(word)
    #gimme=ngrams(word_list,2)
    #ps=PorterStemmer()
    #tokenized=[]
    #tokenized=[ps.stem(w) for w in word_list if ps.stem(w) not in tokenized ]
    #cv = CountVectorizer(tokenizer=tokenizer_, ngram_range=(1, 2))
    #vector = cv.fit_transform(tokenized).todense()
    #vc=vectorised_
    #length = sum(1 for el in gimme())
    #my_array = np.empty(length)
    #for i, el in enumerate(gimme()): my_array[i] = el
    return (word_list)
angry=tokenizer_(angry)
sad=tokenizer_(sad)
nervous=tokenizer_(nervous)
happy=tokenizer_(happy)
kind=tokenizer_(kind)
shy=tokenizer_(shy)
c1=len(angry)
c2=len(angry)+len(sad)
c3=len(angry)+len(sad)+len(happy)
c4=len(shy)+len(angry)+len(sad)+len(happy)
c5=len(angry)+len(sad)+len(happy)+len(shy)+len(kind)
c6=len(angry)+len(sad)+len(happy)+len(shy)+len(nervous)+len(kind)
one=np.ones(c6)#1-Angry
one[:c2]*=2#sad
one[c2:c3]*=3#happy
one[c3:c4]*=4#shy
one[c4:c5]*=5#kind
one[c5:]*=6#nervous
vocab_to_int = {i: w for i,w in enumerate(angry+sad+happy+shy+kind+nervous, 1)}
def text_tokenize(text):
    test=tokenizer_(text)
    int=[key for key,word in vocab_to_int.items() if word in test]
    int=np.array(int)
    return int
X_train=np.array(list(vocab_to_int.keys()))
X_train=X_train.reshape(-1,1)
Y_train=one
X_test=text_tokenize(test).reshape(-1,1)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def prediction(X_train,Y_train,X_test):
    nb=GaussianNB()
    nb.fit(X_train,Y_train)
    Y_pred=nb.predict(X_test)
    Pred=np.unique(Y_pred,return_counts=True)
    Class_=np.argmax(Pred[0])+1
    return Class_
def print_RES(r):
    if r==1:
        print("Angry : Negative")
    elif r==2:
        print("Sad : Negative")
    elif r==3:
        print("Happy : Positive")
    elif r==4:
        print("Shy : Neutral")
    elif r==5:
        print("Kind : Positive")
    else:
        print("Nervous: Negative")
RESULT=prediction(X_train,Y_train,X_test)
print_RES(RESULT)