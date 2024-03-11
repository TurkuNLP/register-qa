from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import sys
import json
import random

print(sys.argv)
input = sys.argv[1]
amount = int(sys.argv[2])
topics = int(sys.argv[3])

stop = set(stopwords.words('finnish'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(lemma.lemmatize(ch) for ch in stop_free if ch not in exclude)
    return punc_free

with open(input, 'r') as f:
    lines = f.readlines()
f.close()


data = []
for line in lines:
    t = json.loads(line)["qa"]
    data.append(" ".join(t.values()))
    
random.shuffle(data)
data = data[0:amount]

doc_clean = [clean(doc).split() for doc in data]

dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

from pprint import pprint
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=topics, id2word =dictionary, passes=30)
pprint(ldamodel.print_topics(num_topics=topics, num_words =5))
print("")
#print("perpelixity...?")
#print(ldamodel.log_perplexity(data))

