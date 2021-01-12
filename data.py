import sqlite3
import numpy as np
import pandas as pd
import re
import string
import nltk
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

con = sqlite3.connect('database.sqlite')
filtered_data = pd.read_sql_query("""
SELECT * FROM Reviews WHERE Score !=3
""",con)

def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'
actualscore = filtered_data['Score']
posneg = actualscore.map(partition)
filtered_data['Score'] = posneg

sorted_data = filtered_data.sort_values('ProductId',axis=0,ascending=True,kind="quicksort")
final = sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final = final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english')

def cleanhtml(sentence):
    cleanr = re.compile('<,*?>')
    cleantext = re.sub(cleanr, '',sentence)
    return cleantext
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|*|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r'',cleaned)
    return cleaned

i=0
strl = ''
final_string = []
all_positive_words = []
all_negative_words = []
s = ''
for sent in final['Text'].values:
    filtered_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower()))
                    filtered_sentence.append(s)
                    if(final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s)
                    if (final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    strl = ' '.join(filtered_sentence)

    final_string.append(strl)
    i+=1
final['cleanedText']= final_string
print(final.head())
conn = sqlite3.connect('final.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews',conn, schema=None, if_exists='replace',chunksize=None,method=None)