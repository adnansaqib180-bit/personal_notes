import subprocess
import string
try :
    import nltk
except:
    subprocess.run('pip install nltk')
from nltk.corpus import stopwords
nltk.download('punnk_tab')
nltk.download('stopwords')
stop_words = stopwords.words('english')
text = 'hello how are you my name is adnan saqib i read in class 11 th currently doing intermediate computer science and i am living in rawalpindi a city of pakistan '
tokens = nltk.word_tokenize(text)   
cleaned_text = [word for word in tokens if word not in string.punctuation and word not in stop_words]
print(cleaned_text)
