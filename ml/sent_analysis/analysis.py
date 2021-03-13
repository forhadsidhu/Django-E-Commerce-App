import  pickle
import  pandas as pd
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

stopwords = set(STOPWORDS)
stopwords.remove("not")
from django.conf import settings
BASE_DIR = settings.BASE_DIR

def predict(sample):

    # load the vectorizer
    loaded_vectorizer = pickle.load(open(BASE_DIR+'\\ml\\sent_analysis\\model\\vectorizer.pickle', 'rb'))
    # load the model
    loaded_model = pickle.load(open(BASE_DIR+'\\ml\\sent_analysis\\model\\classification.model', 'rb'))


    # make a prediction
    output=loaded_model.predict(loaded_vectorizer.transform([sample]))
    output=output.astype(str)
    return output



# if __name__ == '__main__':
#     predict("product is not so good")
#     predict("what a product")
