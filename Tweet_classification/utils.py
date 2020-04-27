import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;:$!]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_?&]')
STOPWORDS = set(stopwords.words('english'))
single_chars = re.compile(r'\s+[a-zA-Z]\s+')

def clean_text(text: str)-> str:
    """
    Preprocesses text and returns a cleaned
    piece of text with unwanted characters removed
    
    Args:
       text: a string
    Returns: 
        Preprocessed text
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwords from text
    text = single_chars.sub('', text) #remove single-characters
    return text

def remove_emoji(text: str)-> str:
    """
    Remove emojis from text.
    Args:
        `text`: a string
    Returns:
        clean text with no emojis.
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


def remove_URL(text: str)-> str:
    """
    Removes URL patterns from text.
    Args:
        `text`: A string, word/sentence
    Returns:
        Text without url patterns.
    """
    url = re.compile('https?://\S+|www\.\S+')
    text = url.sub('',text)
    return text
 
    
def remove_html(text)-> str:
    """
    Removes html tags from text.
    Args:
        `text`: A string, word/sentence
    Returns:
        Text without html tags.
    """
    html = re.compile('<.*?>')
    text = html.sub('',text)
    return text


def count_vectorize(data: str) -> [int]:
    """
    Create word vectors/tokens for input data
    Args:
        `data`: text to be vectorized
    """
    count_vectorizer = CountVectorizer(min_df=3, analyzer='word',
                                       ngram_range=(1, 2)) #TfidfVectorizer()
    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


def rmse(y_true, y_pred)-> int:
    """
    Computes the root_mean_squared error between
    a true values and predicted values.
    Args:
        `y_true`: true values
        `y_pred`: predicted values
    Returns:
        The Root Mean Squared error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))