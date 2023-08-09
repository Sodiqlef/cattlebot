from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required resources (only run once)
nltk.download('punkt')
nltk.download('stopwords')

# Set up NLTK
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words


def train_word2vec_model(sentences):
    preprocessed_sentences = [word_tokenize(
        sentence.lower()) for sentence in sentences]
    preprocessed_sentences = [[word for word in sent if word.isalnum(
    ) and word not in stop_words] for sent in preprocessed_sentences]

    model = Word2Vec(sentences=preprocessed_sentences,
                     vector_size=100, window=5, min_count=1, sg=0)
    return model


def save_word2vec_model(model, path):
    model.save(path)


def load_word2vec_model(path):
    return Word2Vec.load(path)
