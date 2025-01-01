import os
import logging
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MemoryHandler:
    def __init__(self, memory_dir):
        self.memory_dir = memory_dir
        self.model_path = os.path.join(self.memory_dir, "word2vec.model")
        self.word2vec_model = None
        self.load_or_train_word2vec_model()

    def load_or_train_word2vec_model(self):
        """
        Load the Word2Vec model if it exists, otherwise train a new one.
        """
        if os.path.exists(self.model_path):
            logging.info("Loading existing Word2Vec model.")
            self.word2vec_model = Word2Vec.load(self.model_path)
        else:
            logging.info("Training new Word2Vec model.")
            # Train a new Word2Vec model with default data
            sentences = [["default", "sentence", "for", "training"]]
            self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            self.word2vec_model.save(self.model_path)

    def sentence_to_vec(self, sentence):
        """
        Convert a sentence to a vector using Word2Vec.
        Ensure an embedding is always generated, even for short or single-word messages.
        """
        if not self.word2vec_model:
            logging.error("Word2Vec model is not loaded.")
            return None

        try:
            # Preprocess the sentence into words
            words = simple_preprocess(sentence)
            if not words:  # If no words are found, use the entire sentence as a single word
                words = [sentence.strip()]

            # Generate vectors for each word in the sentence
            vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    vectors.append(self.word2vec_model.wv[word])
                else:
                    # If the word is not in the vocabulary, generate a random vector
                    vectors.append(np.random.rand(self.word2vec_model.vector_size))

            if vectors:
                # Return the average of all word vectors
                return np.mean(vectors, axis=0)
            else:
                # If no valid words are found, return a zero vector
                logging.warning("No valid words found in the sentence for Word2Vec. Returning a zero vector.")
                return np.zeros(self.word2vec_model.vector_size)
        except Exception as e:
            logging.error(f"Error converting sentence to vector: {str(e)}")
            return np.zeros(self.word2vec_model.vector_size)