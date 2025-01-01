import os
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MemoryHandler:
    def __init__(self, memory_dir):
        """
        Initialize the MemoryHandler with a specific memory directory.
        The Word2Vec model will be saved in the conversation-specific subfolder.
        """
        self.memory_dir = memory_dir
        self.model_path = os.path.join(self.memory_dir, "word2vec.model")  # Path to save/load the Word2Vec model
        self.word2vec_model = None  # Word2Vec model instance
        self.load_or_train_word2vec_model()  # Load or train the Word2Vec model

    def load_or_train_word2vec_model(self):
        """
        Load the Word2Vec model if it exists in the memory directory, otherwise train a new one.
        """
        if os.path.exists(self.model_path):
            logging.info("Loading existing Word2Vec model.")
            self.word2vec_model = Word2Vec.load(self.model_path)
        else:
            logging.info("Training new Word2Vec model.")
            # Train a new Word2Vec model with default data
            sentences = [["default", "sentence", "for", "training"]]
            self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            self.word2vec_model.save(self.model_path)  # Save the model for future use
            logging.info(f"Word2Vec model saved at {self.model_path}")

    def train_word2vec(self, sentences):
        """
        Train the Word2Vec model on a list of sentences.
        """
        if not self.word2vec_model:
            logging.info("Initializing new Word2Vec model for training.")
            self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        else:
            logging.info("Updating existing Word2Vec model with new sentences.")
            self.word2vec_model.build_vocab(sentences, update=True)
            self.word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)
        self.word2vec_model.save(self.model_path)  # Save the updated model
        logging.info("Word2Vec model updated and saved.")

    def sentence_to_vec(self, sentence):
        """
        Convert a sentence to a vector using Word2Vec.
        If the sentence contains unknown words, it generates a random vector for them.
        """
        if not self.word2vec_model:
            logging.error("Word2Vec model is not loaded.")
            return np.zeros(100)  # Return a zero vector if the model is not loaded

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

    def calculate_similarity(self, vec1, vec2):
        """
        Calculate the cosine similarity between two vectors.
        """
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0  # Avoid division by zero
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_most_similar(self, query_embedding, embeddings_list):
        """
        Find the most similar embedding to the query embedding.
        """
        max_similarity = -1
        most_similar_index = -1
        for i, embedding in enumerate(embeddings_list):
            similarity = self.calculate_similarity(query_embedding, embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_index = i
        return most_similar_index, max_similarity