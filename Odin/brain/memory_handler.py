import os
import datetime
import sqlite3
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MemoryHandler:
    def __init__(self, memory_dir):
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        self.model_path = os.path.join(self.memory_dir, "word2vec.model")
        self.word2vec_model = self.load_or_train_word2vec_model()

    def load_or_train_word2vec_model(self):
        """
        Load the Word2Vec model if it exists. If not, train a new model with meaningful data.
        """
        if os.path.exists(self.model_path):
            logging.info("Loading existing Word2Vec model.")
            return Word2Vec.load(self.model_path)
        else:
            # Train a new model with meaningful data
            meaningful_data = [
                "Initialize Word2Vec model with meaningful data",
                "This ensures the model generates meaningful embeddings",
                "Word2Vec embeddings are used for semantic understanding",
                "The model learns from sentences and generates vectors",
                "Meaningful embeddings are essential for AI applications"
            ]
            sentences = [simple_preprocess(sentence) for sentence in meaningful_data]
            self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            self.word2vec_model.save(self.model_path)
            logging.info("Word2Vec model initialized with meaningful data and saved.")
            return self.word2vec_model

    def sentence_to_vec(self, sentence):
        """
        Convert a sentence to a vector using the Word2Vec model.
        If the model cannot generate an embedding, retrain the model dynamically.
        """
        if self.word2vec_model is None:
            logging.error("Word2Vec model is not initialized.")
            raise ValueError("Word2Vec model is not initialized.")

        words = simple_preprocess(sentence)
        if not words:
            logging.warning("No words found in the sentence.")
            raise ValueError("No words found in the sentence.")

        # Try to generate embeddings
        try:
            word_vecs = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
            if not word_vecs:
                logging.warning(f"No word vectors found for sentence: {sentence}")
                # Retrain the model dynamically with the new sentence
                self.update_word2vec_model(sentence)
                word_vecs = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
                if not word_vecs:
                    raise ValueError(f"Unable to generate embeddings for sentence: {sentence}")
            return list(np.mean(word_vecs, axis=0))
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise ValueError(f"Error generating embeddings: {str(e)}")

    def init_db(self, db_path):
        """
        Initialize the SQLite database with the necessary table.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                message TEXT,
                embedding TEXT
            )''')
        conn.commit()
        conn.close()

    def save_conversation_to_db(self, db_path, message, embedding, timestamp):
        """
        Save the conversation to the database, including the timestamp, message, and embedding.
        """
        # Ensure embedding is not None
        if embedding is None:
            raise ValueError("Embedding cannot be None.")

        # Convert the embedding list to a string
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        
        logging.info(f"Saving conversation to database: Timestamp={timestamp}, Message={message}, Embedding={embedding_str}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversations (timestamp, message, embedding) VALUES (?, ?, ?)", 
                       (timestamp, message, embedding_str))
        conn.commit()
        conn.close()

        # Update the Word2Vec model with the new message
        self.update_word2vec_model(message)

    def update_word2vec_model(self, message):
        """
        Update the Word2Vec model with the new message.
        """
        if self.word2vec_model is None:
            logging.error("Word2Vec model is not initialized.")
            raise ValueError("Word2Vec model is not initialized.")

        # Preprocess the message into words
        words = simple_preprocess(message)
        if not words:
            logging.warning("No words found in the message for updating the Word2Vec model.")
            return

        # Update the Word2Vec model with the new words
        self.word2vec_model.build_vocab([words], update=True)
        self.word2vec_model.train([words], total_examples=1, epochs=1)
        self.word2vec_model.save(self.model_path)
        logging.info("Word2Vec model updated with new message.")

    def train_word2vec_model(self, db_path):
        """
        Train the Word2Vec model using the conversation messages stored in the database.
        """
        logging.info("Training Word2Vec model...")
        sentences = []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT message FROM conversations")
        rows = cursor.fetchall()
        for row in rows:
            message = row[0].split("\n")  # Split by newline to separate User and AI messages
            for part in message:
                part = part.strip()
                if part.startswith("User: "):
                    part = part.replace("User: ", "").strip()
                elif part.startswith("AI: "):
                    part = part.replace("AI: ", "").strip()
                sentences.append(simple_preprocess(part))
        conn.close()

        if sentences:
            self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            self.word2vec_model.save(self.model_path)
            logging.info("Word2Vec model trained and saved.")
        else:
            logging.info("No messages available to train Word2Vec model.")