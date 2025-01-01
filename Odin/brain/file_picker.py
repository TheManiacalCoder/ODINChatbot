import os
import tkinter as tk
from tkinter import filedialog
import logging
import chardet
import PyPDF2
import docx
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FilePicker:
    def __init__(self, conversation_manager):
        self.conversation_manager = conversation_manager

    def pick_file(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a file to upload")
        if file_path:
            self.process_file(file_path)

    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    def read_file_content(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
            return content
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            return content
        elif file_extension == '.txt':
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        elif file_extension == '.csv':
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.reader(file)
                content = "\n".join([",".join(row) for row in reader])
            return content
        elif file_extension in ['.py', '.js', '.java', '.html', '.css', '.cpp', '.c', '.sh', '.sql']:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def chunk_content(self, content):
        """
        Split the content into logical sections based on line breaks.
        """
        lines = content.split('\n')
        chunks = []
        current_chunk = ""

        for line in lines:
            if line.strip():  # If the line is not empty
                current_chunk += line + "\n"
            else:  # If the line is empty, finalize the current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = ""

        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def process_file(self, file_path):
        """
        Process the file as a single, unified context.
        """
        try:
            content = self.read_file_content(file_path)
            file_name = os.path.basename(file_path)

            # Split the content into chunks (if needed for processing)
            chunks = self.chunk_content(content)

            # Concatenate all chunks into a single context
            full_context = "\n".join(chunks)

            # Clear previous file chunks and store the full context
            self.conversation_manager.file_chunks = [full_context]

            # Train Word2Vec on the full context
            self.conversation_manager.memory_handler.train_word2vec([full_context])

            # Assign an agentic role to the file
            agentic_role = f"Agentic Role for {file_name}"
            self.conversation_manager.set_agentic_role(agentic_role)

            # Save the full context to the database
            self.conversation_manager.save_file_chunks(file_name, [full_context])

            logging.info(f"File '{file_name}' uploaded and processed as a single context.")
            self.conversation_manager.notify(f"File '{file_name}' has been fully uploaded and processed.")
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            self.conversation_manager.notify(f"Error processing file: {str(e)}")