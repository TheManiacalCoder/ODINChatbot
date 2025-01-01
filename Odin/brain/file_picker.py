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
        """
        Open a file dialog to select a file and process it.
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a file to upload")
        if file_path:
            self.process_file(file_path)

    def detect_encoding(self, file_path):
        """
        Detect the encoding of a file.
        """
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    def read_file_content(self, file_path):
        """
        Read the content of a file based on its extension.
        """
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

    def chunk_content(self, content, chunk_size=500):
        """
        Split the content into chunks of a specified size.
        """
        words = content.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def process_file(self, file_path):
        """
        Process the file content, split it into chunks, and save each chunk to the database.
        """
        try:
            # Read the file content
            content = self.read_file_content(file_path)
            file_name = os.path.basename(file_path)

            # Split the content into chunks
            chunks = self.chunk_content(content)

            # Save each chunk to the database with a special role
            for chunk in chunks:
                self.conversation_manager.save_conversation(
                    chunk,
                    role=f"file_chunk:{file_name}"  # Use a special role to identify file chunks
                )

            logging.info(f"File '{file_name}' processed and stored in memory.")
            self.conversation_manager.notify(f"File '{file_name}' has been uploaded and processed.")
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            self.conversation_manager.notify(f"Error processing file: {str(e)}")