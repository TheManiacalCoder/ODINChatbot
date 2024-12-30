import os
import datetime
import numpy as np
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI as Client
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import sqlite3
from .memory_handler import MemoryHandler
from .conversation_manager import ConversationManager

conversation_manager = ConversationManager()