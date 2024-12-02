from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

DB_NAME = "websearch4"
DB_HOST = "localhost"
DB_PORT = 27017

client = MongoClient(DB_HOST, DB_PORT)
db = client[DB_NAME]
termsCol = db["terms"]
docsCol = db["documents"]

documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The medication caused a headache and nausea, but no dizziness was reported.",
    "Headache and dizziness are common effects of this medication.",
]

queries = [
    "nausea and dizziness",
    "effects",
    "nausea was reported",
    "dizziness",
    "the medication",
]
