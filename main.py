from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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

vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3))
docVector = vectorizer.fit_transform(documents)

vocabulary = vectorizer.vocabulary_
tfidfTK = vectorizer.get_feature_names_out()

tfidfMX = pd.DataFrame(
    data=docVector.toarray(),
    index=['Doc1', 'Doc2', 'Doc3'],
    columns=tfidfTK,
)

for doc_id, document in enumerate(documents):
    docsCol.update_one(
        {"_id": doc_id + 1},
        {"$set": {"content": document}},
        upsert=True
    )

for term, position in vocabulary.items():
    docsT = [
        {
            "doc_id": doc_id + 1,
            "tf_idf": tfidfMX.at[
                f"Doc{doc_id+1}", term
            ],
        }
        for doc_id in range(len(documents))
        if tfidfMX.at[f"Doc{doc_id+1}", term] > 0
    ]
    termsCol.update_one(
        {"_id": position},
        {"$set": {"pos": position, "docs": docsT}},
        upsert=True,
    )

for query_index, query in enumerate(queries):
    print(f"Query {query_index + 1}: '{query}'")

    query_vector = vectorizer.transform([query])

    docsRel = []
    for term in vectorizer.get_feature_names_out():
        term_doc = termsCol.find_one({"_id": vocabulary[term]})
        if term_doc:
            docsRel += term_doc["docs"]

    cosine_sim = cosine_similarity(query_vector, docVector)
    scores = [
        (doc_index, similarity)
        for doc_index, similarity in enumerate(cosine_sim[0])
        if similarity > 0
    ]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for doc_index, similarity in scores:
        print(f"{documents[doc_index]} \nSimilarity: {similarity:.3f}")
    print()
