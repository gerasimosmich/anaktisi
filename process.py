# -*- coding: utf-8 -*-

import os
import json
import string
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
from collections import Counter, defaultdict

# Κατεβάστε αν χρειάζεται τα δεδομένα του nltk
nltk.download('all')

# ΕΡΩΤΗΜΑΤΑ 1 ΚΑΙ 2
# Path για τα δεδομένα
dataset_path = './archive/bbc'
output_json_path = './bbc_dataset.json'

# Ανάγνωση JSON ή δημιουργία αν δεν υπάρχουν
if os.path.exists(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as json_file:
        documents = json.load(json_file)
else:
    documents = []
    # Επεξεργασία TXT αρχείων και μετατροπή σε JSON
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):  # Ελέγχει αν είναι φάκελος
            for file_name in os.listdir(category_path):
                file_path = os.path.join(category_path, file_name)
                if os.path.isfile(file_path):  # Ελέγχει αν είναι αρχείο
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        # Πρώτη γραμμή ως τίτλος, υπόλοιπο ως περιεχόμενο
                        title = lines[0].strip() if lines else "No Title"
                        content = " ".join(line.strip() for line in lines[1:])
                        documents.append({
                            "category": category,
                            "title": title,
                            "content": content
                        })
    # Αποθήκευση στο JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(documents, json_file, indent=4, ensure_ascii=False)

# Προεπεξεργασία δεδομένων
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Καθαρισμός ειδικών χαρακτήρων
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text, language='english')
    
    # Αφαίρεση stop words
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming και Lemmatization
    stemmed = [stemmer.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    return {
        "original": tokens,
        "stemmed": stemmed,
        "lemmatized": lemmatized
    }

# Εφαρμογή προεπεξεργασίας σε κάθε έγγραφο
for document in documents:
    processed_content = preprocess_text(document['content'])
    document['processed_content'] = processed_content

# Αποθήκευση προεπεξεργασμένων δεδομένων
preprocessed_json_path = './bbc_preprocessed_dataset.json'
with open(preprocessed_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(documents, json_file, indent=4, ensure_ascii=False)

print("The preprocessed dataset has been saved at {}".format(preprocessed_json_path))


# ΕΡΩΤΗΜΑ 3
# Διαδρομή στα δεδομένα
inverted_index_output_path = './inverted_index.json'

# Φόρτωση επεξεργασμένων δεδομένων
with open(preprocessed_json_path, 'r', encoding='utf-8') as json_file:
    documents = json.load(json_file)

# Δημιουργία ανεστραμμένου ευρετηρίου
inverted_index = defaultdict(list)

# Κατασκευή ευρετηρίου
for doc_id, document in enumerate(documents):
    words = document['processed_content']['lemmatized']  # Χρησιμοποιούμε τη lemmatized μορφή
    for word in set(words):  # Χρησιμοποιούμε set για να αποφύγουμε διπλοεγγραφές λέξεων
        inverted_index[word].append(doc_id)

# Αποθήκευση του ευρετηρίου σε αρχείο JSON
with open(inverted_index_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(inverted_index, json_file, indent=4, ensure_ascii=False)

print("Το ανεστραμμένο ευρετήριο αποθηκεύτηκε στο {}".format(inverted_index_output_path))


# ΕΡΩΤΗΜΑ 4
def process_query(query, inverted_index):
    """
    Επεξεργασία του ερωτήματος και επιστροφή σχετικών εγγράφων.
    Υποστηρίζει απλές λέξεις-κλειδιά και Boolean queries.
    """
    # Χωρισμός ερωτήματος σε λέξεις
    terms = query.lower().split()
    
    # Ανάλυση Boolean Queries
    if 'and' in terms or 'or' in terms or 'not' in terms:
        result_set = set()
        current_set = set()
        operator = None
        
        for term in terms:
            if term in ['and', 'or', 'not']:
                operator = term
            else:
                # Λήψη των εγγράφων από το ευρετήριο
                term_docs = set(inverted_index.get(term, []))
                
                if operator == 'not':
                    term_docs = set(range(len(documents))) - term_docs
                
                if not result_set:
                    result_set = term_docs
                else:
                    if operator == 'and':
                        result_set &= term_docs
                    elif operator == 'or':
                        result_set |= term_docs
        return list(result_set)
    
    # Απλή αναζήτηση (λέξεις-κλειδιά)
    else:
        result_docs = set()
        for term in terms:
            result_docs.update(inverted_index.get(term, []))
        return list(result_docs)

# Φόρτωση του ευρετηρίου
with open('./inverted_index.json', 'r', encoding='utf-8') as json_file:
    inverted_index = json.load(json_file)

# Παράδειγμα λειτουργίας
query = input("Δώσε το ερώτημα σου (Boolean ή απλές λέξεις-κλειδιά): ")
matching_docs = process_query(query, inverted_index)

# Εμφάνιση αποτελεσμάτων
if matching_docs:
    print("Βρέθηκαν {} σχετικά έγγραφα:". format(len(matching_docs)))
    for doc_id in matching_docs:
        print("- {} (Category: {})".format(documents[doc_id]['title'], documents[doc_id]['category']))

else:
    print("Δεν βρέθηκαν σχετικά έγγραφα.")

# ΕΡΩΤΗΜΑ 5
# Υπολογισμός TF
def compute_tf(term, document):
    tokens = document['processed_content']['lemmatized']
    term_count = tokens.count(term)
    return term_count / len(tokens) if tokens else 0

# Υπολογισμός IDF
def compute_idf(term, inverted_index, total_documents):
    doc_count = len(inverted_index.get(term, []))
    return math.log((1 + total_documents) / (1 + doc_count))

# Υπολογισμός TF-IDF για ένα ερώτημα
def rank_documents(query, documents, inverted_index):
    total_documents = len(documents)
    query_terms = query.lower().split()
    scores = defaultdict(float)

    # Υπολογισμός IDF για κάθε όρο του ερωτήματος
    idf_values = {term: compute_idf(term, inverted_index, total_documents) for term in query_terms}

    # Υπολογισμός TF-IDF για κάθε έγγραφο
    for term in query_terms:
        for doc_id in inverted_index.get(term, []):
            tf = compute_tf(term, documents[doc_id])
            scores[doc_id] += tf * idf_values[term]

    # Ταξινόμηση εγγράφων με βάση το σκορ TF-IDF
    ranked_documents = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked_documents

# Παράδειγμα εκτέλεσης
query = input("Δώσε το ερώτημά σου (λέξεις-κλειδιά): ")

# Βαθμολόγηση εγγράφων με βάση το ερώτημα
ranked_docs = rank_documents(query, documents, inverted_index)

# Εμφάνιση αποτελεσμάτων
if ranked_docs:
    print("Βρέθηκαν {} σχετικά έγγραφα με κατάταξη:".format(len(ranked_docs)))
    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
        print("{}. {} (Category: {}, Score: {:.4f})".format(rank, documents[doc_id]['title'], documents[doc_id]['category'], score))
else:
    print("Δεν βρέθηκαν σχετικά έγγραφα.")