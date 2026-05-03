import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Η διαδρομή που βρίσκεται η βάση μας
CHROMA_PATH = "chroma_db"

def search_medical_guidelines(query_text):
    print(f"🔍 Ερώτηση: '{query_text}'\n")
    
    # 1. Φορτώνουμε το ΙΔΙΟ μοντέλο ενσωμάτωσης που χρησιμοποιήσαμε στο χτίσιμο
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Συνδεόμαστε στην υπάρχουσα βάση δεδομένων
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    # 3. Εκτελούμε Αναζήτηση Ομοιότητας (Similarity Search)
    # Το k=2 σημαίνει ότι θέλουμε το σύστημα να μας φέρει τα 2 πιο σχετικά chunks
    results = db.similarity_search_with_score(query_text, k=2)
    
    if len(results) == 0:
        print("Δεν βρέθηκαν σχετικά αποτελέσματα.")
        return

    print("--- Αποτελέσματα Ανάκτησης (RAG) ---")
    for i, (doc, score) in enumerate(results):
        # Στο ChromaDB, το score υπολογίζει την μαθηματική "απόσταση" (L2 distance). 
        # Όσο πιο ΜΙΚΡΟ είναι το νούμερο, τόσο πιο κοντά είναι η απάντηση στην ερώτηση.
        print(f"\n[Αποτέλεσμα {i+1} | Μαθηματική Απόσταση: {score:.4f}]")
        print(f"Κείμενο: {doc.page_content}")

if __name__ == "__main__":
    # Το test μας: Ρωτάμε ποιες είναι οι αντενδείξεις φαρμάκων στην εγκυμοσύνη
    test_query = "What medications are contraindicated during pregnancy?"
    
    search_medical_guidelines(test_query)