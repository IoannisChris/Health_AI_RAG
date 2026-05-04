import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Ορίζουμε τις διαδρομές των αρχείων (paths)
DATA_PATH = "data/hypertension_guidelines.txt"
CHROMA_PATH = "chroma_db"

def build_vector_database():
    print("1. Φόρτωση του ιατρικού εγγράφου...")
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()

    print("2. Τεμαχισμός (Chunking) του κειμένου...")
    # Χωρίζουμε το κείμενο σε chunks των 400 χαρακτήρων. 
    # Το overlap (50) διασφαλίζει ότι δεν θα κοπεί μια ιατρική πρόταση στη μέση χάνοντας το νόημα (context).
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0, # Δεν χρειαζόμαστε overlap αν κόβουμε σε φυσικές παραγράφους
        separators=["\n\n", "\n", " ", ""] 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--> Δημιουργήθηκαν {len(chunks)} αυτόνομα chunks.")

    print("3. Φόρτωση του μοντέλου Ενσωμάτωσης (Embedding Model)...")
    # Χρησιμοποιούμε το all-MiniLM-L6-v2. Είναι μικρό, γρήγορο και εκπαιδευμένο να βρίσκει σημασιολογικές ομοιότητες.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("4. Αποθήκευση στη Vector Database (ChromaDB)...")
    # Μετατρέπουμε τα chunks σε διανύσματα και τα γράφουμε στον δίσκο.
    db = Chroma.from_documents(
        chunks, 
        embedding_model, 
        persist_directory=CHROMA_PATH
    )
    
    print("✅ Επιτυχία! Η βάση διανυσμάτων χτίστηκε και αποθηκεύτηκε στο φάκελο 'chroma_db'.")

if __name__ == "__main__":
    build_vector_database()