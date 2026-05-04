import os
import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Κρύβουμε τα warnings για να έχουμε καθαρό τερματικό κατά την αξιολόγηση
warnings.filterwarnings("ignore")

CHROMA_PATH = "chroma_db"

# 1. Το "Ground Truth" (Η πραγματικότητα)
# Αυτό είναι το test set μας. Ορίζουμε ερωτήσεις και τις λέξεις-κλειδιά που ΑΠΑΙΤΟΥΜΕ 
# να υπάρχουν μέσα στα chunks που θα φέρει πίσω η βάση μας.
EVALUATION_DATA = [
    {
        "question": "What medications are contraindicated during pregnancy?",
        "expected_keywords": ["fetal toxicity", "renal malformations", "arbs"]
    },
    {
        "question": "What is the classification for Stage 1 hypertension?",
        "expected_keywords": ["130-139", "80-89", "stage 1"]
    },
    {
        "question": "When should follow-up happen after starting new medication?",
        "expected_keywords": ["4 weeks", "potassium", "renal function"]
    }
]

def run_evaluation():
    print("⏳ Έναρξη Αυτοματοποιημένης Αξιολόγησης (MLOps Benchmark)...\n")
    
    # Φόρτωση του ίδιου μοντέλου
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

    total_questions = len(EVALUATION_DATA)
    total_recall = 0

    # 2. Εκτέλεση του Test
    for i, item in enumerate(EVALUATION_DATA):
        query = item["question"]
        keywords = item["expected_keywords"]

        print(f"[{i+1}/{total_questions}] Τεστάρισμα Ερώτησης: '{query}'")

        # Ζητάμε τα 2 πιο σχετικά chunks
        results = db.similarity_search_with_score(query, k=2)

        # Ενώνουμε τα κείμενα που βρήκε και τα κάνουμε πεζά (lowercase) για εύκολη σύγκριση
        retrieved_text = " ".join([doc.page_content.lower() for doc, _ in results])

        # 3. Υπολογισμός Μετρικής (Keyword-based Recall)
        # Ελέγχουμε πόσα από τα αναμενόμενα keywords υπάρχουν στο κείμενο που ανακτήθηκε
        hits = sum(1 for kw in keywords if kw.lower() in retrieved_text)
        
        # Το Recall είναι: (Λέξεις που βρέθηκαν) / (Σύνολο λέξεων που έπρεπε να βρεθούν)
        recall_score = hits / len(keywords)
        total_recall += recall_score

        print(f"   > Αναμενόμενα Keywords: {keywords}")
        print(f"   > Βρέθηκαν: {hits}/{len(keywords)}")
        print(f"   > Σκορ Ανάκλησης (Recall): {recall_score * 100:.0f}%\n")

    # 4. Τελικό Αποτέλεσμα
    avg_recall = total_recall / total_questions
    
    print("==============================================")
    print("📊 ΤΕΛΙΚΗ ΑΝΑΦΟΡΑ ΑΞΙΟΛΟΓΗΣΗΣ (SYSTEM BENCHMARK)")
    print("==============================================")
    print(f"Σύνολο Ερωτήσεων (Test Set): {total_questions}")
    print(f"Μέση Ανάκληση (Average Recall): {avg_recall * 100:.1f}%")
    
    if avg_recall == 1.0:
        print("\n✅ Κατάσταση: PRODUCTION READY. Το σύστημα ανακτά το 100% της κρίσιμης πληροφορίας.")
    else:
        print("\n⚠️ Κατάσταση: NEEDS TUNING. Κάποια πληροφορία χάνεται στην ανάκτηση.")
    print("==============================================\n")

if __name__ == "__main__":
    run_evaluation()