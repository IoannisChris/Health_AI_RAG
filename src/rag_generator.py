import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Η διαδρομή της βάσης μας
CHROMA_PATH = "chroma_db"

def answer_medical_query(query_text):
    print(f"\n🔍 Ερώτηση: '{query_text}'")
    
    # --- 1. ΑΝΑΚΤΗΣΗ (Retrieval) ---
    print("...Αναζήτηση στην ιατρική βιβλιογραφία...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    results = db.similarity_search_with_score(query_text, k=2)
    
    if len(results) == 0:
        print("Δεν βρέθηκαν πληροφορίες στη βάση δεδομένων.")
        return

    context_text = "\n\n".join([doc.page_content for doc, _ in results])

    # --- 2. ΔΗΜΙΟΥΡΓΙΑ ΠΡΟΜΠΤ (Prompt Engineering) ---
    # Κλειδώνουμε το μοντέλο να απαντάει ΑΥΣΤΗΡΑ με βάση το ιατρικό κείμενο
    prompt_template = """
    You are an expert medical AI assistant. Answer the user's question based ONLY on the following context.
    If the answer is not contained within the context, simply say "I don't know based on the provided guidelines."
    Do not use outside knowledge.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    final_prompt = prompt.format(context=context_text, question=query_text)

    # --- 3. ΠΑΡΑΓΩΓΗ (Generation) ---
    print("...Σύνταξη απάντησης από το τοπικό LLM (dolphin-llama3)...")
    
    # ΕΔΩ ΣΥΝΔΕΟΜΑΣΤΕ ΣΤΟ ΔΙΚΟ ΣΟΥ ΜΟΝΤΕΛΟ
    llm = Ollama(model="dolphin-llama3")
    
    # Παίρνουμε την τελική απάντηση
    response = llm.invoke(final_prompt)
    
    print("\n==============================================")
    print("🏥 Ιατρική Απάντηση:")
    print("==============================================")
    print(response)
    print("==============================================\n")

if __name__ == "__main__":
    # Test 1: Ερώτηση που ΥΠΑΡΧΕΙ στη βάση (για την εγκυμοσύνη)
    answer_medical_query("What happens if a pregnant woman takes ACE inhibitors?")
    
    # Test 2: Ερώτηση που ΔΕΝ ΥΠΑΡΧΕΙ στη βάση (για να ελέγξουμε αν θα πει "δεν ξέρω")
    answer_medical_query("What is the best treatment for diabetes?")