import os
import sys
# --- dotenv დამატება ---
from dotenv import load_dotenv 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma 
from google import genai 

# --- კონფიგურაცია ---

# ლოკალური გამოყენებისთვის ჩატვირთეთ გარემოს ცვლადები .env-დან
load_dotenv()

# გასაღებების მოტანა გარემოს ცვლადებიდან
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY 
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 
else:
    print("❌ ERROR: Gemini API გასაღები ვერ მოიძებნა გარემოს ცვლადებში. ინდექსირება შეუძლებელია.")
    sys.exit(1)


# !!! 2. მიუთითეთ თქვენი PDF-ების საქაღალდის ზუსტი სახელი !!!
DOCS_DIR = "Steam" 

# ვექტორული ბაზის ლოკალური გზა (ეს უნდა ემთხვეოდეს main.py-ს)
CHROMA_PATH = "chroma_db" 

# --- ინდექსირების ფუნქცია ---
def ingest_documents():
    """კითხულობს PDF-ებს, ანაწილებს მათ და ინახავს ChromaDB-ში."""
    
    if not os.path.exists(DOCS_DIR):
        print(f"❌ ERROR: დოკუმენტების საქაღალდე '{DOCS_DIR}' ვერ მოიძებნა. გთხოვთ, შექმნათ და ჩაყაროთ PDF-ები.")
        return

    documents = []
    
    # 1. დოკუმენტების ჩატვირთვა
    print(f"🔄 დოკუმენტების ჩატვირთვა საქაღალდიდან: {DOCS_DIR}...")
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    
    for filename in pdf_files:
        filepath = os.path.join(DOCS_DIR, filename)
        try:
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"   ✅ ჩაიტვირთა: {filename}")
        except Exception as e:
            print(f"   ❌ შეცდომა ჩატვირთვისას {filename}: {e}")
            
    if not documents:
        print("❌ ERROR: ვერ მოიძებნა PDF ფაილები ჩასატვირთად.")
        return

    # 2. ტექსტის დანაწილება (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📊 დოკუმენტები დანაწილდა {len(chunks)} ფრაგმენტად (Chunks).")
    
    # 3. Embedding-ების გენერაცია და ChromaDB-ში შენახვა
    print("💾 ვექტორების გენერაცია და ChromaDB-ში შენახვა...")
    try:
        # GoogleGenerativeAIEmbeddings ავტომატურად იყენებს os.environ["GEMINI_API_KEY"]-ს
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") 
        
        vector_store = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=CHROMA_PATH
        )
        vector_store.persist()
        print(f"✅ ინდექსირება დასრულდა! მონაცემთა ბაზა შენახულია: {CHROMA_PATH}")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: ვექტორების შექმნა ვერ მოხერხდა.")
        print(f"დეტალები: {e}")
        sys.exit(1)


if __name__ == "__main__":
    ingest_documents()
