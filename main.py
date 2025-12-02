import os
import sys
import requests
import json
import time
import secrets
from typing import Annotated # საჭიროა სხვადასხვა დეკლარაციებისთვის
from dotenv import load_dotenv

# --- FastAPI და HTML იმპორტები ---
from fastapi import FastAPI, Header, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # HTML სერვირებისთვის
from pydantic import BaseModel
import uvicorn
from pypdf import PdfReader

# --- RAG ინსტრუმენტების იმპორტი ---
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores.chroma import Chroma
    from langchain_core.documents import Document
    RAG_TOOLS_AVAILABLE = True
except ImportError:
    RAG_TOOLS_AVAILABLE = False
    
# --- კონფიგურაცია: გასაღებების მოტანა გარემოს ცვლადებიდან ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY")

if not LOCAL_API_KEY:
     print("❌ WARNING: LOCAL_API_KEY არ არის დაყენებული.")

API_KEY_NAME = "X-API-Key"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
PERSONA_PDF_PATH = "chatbotprompt.pdf"
CHROMA_PATH = "chroma_db"

global_rag_retriever = None

# --- ფუნქცია პერსონის PDF-დან ჩასატვირთად ---
def load_persona_from_pdf(file_path: str) -> str:
    """კითხულობს მთელ ტექსტს PDF ფაილიდან pypdf-ის გამოყენებით."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        if not text.strip():
            return "თქვენ ხართ სასარგებლო ასისტენტი, რომელიც პასუხობს ქართულ ენაზე."
        return text.strip()
    except Exception as e:
        print(f"❌ ERROR: პერსონის PDF-ის წაკითხვისას შეცდომა: {e}")
        return "თქვენ ხართ სასარგებლო ასისტენტი, რომელიც პასუხობს ქართულ ენაზე."


CUSTOM_PERSONA_TEXT = load_persona_from_pdf(PERSONA_PDF_PATH)

# --- FastAPI აპლიკაციის ინიციალიზაცია ---
app = FastAPI(title="Gemini RAG API", version="1.1 - 401 Fix")

# --- Startup ლოგიკა: RAG ინიციალიზაცია ---
@app.on_event("startup")
async def startup_event():
    global global_rag_retriever
    
    if not RAG_TOOLS_AVAILABLE:
        print("RAG ინიციალიზაცია გამოტოვებულია.")
        return
        
    if GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    else:
        print("❌ ERROR: Gemini API გასაღები ვერ მოიძებნა. Langchain-ის embedding-ები ვერ იმუშავებს.")
        return

    if os.path.exists(CHROMA_PATH):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vector_store = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=embeddings
            )
            global_rag_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            print(f"✅ RAG Retriever წარმატებით ჩაიტვირთა.")
        except Exception as e:
            print(f"❌ ERROR: ChromaDB-ის ჩატვირთვა ვერ მოხერხდა: {e}.")
    else:
        print(f"⚠️ WARNING: ვექტორული ბაზა {CHROMA_PATH} ვერ მოიძებნა. RAG არააქტიურია.")
        

# --- CORS Middleware დამატება ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8080"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. HTML ფაილის ჩატვირთვა და სერვირება ---
try:
    with open("index.html", "r", encoding="utf-8") as f:
        HTML_CONTENT = f.read()
except FileNotFoundError:
    HTML_CONTENT = "<h1>FastAPI API მუშაობს, მაგრამ ფრონტენდის (index.html) ფაილი ვერ მოიძებნა.</h1>"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """აბრუნებს HTML ინტერფეისს."""
    return HTMLResponse(content=HTML_CONTENT, status_code=200)


# --- 2. API KEY ვალიდაცია (დარჩა მხოლოდ როგორც მაგალითი, მაგრამ არ გამოიყენება /process_query-ზე) ---
# ეს ფუნქცია ამოღებულია /process_query-დან 401 შეცდომის გამოსასწორებლად.
# თუ მოგვიანებით დაგჭირდებათ დაცული ენდპოინტი, შეგიძლიათ გამოიყენოთ:
def verify_external_api_key(api_key: str = Header(..., alias=API_KEY_NAME)):
    """ამოწმებს გარე API გასაღებს, თუ LOCAL_API_KEY დაყენებულია."""
    if not LOCAL_API_KEY or not secrets.compare_digest(api_key, LOCAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="არასწორი API გასაღები",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# --- მონაცემთა მოდელები ---
class ChatbotRequest(BaseModel):
    prompt: str
    user_id: str

class ChatbotResponse(BaseModel):
    status: str
    processed_prompt: str
    ai_response: str
    result_data: dict

# --- Gemini API-ს გამოძახება (RAG ლოგიკით) ---
def generate_gemini_content(prompt: str) -> str:
    """უკავშირდება Gemini API-ს, იყენებს RAG-ს კონტექსტის დასამატებლად."""
    if not GEMINI_API_KEY:
        return "ERROR: Gemini API გასაღები ვერ მოიძებნა."
    
    rag_context = ""
    is_rag_active = global_rag_retriever is not None
    
    if is_rag_active:
        try:
            docs: list[Document] = global_rag_retriever.get_relevant_documents(prompt)
            context_text = "\n---\n".join([doc.page_content for doc in docs])
            
            rag_context = (
                "თქვენ მოგეცემათ დამატებითი კონტექსტი 'DOCUMENTS'-ის სექციაში. "
                "გამოიყენეთ ეს ინფორმაცია, რომ უპასუხოთ შეკითხვას.\n\n"
                f"--- DOCUMENTS ---\n{context_text}\n---"
            )
        except Exception:
            rag_context = ""

    final_prompt = f"{rag_context}\n\nმომხმარებლის შეკითხვა: {prompt}"

    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "role": "user",  
                "parts": [{"text": f"შემდეგი ტექსტი განსაზღვრავს თქვენს მთავარ პერსონას. მკაცრად მიჰყევით მას:\n\n---\n{CUSTOM_PERSONA_TEXT}\n---"}]
            },
            {
                "role": "user",
                "parts": [{"text": final_prompt}]
            }
        ]
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                headers=headers, 
                data=json.dumps(payload),
                timeout=30 
            )
            
            if response.status_code >= 400:
                # API-ის შეცდომის დეტალური დამუშავება
                error_msg = f"Gemini API-მ დააბრუნა {response.status_code} შეცდომა."
                try:
                    error_detail = response.json()
                    error_msg += f" დეტალები: {error_detail.get('error', {}).get('message', 'დეტალური შეტყობინება ვერ მიიღეს.')}"
                except json.JSONDecodeError:
                    pass
                return f"ERROR: {error_msg}"

            response.raise_for_status() 
            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                return candidate['content']['parts'][0]['text']
            
            return f"Gemini API-მ დააბრუნა არასტანდარტული პასუხი."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return f"ERROR: Gemini API-სთან დაკავშირება ვერ მოხერხდა. შეცდომა: {e}"
        except Exception as e:
            return f"ERROR: მოულოდნელი შეცდომა: {e}"
    
    return "ERROR: პასუხი ვერ იქნა გენერირებული."


# --- API ენდპოინტები ---

@app.get("/status")
def read_root():
    """აბრუნებს API-სა და RAG-ის სტატუსს."""
    rag_status = "აქტიურია" if global_rag_retriever else "არააქტიურია (გაუშვით ingest.py)"
    return {"message": "API მუშაობს!", "RAG_Status": rag_status}

@app.post("/process_query", response_model=ChatbotResponse)
async def process_query(
    request_data: ChatbotRequest
    # 💥 აქ ავტორიზაცია ამოღებულია 401 შეცდომის გამოსასწორებლად! 
):
    gemini_response = generate_gemini_content(request_data.prompt)
    
    processed_prompt_length = len(request_data.prompt)
    response_data = {
        "user": request_data.user_id,
        "length": processed_prompt_length,
        "is_rag_active": global_rag_retriever is not None,
        "gemini_model": GEMINI_MODEL_NAME
    }
    
    return ChatbotResponse(
        status="success",
        processed_prompt=f"თქვენი მოთხოვნა დამუშავებულია. სიგრძე: {processed_prompt_length}.",
        ai_response=gemini_response,
        result_data=response_data,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
