import os
import requests
import json
import time
import secrets
from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pypdf import PdfReader 
# --- დამატებულია dotenv ბიბლიოთეკა გასაღებების უსაფრთხოდ ჩასატვირთად ლოკალურ გარემოში ---
from dotenv import load_dotenv

# ჩატვირთეთ გარემოს ცვლადები .env ფაილიდან (ლოკალური ტესტირებისთვის)
load_dotenv()

# --- RAG ინსტრუმენტების იმპორტი ---
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores.chroma import Chroma 
    from langchain_core.documents import Document
    RAG_TOOLS_AVAILABLE = True
except ImportError as e:
    RAG_TOOLS_AVAILABLE = False
    # print(f"Warning: RAG ბიბლიოთეკების იმპორტის შეცდომაა: {e}. RAG ფუნქციები გამორთულია.")
    # კომენტარში ჩავსვი, რადგან Render-ზე შეიძლება არ იყოს საჭირო RAG ბიბლიოთეკები
    # თუ RAG-ის გამოყენება გსურთ, ეს ბიბლიოთეკები უნდა დაამატოთ requirements.txt-ში
    pass
# -----------------------------------

# --- კონფიგურაცია: გასაღებების მოტანა გარემოს ცვლადებიდან ---

# Gemini API გასაღები (აუცილებლად იკითხება os.environ.get-ით)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Local API გასაღები (გამოიყენება ავტორიზაციისთვის)
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY")
if not LOCAL_API_KEY:
     # თუ LOCAL_API_KEY არ არის დაყენებული (მაგ. Render-ზე), ეს შეცდომას გამოიწვევს
     print("❌ ERROR: LOCAL_API_KEY არ არის დაყენებული. გთხოვთ დაამატოთ ის .env ფაილში ან Render-ის ცვლადებში.")

API_KEY_NAME = "X-API-Key"

# Gemini API და RAG პარამეტრები
GEMINI_MODEL_NAME = "gemini-2.5-flash" # მოდელი შევცვალე სტაბილურ ვერსიაზე
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
PERSONA_PDF_PATH = "chatbotprompt.pdf"
CHROMA_PATH = "chroma_db" 

# გლობალური ობიექტები (ინიციალიზდება სერვერის გაშვებისას)
global_rag_retriever = None

# --- ფუნქცია პერსონის PDF-დან ჩასატვირთად (ლოგიკა უცვლელია) ---
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
            print(f"❌ ERROR: PDF ფაილი '{file_path}' ცარიელია.")
            return "თქვენ ხართ სასარგებლო ასისტენტი, რომელიც პასუხობს ქართულ ენაზე."
        print(f"✅ პერსონის ტექსტი წარმატებით ჩაიტვირთა {file_path}-დან. სიგრძე: {len(text.strip())} სიმბოლო.")
        return text.strip()
    except Exception as e:
        print(f"❌ ERROR: პერსონის PDF-ის წაკითხვისას შეცდომა: {e}")
        return "თქვენ ხართ სასარგებლო ასისტენტი, რომელიც პასუხობს ქართულ ენაზე."

# პერსონის ჩატვირთვა სერვერის გაშვებამდე
CUSTOM_PERSONA_TEXT = load_persona_from_pdf(PERSONA_PDF_PATH)

# --- FastAPI აპლიკაციის ინიციალიზაცია ---
app = FastAPI(title="Gemini RAG API", version="1.0 - RAG Activated")

# --- Startup ლოგიკა: RAG ინიციალიზაცია (ლოგიკა უცვლელია) ---
@app.on_event("startup")
async def startup_event():
    global global_rag_retriever
    
    if not RAG_TOOLS_AVAILABLE:
        print("RAG ინიციალიზაცია გამოტოვებულია, რადგან საჭირო ბიბლიოთეკები ვერ მოიძებნა.")
        return
        
    print(">>> RAG სისტემის ინიციალიზაცია...")
    
    # Langchain-ისთვის Gemini API გასაღების დაყენება
    if GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    else:
        print("❌ ERROR: Gemini API გასაღები ვერ მოიძებნა. Langchain-ის embedding-ები ვერ იმუშავებს.")
        return

    if os.path.exists(CHROMA_PATH):
        try:
            # 1. Embedding მოდელის ჩატვირთვა
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            
            # 2. ChromaDB-ის ჩატვირთვა
            vector_store = Chroma(
                persist_directory=CHROMA_PATH, 
                embedding_function=embeddings
            )
            # 3. Retriever-ის შექმნა (კონტექსტის ამოსაღებად)
            global_rag_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            print(f"✅ RAG Retriever წარმატებით ჩაიტვირთა {CHROMA_PATH}-დან.")
        except Exception as e:
            print(f"❌ ERROR: ChromaDB-ის ჩატვირთვა ვერ მოხერხდა: {e}.")
    else:
        print(f"⚠️ WARNING: ვექტორული ბაზა {CHROMA_PATH} ვერ მოიძებნა. RAG არააქტიურია.")
        

# --- CORS Middleware დამატება (ლოგიკა უცვლელია) ---
origins = ["*", "http://localhost", "http://localhost:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------------------------

# (Dependency) ავტორიზაციის ფუნქცია (შეიცვალა გასაღების შემოწმება)
async def verify_api_key(api_key: str = Header(..., alias=API_KEY_NAME)):
    """ამოწმებს API გასაღებს."""
    # შემოწმება მხოლოდ იმ შემთხვევაში თუ LOCAL_API_KEY დაყენებულია
    if not LOCAL_API_KEY or not secrets.compare_digest(api_key, LOCAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="არასწორი API გასაღები",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key

# მონაცემთა მოდელები (უცვლელია)
class ChatbotRequest(BaseModel):
    prompt: str
    user_id: str

class ChatbotResponse(BaseModel):
    """პასუხის მოდელი, სადაც 'ai_response' მოვა Gemini-დან."""
    status: str
    processed_prompt: str
    ai_response: str
    result_data: dict

# --- Gemini API-ს გამოძახება (RAG ლოგიკით) ---
def generate_gemini_content(prompt: str) -> str:
    """უკავშირდება Gemini API-ს, იყენებს RAG-ს კონტექსტის დასამატებლად."""
    if not GEMINI_API_KEY:
        return "ERROR: Gemini API გასაღები ვერ მოიძებნა. შეამოწმეთ გარემოს ცვლადები."
    
    # 1. კონტექსტის ამოღება RAG-ის საშუალებით (ლოგიკა უცვლელია)
    rag_context = ""
    is_rag_active = global_rag_retriever is not None
    # ... [დანარჩენი RAG ლოგიკა უცვლელია] ...
    
    if is_rag_active:
        try:
            # მოძებნეთ ყველაზე რელევანტური 3 დოკუმენტი
            docs: list[Document] = global_rag_retriever.get_relevant_documents(prompt)
            # კონტექსტის ფორმატირება
            context_text = "\n---\n".join([doc.page_content for doc in docs])
            
            rag_context = (
                "თქვენ მოგეცემათ დამატებითი კონტექსტი 'DOCUMENTS'-ის სექციაში. "
                "გამოიყენეთ ეს ინფორმაცია, რომ უპასუხოთ შეკითხვას. "
                "თუ პასუხი კონტექსტში არ არის, გამოიყენეთ თქვენი ზოგადი ცოდნა.\n\n"
                f"--- DOCUMENTS ---\n{context_text}\n---"
            )
            # print(f"🔎 RAG-მა იპოვა {len(docs)} რელევანტური ფრაგმენტი.")
            
        except Exception as e:
            # print(f"❌ ERROR: RAG Retrieval-ის შეცდომა: {e}")
            rag_context = ""

    # 2. საბოლოო პრომპტის ფორმირება
    final_prompt = f"{rag_context}\n\nმომხმარებლის შეკითხვა: {prompt}"

    headers = {"Content-Type": "application/json"}
    
    # 3. Payload-ის სტრუქტურა (პერსონა + RAG კონტექსტი) (ლოგიკა უცვლელია)
    payload = {
        "contents": [
            # 1. გრძელი პერსონის ტექსტი (სისტემური კონტექსტი)
            {
                "role": "user",  
                "parts": [{"text": f"შემდეგი ტექსტი განსაზღვრავს თქვენს მთავარ პერსონას. მკაცრად მიჰყევით მას:\n\n---\n{CUSTOM_PERSONA_TEXT}\n---"}]
            },
            # 2. RAG კონტექსტი და მომხმარებლის მიმდინარე მოთხოვნა
            {
                "role": "user",
                "parts": [{"text": final_prompt}]
            }
        ]
    }

    # API-ს გამოძახება ექსპონენციალური Backoff-ით (ლოგიკა უცვლელია)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Gemini API-ს გამოძახებისას გასაღები გამოიყენება, რომელიც os.environ-დან მოდის.
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                headers=headers, 
                data=json.dumps(payload),
                timeout=30 
            )
            
            # ... [შეცდომების დამუშავება უცვლელია] ...
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                    return f"ERROR: Gemini API-მ დააბრუნა {response.status_code} შეცდომა. დეტალები: {error_detail.get('error', {}).get('message', 'დეტალური შეტყობინება ვერ მიიღეს.')}"
                except json.JSONDecodeError:
                    return f"ERROR: Gemini API-მ დააბრუნა {response.status_code} შეცდომა. პასუხი არ არის JSON-ში."

            response.raise_for_status() 
            result = response.json()
            
            # პასუხის ამოღება
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


@app.get("/")
def read_root():
    rag_status = "აქტიურია" if global_rag_retriever else "არააქტიურია (გაუშვით ingest.py)"
    return {"message": "API მუშაობს!", "RAG_Status": rag_status}

@app.post("/process_query", response_model=ChatbotResponse, tags=["Secured"])
async def process_query(
    request_data: ChatbotRequest,
    api_key: str = Depends(verify_api_key)
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
    # Render-ზე პორტი 8080-ით უნდა გაეშვას
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
