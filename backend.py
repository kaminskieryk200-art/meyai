import os
import sys
import json
import subprocess
import signal
import time
from typing import List, Optional
import torch
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import gc

from model import MeyTokenizer, MeyTransformer
from rag import MeyRAG
from web_search import search_web_ddg

app = FastAPI(title="Mey AI Backend")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATA_DIR, "mey_custom.pt")
VOCAB_PATH = os.path.join(DATA_DIR, "mey_vocab.json")
STATUS_PATH = os.path.join(DATA_DIR, "training_status.json")
DB_PATH = os.path.join(DATA_DIR, "knowledge_base.json")

# Initialize RAG
rag = MeyRAG(db_path=DB_PATH)

# Global variables for training process and cached PyTorch model
training_process = None
cached_custom_model = None
cached_custom_tokenizer = None
cached_model_path_mtime = 0

# Hugging Face cache
hf_model = None
hf_tokenizer = None
hf_model_id = None

# Pydantic models for API
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str  # "mey:1.0", "mey:1.0-nano", "mey_custom"
    temperature: float = 0.7
    use_rag: bool = False
    system_prompt: Optional[str] = None

@app.get("/api/models")
def get_models():
    """List available models and check availability"""
    custom_model_available = os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)
    
    models = []
    
    # Standard Mey 1.0 (Hugging Face)
    models.append({
        "id": "mey:1.0",
        "name": "Mey 1.0 (Standard)",
        "description": "Inteligentny model Hugging Face (Qwen 2.5 1.5B) działający w 100% lokalnie w Pythonie.",
        "status": "Gotowy",
        "local": True,
        "type": "huggingface"
    })
    
    # Mey 1.0 Nano (Hugging Face)
    models.append({
        "id": "mey:1.0-nano",
        "name": "Mey 1.0 Nano (Lite)",
        "description": "Szybki, lekki model Hugging Face (Qwen 2.5 0.5B) działający w 100% lokalnie w Pythonie.",
        "status": "Gotowy",
        "local": True,
        "type": "huggingface"
    })
    
    # Mey 1.0 Pro (Hugging Face + Web Search)
    models.append({
        "id": "mey:pro",
        "name": "Mey 1.0 Pro (Web Search)",
        "description": "Najmocniejszy model (Qwen 2.5 3B) z dostępem do internetu (wyszukiwanie w czasie rzeczywistym).",
        "status": "Gotowy",
        "local": True,
        "type": "huggingface"
    })

    # Custom trained model
    models.append({
        "id": "mey_custom",
        "name": "Mey Custom (PyTorch)",
        "description": "Twój własny model Transformer trenowany lokalnie od zera w panelu treningowym.",
        "status": "Gotowy do użycia" if custom_model_available else "Wymaga wytrenowania",
        "local": True,
        "type": "pytorch"
    })
    
    return {
        "system_connected": True,  # Keep green dot in UI
        "models": models
    }

def get_custom_model():
    """Load or return cached local custom PyTorch model"""
    global cached_custom_model, cached_custom_tokenizer, cached_model_path_mtime
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        raise HTTPException(status_code=400, detail="Model custom nie został jeszcze wytrenowany.")
        
    mtime = os.path.getmtime(MODEL_PATH)
    
    # If file modified or not loaded yet, reload
    if cached_custom_model is None or mtime > cached_model_path_mtime:
        try:
            # Load tokenizer
            tokenizer = MeyTokenizer()
            tokenizer.load(VOCAB_PATH)
            
            # Load checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            hp = checkpoint.get('hyperparameters', {'embed_dim': 128, 'num_heads': 4, 'num_layers': 4, 'block_size': 64})
            
            # Initialize model
            model = MeyTransformer(
                vocab_size=checkpoint['vocab_size'],
                embed_dim=hp['embed_dim'],
                num_heads=hp['num_heads'],
                num_layers=hp['num_layers'],
                block_size=hp['block_size']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            cached_custom_model = model
            cached_custom_tokenizer = tokenizer
            cached_model_path_mtime = mtime
            print("Loaded custom PyTorch model into memory.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Błąd ładowania modelu PyTorch: {str(e)}")
            
    return cached_custom_model, cached_custom_tokenizer

def get_hf_model(model_name: str):
    """Load or return cached local Hugging Face model on CPU"""
    global hf_model, hf_tokenizer, hf_model_id
    
    if model_name == "mey:1.0":
        target_id = "Qwen/Qwen2.5-1.5B-Instruct"
    elif model_name == "mey:1.0-nano":
        target_id = "Qwen/Qwen2.5-0.5B-Instruct"
    elif model_name == "mey:pro":
        target_id = "Qwen/Qwen2.5-3B-Instruct"
    else:
        raise HTTPException(status_code=400, detail=f"Nieprawidłowy model: {model_name}")
        
    if hf_model is None or hf_model_id != target_id:
        print(f"Ładowanie modelu Hugging Face {target_id} na CPU...")
        
        # Free memory of previous model
        if hf_model is not None:
            del hf_model
            del hf_tokenizer
            hf_model = None
            hf_tokenizer = None
            gc.collect()
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            hf_tokenizer = AutoTokenizer.from_pretrained(target_id)
            # Load in bfloat16 to optimize RAM and speed on CPU
            hf_model = AutoModelForCausalLM.from_pretrained(
                target_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).to("cpu")
            hf_model_id = target_id
            print(f"Model {target_id} załadowany pomyślnie w bfloat16 na CPU.")
        except Exception as e:
            try:
                print(f"Błąd bfloat16, próba w float32 dla {target_id}: {e}")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                hf_tokenizer = AutoTokenizer.from_pretrained(target_id)
                hf_model = AutoModelForCausalLM.from_pretrained(
                    target_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to("cpu")
                hf_model_id = target_id
                print(f"Model {target_id} załadowany pomyślnie w float32 na CPU.")
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Błąd ładowania modelu Hugging Face: {str(e2)}")
                
    return hf_model, hf_tokenizer

def should_search_web(query: str) -> bool:
    """
    Decide whether we should run web search based on user message.
    Avoids searching for greetings, simple replies, or empty inputs.
    """
    cleaned = query.strip().lower().strip("?.!,")
    if len(cleaned) < 3:
        return False
        
    greetings = {
        "hej", "heja", "hejka", "czesc", "cześć", "witaj", "witajcie", "siema", "siemano", "siemanko", "elo", "elówa",
        "hello", "hi", "hey", "good morning", "dzien dobry", "dzień dobry", "dobry wieczór", "dobry wieczor",
        "dzieki", "dzięki", "thanks", "thank you", "dziekuje", "dziękuję", "super", "ok", "okej", "okay",
        "jasne", "tak", "nie", "no", "yes", "aha", "mhm", "pl", "en", "pytanie", "pomocy", "help", "witaj!"
    }
    if cleaned in greetings:
        return False
        
    # Check if it consists only of greeting words
    words = cleaned.split()
    if len(words) <= 2:
        if all(w in greetings for w in words):
            return False
            
    return True

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Retrieve final user message for RAG
    user_msg = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_msg = msg.content
            break

    # RAG Context injection
    context_str = ""
    if request.use_rag and user_msg:
        search_results = rag.search(user_msg, top_k=3)
        if search_results:
            context_pieces = [f"- {res['text']} (z pliku: {res['filename']})" for res in search_results]
            context_str = "\n".join(context_pieces)
            print(f"RAG matching context found: {len(search_results)} chunks.")

    # 1. Custom local PyTorch model execution
    if request.model == "mey_custom":
        model, tokenizer = get_custom_model()
        
        # Prepare local prompt
        prompt = ""
        if context_str:
            prompt += f"Baza wiedzy:\n{context_str}\n\n"
        prompt += f"Pytanie: {user_msg}\nOdpowiedź:"
        
        # Tokenize and generate
        try:
            tokens = tokenizer.encode(prompt)
            tokens = tokens[-model.block_size + 1:]
            idx = torch.tensor([tokens], dtype=torch.long)
            
            idx_out = model.generate(idx, max_new_tokens=60, temperature=request.temperature, top_k=20)
            generated_tokens = idx_out[0][len(tokens):].tolist()
            response_text = tokenizer.decode(generated_tokens)
            
            if not response_text:
                response_text = "Model nie potrafił wygenerować odpowiedzi na podstawie swojej wiedzy. Spróbuj go dotrenować na większym pliku tekstowym!"
                
            def custom_stream():
                words = response_text.split()
                yield "data: " + json.dumps({"choices": [{"delta": {"content": "[UWAGA: To jest Twój własny miniaturowy model PyTorch. Odpowiada tylko słowami, których się nauczył!]\n\n"}}]}) + "\n\n"
                
                for word in words:
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": word + " "}}]}) + "\n\n"
                    time.sleep(0.08)
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(custom_stream(), media_type="text/event-stream")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Błąd generowania w modelu PyTorch: {str(e)}")

    # 2. Hugging Face execution (Mey 1.0, Mey 1.0 Nano, or Mey 1.0 Pro)
    else:
        if request.model == "mey:1.0":
            target_id = "Qwen/Qwen2.5-1.5B-Instruct"
        elif request.model == "mey:1.0-nano":
            target_id = "Qwen/Qwen2.5-0.5B-Instruct"
        elif request.model == "mey:pro":
            target_id = "Qwen/Qwen2.5-3B-Instruct"
        else:
            raise HTTPException(status_code=400, detail=f"Nieznany model: {request.model}")
            
        is_first_load = (hf_model is None or hf_model_id != target_id)
        
        # Web Search Context injection
        web_context = ""
        if request.model in ("mey:1.0", "mey:1.0-nano", "mey:pro") and user_msg and should_search_web(user_msg):
            print(f"Searching web for: {user_msg}")
            try:
                search_results = search_web_ddg(user_msg)
                if search_results:
                    pieces = []
                    for idx, r in enumerate(search_results):
                        pieces.append(f"[{idx+1}] Tytuł: {r['title']}\n    Link: {r['url']}\n    Treść: {r['snippet']}")
                    web_context = "\n\n".join(pieces)
                    print(f"Found {len(search_results)} search results.")
            except Exception as e:
                print(f"Web search error: {e}")
                
        model, tokenizer = get_hf_model(request.model)
        
        # Build prompt messages for Hugging Face
        hf_messages = []
        
        # Inject system prompt override or use default
        system_prompt = request.system_prompt
        if system_prompt:
            hf_messages.append({"role": "system", "content": system_prompt})
            
        # Re-construct message history
        for msg in request.messages:
            if msg == request.messages[-1] and msg.role == "user":
                enriched_content = ""
                if context_str:
                    enriched_content += f"=== PLIKI LOKALNE (RAG) ===\n{context_str}\n===========================\n\n"
                if web_context:
                    enriched_content += f"=== WYSZUKIWARKA INTERNETOWA (DuckDuckGo) ===\n{web_context}\n=============================================\n\n"
                
                if enriched_content:
                    enriched_content += (
                        "Wykorzystaj powyższe informacje do sformułowania odpowiedzi. "
                        "Jeśli informacja pochodzi z wyszukiwarki internetowej, podaj link źródłowy w tekście.\n\n"
                        f"Pytanie: {msg.content}"
                    )
                    hf_messages.append({"role": "user", "content": enriched_content})
                else:
                    hf_messages.append({"role": "user", "content": msg.content})
            else:
                hf_messages.append({"role": msg.role, "content": msg.content})
                
        try:
            # Format prompt using chat template
            prompt_str = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_str, return_tensors="pt").to("cpu")
            
            # Setup streamer
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Setup generation kwargs
            generation_kwargs = dict(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                streamer=streamer,
                max_new_tokens=512,
                temperature=max(request.temperature, 0.01),
                do_sample=True if request.temperature > 0.05 else False,
                top_p=0.9,
            )
            
            # Start generation in thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            def hf_stream():
                try:
                    if web_context:
                        yield "data: " + json.dumps({"choices": [{"delta": {"content": "🔍 *Przeszukano internet. Odpowiadam na podstawie wyników wyszukiwania...*\n\n"}}]}) + "\n\n"
                        
                    collected = []
                    for new_text in streamer:
                        if new_text:
                            collected.append(new_text)
                    response_text = "".join(collected)
                    if response_text:
                        formatted_chunk = {"choices": [{"delta": {"content": response_text}}]}
                        yield f"data: {json.dumps(formatted_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": f"\nBłąd strumieniowania Hugging Face: {str(e)}"}}]}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    
            return StreamingResponse(hf_stream(), media_type="text/event-stream")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Błąd generowania w modelu Hugging Face: {str(e)}")

@app.post("/api/train/start")
def start_training(
    iters: int = Form(500),
    batch_size: int = Form(32),
    block_size: int = Form(64),
    embed_dim: int = Form(128),
    num_heads: int = Form(4),
    num_layers: int = Form(4),
    lr: float = Form(0.001),
    dataset: str = Form("data_pl.txt")
):
    global training_process
    
    # Check if already running
    if training_process and training_process.poll() is None:
        return JSONResponse(status_code=400, content={"message": "Trening już trwa w tle."})
        
    dataset_path = os.path.join(BASE_DIR, dataset)
    # If the specified dataset doesn't exist, search workspace
    if not os.path.exists(dataset_path):
        # check parent directory (workspace root)
        dataset_path = os.path.join(os.path.dirname(BASE_DIR), dataset)
        
    if not os.path.exists(dataset_path):
        # If still doesn't exist, write standard dataset for Polish
        dataset_path = os.path.join(DATA_DIR, "dataset_mey.txt")
        if not os.path.exists(dataset_path):
            from model import MeyTokenizer
            # Let's write a default dataset
            text = (
                "witaj świecie to jest prosty model językowy po polsku. "
                "polska jest pięknym krajem w europie środkowej ze stolicą w warszawie. "
                "kraków to historyczne miasto z pięknym rynkiem i zamkiem wawelskim. "
                "programowanie to bardzo przydatna umiejętność w dzisiejszych czasach. "
                "python jest popularnym językiem programowania używanym do wielu celów. "
                "sztuczna inteligencja zmienia nasz świat na lepsze każdego dnia. "
                "uczenie maszynowe to dziedzina sztucznej inteligencji oparta na danych. "
                "komputery pomagają nam w codziennym życiu i pracy zawodowej. "
                "warszawa jest największym miastem w polsce i jej stolicą. "
                "gdansk to piękne miasto nad morzem bałtyckim w polsce. "
                "wrocław to miasto mostów i krasnali na dolnym śląsku. "
                "poznań to miasto koziołków i targów międzynarodowych. "
                "edukacja jest kluczem do sukcesu w życiu każdego człowieka. "
                "rodzina jest najważniejsza w życiu każdego z nas. "
                "gotowanie to sztuka tworzenia pysznych potraw i dań. "
                "sport utrzymuje zdrowie i dobrą kondycję fizyczną człowieka. "
                "technologia rozwija się bardzo szybko w dzisiejszych czasach. "
            )
            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write(text * 5)  # repeat to make it long enough
                
    # Prepare virtual environment python path
    venv_python = os.path.join(os.path.dirname(BASE_DIR), "venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable  # fallback to current python
        
    cmd = [
        venv_python,
        os.path.join(BASE_DIR, "training.py"),
        "--text_path", dataset_path,
        "--model_path", MODEL_PATH,
        "--vocab_path", VOCAB_PATH,
        "--status_path", STATUS_PATH,
        "--iters", str(iters),
        "--batch_size", str(batch_size),
        "--block_size", str(block_size),
        "--embed_dim", str(embed_dim),
        "--num_heads", str(num_heads),
        "--num_layers", str(num_layers),
        "--lr", str(lr)
    ]
    
    # Initialize status file
    initial_status = {
        "status": "starting",
        "step": 0,
        "total_steps": iters,
        "loss": 0.0,
        "val_loss": 0.0,
        "message": "Uruchamianie procesu treningu...",
        "history": []
    }
    with open(STATUS_PATH, 'w') as f:
        json.dump(initial_status, f)
        
    try:
        # Start subprocess
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # group process to kill it cleanly later
        )
        return {"status": "started", "message": "Proces treningu uruchomiony."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Błąd uruchamiania treningu: {str(e)}"})

@app.post("/api/train/stop")
def stop_training():
    global training_process
    
    if not training_process or training_process.poll() is not None:
        return {"status": "not_running", "message": "Brak aktywnego treningu do zatrzymania."}
        
    try:
        # Send SIGTERM to process group
        os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
        training_process = None
        
        # Update status file
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH, 'r') as f:
                status = json.load(f)
            status["status"] = "stopped"
            status["message"] = "Trening został zatrzymany przez użytkownika."
            with open(STATUS_PATH, 'w') as f:
                json.dump(status, f)
                
        return {"status": "stopped", "message": "Trening został zatrzymany."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Błąd zatrzymywania procesu: {str(e)}"})

@app.get("/api/train/status")
def get_training_status():
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
            
    return {
        "status": "idle",
        "step": 0,
        "total_steps": 0,
        "loss": 0.0,
        "val_loss": 0.0,
        "message": "Model gotowy do treningu.",
        "history": []
    }

@app.post("/api/knowledge/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate filename
    filename = file.filename
    if not filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Obsługiwane są tylko pliki tekstowe (.txt, .md)")
        
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    try:
        content = await file.read()
        text = content.decode('utf-8', errors='ignore')
        
        # Save to disk
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        # Add to RAG
        rag.add_document(filename, text)
        return {"filename": filename, "status": "success", "message": f"Plik {filename} został zaimportowany i zindeksowany."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd wgrywania pliku: {str(e)}")

@app.get("/api/knowledge/documents")
def list_documents():
    return {
        "documents": [
            {
                "filename": filename,
                "size": os.path.getsize(os.path.join(DOCUMENTS_DIR, filename)),
                "chunks_count": len([c for c in rag.chunks if c["filename"] == filename])
            }
            for filename in os.listdir(DOCUMENTS_DIR)
            if os.path.isfile(os.path.join(DOCUMENTS_DIR, filename))
        ],
        "total_chunks": len(rag.chunks),
        "use_ollama_embeddings": False
    }

@app.delete("/api/knowledge/documents/{filename}")
def delete_document(filename: str):
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plik nie istnieje.")
        
    try:
        os.remove(file_path)
        rag.delete_document(filename)
        return {"filename": filename, "status": "deleted", "message": "Plik został usunięty z bazy wiedzy."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd usuwania pliku: {str(e)}")

# Mount static folder for serving Web UI
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
