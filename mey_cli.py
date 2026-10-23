#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import time
from threading import Thread

# Try importing local modules for standalone fallbacks
try:
    from model import MeyTokenizer, MeyTransformer
    from reasoning import MeyReasoner
    from rag import MeyRAG
    from web_search import search_web_ddg
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    STANDALONE_AVAILABLE = True
except ImportError:
    STANDALONE_AVAILABLE = False

from ollama_client import MODEL_MAP, available as ollama_available, chat_stream as ollama_chat_stream

# ANSI Color Codes for premium terminal styling
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_VIOLET = "\033[38;5;141m"
C_CYAN = "\033[38;5;39m"
C_YELLOW = "\033[38;5;220m"
C_GREEN = "\033[38;5;82m"
C_RED = "\033[38;5;196m"
C_GRAY = "\033[38;5;244m"

BACKEND_URL = "http://127.0.0.1:8000"

def check_backend():
    try:
        res = requests.get(f"{BACKEND_URL}/api/models", timeout=1.5)
        return res.status_code == 200
    except requests.RequestException:
        return False

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


def run_chat_ollama(model_name, temp, system_prompt):
    """Run one of the primary Mey models through the local Ollama daemon."""
    ollama_model = MODEL_MAP[model_name]
    print(f"{C_YELLOW}Połączono z lokalną Ollamą. Model: {ollama_model}{C_RESET}")
    print(f"{C_BOLD}Rozmowa z {model_name}. Wpisz 'exit' lub 'quit' aby wyjść.{C_RESET}\n")
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input(f"{C_CYAN}{C_BOLD}Ty > {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        content = user_input
        if should_search_web(user_input):
            try:
                results = search_web_ddg(user_input, max_results=4)
                if results:
                    sources = "\n\n".join(
                        f"[{i}] {item['title']}\nURL: {item['url']}\nOpis: {item['snippet']}"
                        for i, item in enumerate(results, 1)
                    )
                    content = (
                        "Użyj poniższych aktualnych wyników wyszukiwania. Podaj źródła URL "
                        "i zaznacz, jeśli danych nie da się potwierdzić.\n\n"
                        f"{sources}\n\nPytanie użytkownika: {user_input}"
                    )
                    print(f"{C_GRAY}🔍 Pobrano lokalnie wyniki z internetu.{C_RESET}")
            except Exception as exc:
                print(f"{C_GRAY}Wyszukiwanie pominięte: {exc}{C_RESET}")

        messages.append({"role": "user", "content": content})
        print(f"{C_VIOLET}{C_BOLD}Mey > {C_RESET}", end="", flush=True)
        assistant_content = ""
        try:
            for token in ollama_chat_stream(ollama_model, messages):
                assistant_content += token
                print(token, end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": assistant_content})
        except requests.RequestException as exc:
            print(f"\n{C_RED}Ollama nie mogła załadować modelu: {exc}{C_RESET}")


def start_server():
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)

# --- CHAT ACTIONS ---
def run_chat_backend(model, temp, use_rag, system_prompt):
    print(f"{C_YELLOW}Połączono z lokalnym serwerem Mey AI.{C_RESET}")
    print(f"{C_GRAY}Model: {model} | Temp: {temp} | RAG: {'ON' if use_rag else 'OFF'}{C_RESET}")
    print(f"{C_BOLD}Rozmowa z Mey. Wpisz 'exit' lub 'quit' aby wyjść.{C_RESET}\n")
    
    messages = []
    
    while True:
        try:
            user_input = input(f"{C_CYAN}{C_BOLD}Ty > {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
            
        if not user_input:
            continue
        if user_input.lower() in ('exit', 'quit'):
            break
            
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temp,
            "use_rag": use_rag,
            "system_prompt": system_prompt
        }
        
        print(f"{C_VIOLET}{C_BOLD}Mey > {C_RESET}", end="", flush=True)
        
        try:
            response = requests.post(f"{BACKEND_URL}/api/chat", json=payload, stream=True, timeout=15)
            if response.status_code != 200:
                print(f"{C_RED}Błąd serwera: Kod statusu {response.status_code}{C_RESET}")
                continue
                
            assistant_content = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        data_str = decoded[6:].strip()
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            token = chunk["choices"][0]["delta"].get("content", "")
                            assistant_content += token
                            print(token, end="", flush=True)
                        except Exception:
                            pass
            print() # Newline
            messages.append({"role": "assistant", "content": assistant_content})
            
        except Exception as e:
            print(f"\n{C_RED}Błąd komunikacji z serwerem: {e}{C_RESET}")

def run_chat_standalone(model_name, temp, use_rag, system_prompt):
    if not STANDALONE_AVAILABLE:
        print(f"{C_RED}Błąd: Standalone mode wymaga zainstalowanych pakietów (torch, transformers) w środowisku.{C_RESET}")
        return
        
    print(f"{C_YELLOW}Uruchamianie w trybie Standalone (bez serwera)...{C_RESET}")
    
    # 1. Initialize RAG if needed
    rag_inst = None
    if use_rag:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.json")
        rag_inst = MeyRAG(db_path=db_path)
        print(f"{C_GRAY}Zainicjalizowano wektorową bazę wiedzy RAG.{C_RESET}")

    # 2. Load model
    if model_name == "mey:reasoning":
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mey_custom.pt")
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mey_vocab.json")
        if not os.path.exists(model_path):
            print(f"{C_RED}Brak własnego checkpointu reasoningowego: {model_path}{C_RESET}")
            return
        reasoner = MeyReasoner(model_path, vocab_path)
        print(f"{C_GREEN}Własny model reasoningowy Mey załadowany lokalnie.{C_RESET}")
    elif model_name == "mey_custom":
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mey_custom.pt")
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mey_vocab.json")
        if not os.path.exists(model_path):
            print(f"{C_RED}Błąd: Plik modelu custom {model_path} nie istnieje. Najpierw go wytrenuj.{C_RESET}")
            return
            
        print(f"{C_GRAY}Ładowanie własnego modelu PyTorch...{C_RESET}")
        tokenizer = MeyTokenizer()
        tokenizer.load(vocab_path)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        hp = checkpoint.get('hyperparameters', {'embed_dim': 128, 'num_heads': 4, 'num_layers': 4, 'block_size': 64})
        
        model = MeyTransformer(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=hp['embed_dim'],
            num_heads=hp['num_heads'],
            num_layers=hp['num_layers'],
            block_size=hp['block_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"{C_GREEN}Własny model PyTorch załadowany.{C_RESET}")
    else:
        # Hugging Face
        if model_name == "mey:1.0":
            target_id = "Qwen/Qwen2.5-1.5B-Instruct"
        elif model_name == "mey:1.0-nano":
            target_id = "Qwen/Qwen2.5-0.5B-Instruct"
        elif model_name == "mey:pro":
            target_id = "Qwen/Qwen2.5-3B-Instruct"
        print(f"{C_GRAY}Ładowanie modelu Hugging Face {target_id} na CPU (może chwilę potrwać)...{C_RESET}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(target_id)
            model = AutoModelForCausalLM.from_pretrained(
                target_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).to("cpu")
            print(f"{C_GREEN}Model Hugging Face {target_id} załadowany.{C_RESET}")
        except Exception as e:
            print(f"{C_YELLOW}Błąd bfloat16 ({e}). Próba w float32...{C_RESET}")
            tokenizer = AutoTokenizer.from_pretrained(target_id)
            model = AutoModelForCausalLM.from_pretrained(
                target_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to("cpu")
            print(f"{C_GREEN}Model załadowany w float32.{C_RESET}")
            
    print(f"{C_BOLD}Rozmowa z Mey. Wpisz 'exit' lub 'quit' aby wyjść.{C_RESET}\n")
    
    messages = []
    
    while True:
        try:
            user_input = input(f"{C_CYAN}{C_BOLD}Ty > {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
            
        if not user_input:
            continue
        if user_input.lower() in ('exit', 'quit'):
            break
            
        # RAG Search
        context_str = ""
        if use_rag and rag_inst:
            results = rag_inst.search(user_input, top_k=2)
            if results:
                context_str = "\n".join([f"- {r['text']}" for r in results])
                
        # Web Search if Pro
        web_context = ""
        if model_name in MODEL_MAP and should_search_web(user_input):
            print(f"{C_GRAY}🔍 Przeszukiwanie sieci dla: '{user_input}'...{C_RESET}")
            try:
                search_results = search_web_ddg(user_input)
                if search_results:
                    pieces = []
                    for idx, r in enumerate(search_results):
                        pieces.append(f"[{idx+1}] Tytuł: {r['title']}\n    Link: {r['url']}\n    Treść: {r['snippet']}")
                    web_context = "\n\n".join(pieces)
            except Exception as e:
                pass
                
        print(f"{C_VIOLET}{C_BOLD}Mey > {C_RESET}", end="", flush=True)
        
        if model_name == "mey_custom":
            # Generate custom PyTorch
            prompt = ""
            if context_str:
                prompt += f"Baza wiedzy:\n{context_str}\n\n"
            prompt += f"Pytanie: {user_input}\nOdpowiedź:"
            
            tokens = tokenizer.encode(prompt)
            tokens = tokens[-model.block_size + 1:]
            idx = torch.tensor([tokens], dtype=torch.long)
            
            idx_out = model.generate(idx, max_new_tokens=60, temperature=temp, top_k=20)
            generated_tokens = idx_out[0][len(tokens):].tolist()
            response_text = tokenizer.decode(generated_tokens)
            
            # Print slowly word-by-word
            for word in response_text.split():
                print(word + " ", end="", flush=True)
                time.sleep(0.08)
            print()
            
        elif model_name == "mey:reasoning":
            context_str = ""
            if use_rag and rag_inst:
                context_str = "\n".join(item["text"] for item in rag_inst.search(user_input, top_k=2))
            response_text = reasoner.answer(user_input, context_str)
            print(response_text)

        else:
            # Generate Hugging Face
            hf_messages = []
            if system_prompt:
                hf_messages.append({"role": "system", "content": system_prompt})
            
            for msg in messages:
                hf_messages.append(msg)
                
            # Current message enriched
            enriched_content = ""
            if context_str:
                enriched_content += f"=== PLIKI LOKALNE ===\n{context_str}\n=====================\n\n"
            if web_context:
                enriched_content += f"=== WYSZUKIWARKA INTERNETOWA ===\n{web_context}\n================================\n\n"
                
            if enriched_content:
                enriched_content += (
                    "Wykorzystaj powyższe informacje do sformułowania odpowiedzi. "
                    "Jeśli informacja pochodzi z wyszukiwarki internetowej, podaj link źródłowy w tekście.\n\n"
                    f"Pytanie: {user_input}"
                )
                hf_messages.append({"role": "user", "content": enriched_content})
            else:
                hf_messages.append({"role": "user", "content": user_input})
                
            if model_name == "mey:reasoning":
                try:
                    prompt_str = tokenizer.apply_chat_template(
                        hf_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except TypeError:
                    prompt_str = tokenizer.apply_chat_template(
                        hf_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                prompt_str = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_str, return_tensors="pt").to("cpu")
            
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            is_reasoning = model_name == "mey:reasoning"
            generation_kwargs = dict(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                streamer=streamer,
                max_new_tokens=1024 if is_reasoning else 512,
                temperature=0.6 if is_reasoning else max(temp, 0.01),
                do_sample=True,
                top_p=0.95 if is_reasoning else 0.9,
                top_k=20 if is_reasoning else 0,
            )
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            assistant_content = ""
            for new_text in streamer:
                assistant_content += new_text
                if not is_reasoning:
                    print(new_text, end="", flush=True)
            if is_reasoning:
                if "</think>" in assistant_content:
                    assistant_content = assistant_content.split("</think>", 1)[1]
                assistant_content = assistant_content.replace("<think>", "").strip()
                print(assistant_content, end="", flush=True)
            print()
            
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": assistant_content})

# --- KNOWLEDGE ACTIONS ---
def manage_knowledge(action, args):
    if not check_backend():
        # Standalone
        if not STANDALONE_AVAILABLE:
            print(f"{C_RED}Błąd: Standalone mode niedostępny. Uruchom serwer webowy.{C_RESET}")
            return
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.json")
        rag_inst = MeyRAG(db_path=db_path)
        
        if action == "list":
            print(f"{C_YELLOW}Dokumenty w lokalnej bazie wiedzy:{C_RESET}")
            for filename, text in rag_inst.documents.items():
                chunks = len([c for c in rag_inst.chunks if c["filename"] == filename])
                print(f"- {filename} ({len(text)} znaków, {chunks} fragmentów)")
                
        elif action == "add":
            filepath = args.file
            if not os.path.exists(filepath):
                print(f"{C_RED}Błąd: Plik {filepath} nie istnieje.{C_RESET}")
                return
            filename = os.path.basename(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            rag_inst.add_document(filename, text)
            print(f"{C_GREEN}Zindeksowano plik {filename} w lokalnej bazie wiedzy.{C_RESET}")
            
        elif action == "delete":
            filename = args.filename
            if filename not in rag_inst.documents:
                print(f"{C_RED}Błąd: Plik {filename} nie jest zindeksowany.{C_RESET}")
                return
            rag_inst.delete_document(filename)
            print(f"{C_GREEN}Usunięto plik {filename} z lokalnej bazy wiedzy.{C_RESET}")
            
        elif action == "search":
            query = args.query
            res = rag_inst.search(query, top_k=3)
            print(f"{C_YELLOW}Wyniki wyszukiwania dla: '{query}':{C_RESET}")
            for i, r in enumerate(res):
                print(f"\n[{i+1}] {C_CYAN}{r['filename']}{C_RESET} (zgodność: {r['similarity']:.3f})")
                print(f"    {r['text']}")
    else:
        # Query Backend API
        if action == "list":
            res = requests.get(f"{BACKEND_URL}/api/knowledge/documents").json()
            print(f"{C_YELLOW}Dokumenty w bazie wiedzy serwera:{C_RESET}")
            for doc in res["documents"]:
                print(f"- {doc['filename']} ({doc['size']/1024:.1f} KB, {doc['chunks_count']} fragmentów)")
                
        elif action == "add":
            filepath = args.file
            if not os.path.exists(filepath):
                print(f"{C_RED}Błąd: Plik {filepath} nie istnieje.{C_RESET}")
                return
            filename = os.path.basename(filepath)
            with open(filepath, 'rb') as f:
                res = requests.post(f"{BACKEND_URL}/api/knowledge/upload", files={"file": (filename, f)})
            if res.status_code == 200:
                print(f"{C_GREEN}{res.json()['message']}{C_RESET}")
            else:
                print(f"{C_RED}Błąd: {res.text}{C_RESET}")
                
        elif action == "delete":
            filename = args.filename
            res = requests.delete(f"{BACKEND_URL}/api/knowledge/documents/{filename}")
            if res.status_code == 200:
                print(f"{C_GREEN}Usunięto {filename} z bazy wiedzy serwera.{C_RESET}")
            else:
                print(f"{C_RED}Błąd: {res.text}{C_RESET}")
                
        elif action == "search":
            # Search is done locally as backend doesn't expose raw search endpoint, so use local model
            if not STANDALONE_AVAILABLE:
                print(f"{C_RED}Błąd: Lokalny parser niedostępny. Uruchom polecenie w katalogu z zainstalowanym NumPy.{C_RESET}")
                return
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.json")
            rag_inst = MeyRAG(db_path=db_path)
            res = rag_inst.search(args.query, top_k=3)
            print(f"{C_YELLOW}Wyniki wyszukiwania dla: '{args.query}':{C_RESET}")
            for i, r in enumerate(res):
                print(f"\n[{i+1}] {C_CYAN}{r['filename']}{C_RESET} (zgodność: {r['similarity']:.3f})")
                print(f"    {r['text']}")

# --- TRAINING ACTIONS ---
def run_training(args):
    if check_backend():
        print(f"{C_YELLOW}Uruchamianie treningu w tle na serwerze...{C_RESET}")
        payload = {
            "iters": args.iters,
            "batch_size": args.batch,
            "block_size": args.block,
            "embed_dim": args.embed,
            "num_layers": args.layers,
            "dataset": args.dataset
        }
        res = requests.post(f"{BACKEND_URL}/api/train/start", data=payload)
        if res.status_code == 200:
            print(f"{C_GREEN}Trening wystartował.{C_RESET}")
            # Poll logs
            print(f"{C_GRAY}Śledzenie logów (Ctrl+C aby przestać śledzić - proces będzie trwał dalej):{C_RESET}")
            last_step = -1
            try:
                while True:
                    status = requests.get(f"{BACKEND_URL}/api/train/status").json()
                    if status["step"] != last_step:
                        print(f"[{status['status'].upper()}] Step {status['step']}/{status['total_steps']} - Loss: {status['loss']:.4f}")
                        last_step = status["step"]
                    if status["status"] in ("completed", "stopped", "error"):
                        print(f"\n{C_BOLD}Koniec treningu: {status['message']}{C_RESET}")
                        break
                    time.sleep(1.5)
            except KeyboardInterrupt:
                print(f"\n{C_YELLOW}Zaprzestano śledzenia. Trening nadal trwa na serwerze.{C_RESET}")
        else:
            print(f"{C_RED}Błąd startu: {res.text}{C_RESET}")
    else:
        # Standalone Training
        print(f"{C_YELLOW}Uruchamianie treningu lokalnego w tym procesie...{C_RESET}")
        venv_python = sys.executable
        import subprocess
        cmd = [
            venv_python,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "training.py"),
            "--text_path", args.dataset,
            "--model_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mey_custom.pt"),
            "--vocab_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mey_vocab.json"),
            "--status_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "training_status.json"),
            "--iters", str(args.iters),
            "--batch_size", str(args.batch),
            "--block_size", str(args.block),
            "--embed_dim", str(args.embed),
            "--num_layers", str(args.layers)
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"{C_GREEN}Trening lokalny zakończony pomyślnie!{C_RESET}")
        except subprocess.CalledProcessError as e:
            print(f"{C_RED}Błąd treningu lokalnego: {e}{C_RESET}")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(
        description=f"{C_BOLD}{C_VIOLET}Mey AI CLI - Interfejs konsoli Twojego modelu AI{C_RESET}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy:")

    serve_parser = subparsers.add_parser("serve", help="Uruchom lokalny backend i panel webowy.")
    serve_parser.set_defaults(func=lambda: start_server())
    
    # Subparser chat
    chat_parser = subparsers.add_parser("chat", help="Rozpocznij interaktywną rozmowę z Mey.")
    chat_parser.add_argument("--model", "-m", choices=["mey:1.0", "mey:1.0-nano", "mey:pro"], default="mey:1.0", help="Wybór modelu czatu (domyślnie: mey:1.0)")
    chat_parser.add_argument("--temp", "-t", type=float, default=0.7, help="Temperatura generowania (domyślnie: 0.7)")
    chat_parser.add_argument("--rag", "-r", action="store_true", help="Włącz wstrzykiwanie wiedzy z Bazy Wiedzy (RAG)")
    chat_parser.add_argument("--standalone", "-s", action="store_true", help="Zmuś do działania offline (bez łączenia z serwerem webowym)")
    chat_parser.add_argument("--system", type=str, default=None, help="Prompt systemowy dla modelu")
    
    # Subparser training
    train_parser = subparsers.add_parser("train", help="Uruchom trening własnego modelu Transformer PyTorch.")
    train_parser.add_argument("--dataset", "-d", type=str, default="data_pl.txt", help="Plik tekstowy z danymi (domyślnie: data_pl.txt)")
    train_parser.add_argument("--iters", "-i", type=int, default=500, help="Liczba iteracji treningu (domyślnie: 500)")
    train_parser.add_argument("--batch", "-b", type=int, default=32, help="Rozmiar paczki - batch size (domyślnie: 32)")
    train_parser.add_argument("--block", type=int, default=64, help="Okno kontekstowe (domyślnie: 64)")
    train_parser.add_argument("--embed", type=int, default=128, help="Rozmiar osadzeń (domyślnie: 128)")
    train_parser.add_argument("--layers", type=int, default=4, help="Liczba warstw transformera (domyślnie: 4)")
    
    # Subparser knowledge
    know_parser = subparsers.add_parser("knowledge", help="Zarządzaj lokalną bazą wiedzy RAG.")
    know_subparsers = know_parser.add_subparsers(dest="know_action", help="Akcje bazy wiedzy:")
    
    know_subparsers.add_parser("list", help="Lista wgranych plików.")
    
    add_parser = know_subparsers.add_parser("add", help="Zindeksuj nowy plik tekstowy.")
    add_parser.add_argument("file", type=str, help="Ścieżka do pliku (.txt lub .md)")
    
    del_parser = know_subparsers.add_parser("delete", help="Usuń plik z indeksu.")
    del_parser.add_argument("filename", type=str, help="Nazwa pliku w bazie do usunięcia")
    
    search_parser = know_subparsers.add_parser("search", help="Przeszukaj bazę pod kątem frazy.")
    search_parser.add_argument("query", type=str, help="Zapytanie wyszukiwania")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        args.func()
        return
        
    if args.command == "chat":
        backend_active = check_backend() and not args.standalone
        system_prompt = args.system
        if system_prompt is None:
            # Map default prompt
            default_prompts = {
                'mey:1.0': 'Jesteś Mey 1.0, zaawansowanym i niezwykle inteligentnym asystentem AI stworzonym przez programistę Eryka. Odpowiadasz zawsze w języku polskim.',
                'mey:1.0-nano': 'Jesteś Mey 1.0 Nano, szybką, lekką wersją asystenta AI Mey stworzoną przez programistę Eryka.',
                'mey:pro': 'Jesteś Mey 1.0 Pro, zaawansowanym asystentem AI z dostępem do wyszukiwarki internetowej. Wykorzystuj wyszukane wyniki i podawaj linki źródłowe.',
                'mey:reasoning': 'Jesteś własnym lokalnym modelem reasoningowym Mey. Analizuj pytanie wewnętrznie i nie pokazuj toku rozumowania.',
                'mey_custom': 'Jesteś Mey Custom, modelem Transformer wyszkolonym lokalnie.'
            }
            system_prompt = default_prompts.get(args.model)
            
        if args.model in MODEL_MAP and ollama_available():
            run_chat_ollama(args.model, args.temp, system_prompt)
        elif backend_active:
            run_chat_backend(args.model, args.temp, args.rag, system_prompt)
        else:
            run_chat_standalone(args.model, args.temp, args.rag, system_prompt)
            
    elif args.command == "knowledge":
        if not args.know_action:
            know_parser.print_help()
            sys.exit(0)
        manage_knowledge(args.know_action, args)
        
    elif args.command == "train":
        run_training(args)

if __name__ == "__main__":
    main()
