#!/usr/bin/env python3
"""
cli_chat.py — CLI interface for local OpenVINO model execution.
Loads configuration from workspace .env file, prompts for model path if not set,
respects customized context length (MAX_INPUT_TOKENS), and runs a colorized interactive chat loop.
"""

import os
import sys
import time
import readline  # For terminal input editing and history support
from dotenv import load_dotenv

# ANSI escape codes for premium terminal styling
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_GREEN = "\033[32m"
CLR_YELLOW = "\033[33m"
CLR_BLUE = "\033[34m"
CLR_MAGENTA = "\033[35m"
CLR_CYAN = "\033[36m"
CLR_RESET_UNDERLINE = "\033[24m"
CLR_RED = "\033[31m"

def print_banner():
    banner = f"""
{CLR_BOLD}{CLR_MAGENTA}=============================================================
             🧠 local-ai-workstation CLI Chat
============================================================={CLR_RESET}
    """
    print(banner)

def main():
    print_banner()

    # Load workspace .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"{CLR_GREEN}Loaded configuration from {env_path}{CLR_RESET}")
    else:
        print(f"{CLR_YELLOW}No .env file found at {env_path}. Using default settings.{CLR_RESET}")

    # Read config values
    device = os.getenv("DEVICE", "AUTO")
    max_new_tokens = os.getenv("MAX_NEW_TOKENS", "2048")
    context_length = os.getenv("MAX_INPUT_TOKENS", "1024")
    MODEL_PATH = os.getenv("MODEL_PATH", "").strip()

    # Prompt user for model path if not defined in .env
    if not MODEL_PATH:
        print(f"\n{CLR_YELLOW}Warning: MODEL_PATH is not defined in your .env file.{CLR_RESET}")
        while True:
            try:
                user_path = input(f"{CLR_BOLD}{CLR_CYAN}Enter path to your local OpenVINO model directory: {CLR_RESET}").strip()
                if not user_path:
                    continue
                expanded_path = os.path.expanduser(user_path)
                if os.path.isdir(expanded_path):
                    MODEL_PATH = expanded_path
                    break
                else:
                    print(f"{CLR_RED}Directory not found: '{user_path}'{CLR_RESET}")
            except KeyboardInterrupt:
                print(f"\n{CLR_YELLOW}Exiting CLI Chat.{CLR_RESET}")
                sys.exit(0)

    # Absolute path verification
    MODEL_PATH = os.path.abspath(MODEL_PATH)
    if not os.path.isdir(MODEL_PATH):
        print(f"{CLR_RED}Error: The specified model directory does not exist: {MODEL_PATH}{CLR_RESET}")
        sys.exit(1)

    print(f"\n{CLR_BOLD}--- Configuration ---{CLR_RESET}")
    print(f"📁 Model Path:    {CLR_BLUE}{MODEL_PATH}{CLR_RESET}")
    print(f"⚡ Device:        {CLR_BLUE}{device}{CLR_RESET}")
    print(f"📏 Context Limit: {CLR_BLUE}{context_length} tokens{CLR_RESET} (MAX_INPUT_TOKENS)")
    print(f"🪙 Max Response:  {CLR_BLUE}{max_new_tokens} tokens{CLR_RESET} (MAX_NEW_TOKENS)")
    print("---------------------\n")

    # Import engine manager (done after dotenv setup to ensure engine reads correct values)
    try:
        from model_engine import MultiModelManager
    except ImportError as e:
        print(f"{CLR_RED}Error: Failed to import model_engine.py. Are you in the correct directory?{CLR_RESET}")
        print(f"Details: {e}")
        sys.exit(1)

    print(f"{CLR_YELLOW}Initializing model engine and loading model...{CLR_RESET}")
    print(f"{CLR_YELLOW}(Note: First startup compilation on device '{device}' can take up to a minute){CLR_RESET}")
    
    engine = MultiModelManager()
    start_load = time.time()
    
    try:
        result = engine.load_model(MODEL_PATH, device=device)
        if not result["success"]:
            print(f"\n{CLR_RED}❌ Failed to load model: {result['message']}{CLR_RESET}")
            sys.exit(1)
        
        load_duration = time.time() - start_load
        print(f"\n{CLR_GREEN}✅ Model loaded successfully in {load_duration:.2f}s!{CLR_RESET}")
        print(f"Loaded Model: {CLR_BOLD}{result['model_name']}{CLR_RESET}\n")
    except Exception as e:
        print(f"\n{CLR_RED}❌ Unexpected error during model compilation: {e}{CLR_RESET}")
        sys.exit(1)

    # Interactive Chat loop
    print(f"{CLR_BOLD}{CLR_MAGENTA}Chat loop active. Commands:{CLR_RESET}")
    print(f"  {CLR_CYAN}/exit{CLR_RESET} or {CLR_CYAN}/quit{CLR_RESET} - Exit chat session")
    print(f"  {CLR_CYAN}/clear{CLR_RESET}             - Clear conversation history")
    print(f"  {CLR_CYAN}/rag{CLR_RESET}               - Toggle local Notes RAG retrieval")
    print(f"  {CLR_CYAN}/web{CLR_RESET}               - Toggle live Web Search RAG retrieval")
    print(f"  {CLR_CYAN}/help{CLR_RESET}              - Show this help message")
    print(f"-------------------------------------------------------------")

    history = []
    rag_enabled = False
    web_enabled = False

    while True:
        try:
            # Colorized user prompt
            user_input = input(f"\n{CLR_BOLD}{CLR_GREEN}👤 User: {CLR_RESET}").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ('/exit', '/quit'):
                print(f"{CLR_YELLOW}Exiting. Goodbye!{CLR_RESET}")
                break
            elif user_input.lower() == '/clear':
                history = []
                print(f"{CLR_YELLOW}🧹 Chat history cleared.{CLR_RESET}")
                continue
            elif user_input.lower() == '/rag':
                rag_enabled = not rag_enabled
                print(f"{CLR_GREEN}📚 Local RAG (Notes Retrieval): {'ENABLED' if rag_enabled else 'DISABLED'}{CLR_RESET}")
                continue
            elif user_input.lower() == '/web':
                web_enabled = not web_enabled
                print(f"{CLR_GREEN}🌐 Web RAG (Live Search): {'ENABLED' if web_enabled else 'DISABLED'}{CLR_RESET}")
                continue
            elif user_input.lower() == '/help':
                print(f"\n{CLR_BOLD}Available Commands:{CLR_RESET}")
                print(f"  /exit, /quit - Terminate CLI session")
                print(f"  /clear       - Reset conversation history")
                print(f"  /rag         - Toggle retrieval from Mousepad Notes")
                print(f"  /web         - Toggle live search query on the Web")
                print(f"  /help        - Display options")
                continue

            # Add message to history
            history.append({"role": "user", "content": user_input})

            active_history = history
            context_block = ""
            
            # If local RAG is enabled, retrieve context and inject it
            if rag_enabled:
                from rag_engine import BM25Retriever
                # Determine notes directory
                notes_dir = os.path.abspath(os.path.join(os.getcwd(), "notes"))
                config_path = os.path.abspath(os.path.join(os.getcwd(), "notes_config.json"))
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                            custom_path = config.get("notes_dir")
                            if custom_path:
                                notes_dir = os.path.abspath(os.path.expanduser(custom_path))
                    except Exception:
                        pass
                
                try:
                    retriever = BM25Retriever(notes_dir)
                    results = retriever.retrieve(user_input, top_k=3)
                    if results:
                        print(f"{CLR_YELLOW}🔍 [RAG] Matching local notes found: {', '.join([r['filename'] for r in results])}{CLR_RESET}")
                        context_lines = []
                        for r in results:
                            context_lines.append(f"[File: {r['filename']}]\n{r['content']}")
                        context_str = "\n\n".join(context_lines)
                        context_block += (
                            "[Context retrieved from Mousepad Notes]\n"
                            "----------------------------------------\n"
                            f"{context_str}\n"
                            "----------------------------------------\n\n"
                        )
                except Exception as e:
                    print(f"{CLR_RED}⚠️ Local RAG Search Error: {e}{CLR_RESET}")

            # If Web search is enabled, retrieve live results
            if web_enabled:
                try:
                    from rag_engine import retrieve_web_context
                    web_results = retrieve_web_context(user_input, top_k=3)
                    if web_results:
                        print(f"{CLR_YELLOW}🔍 [Web RAG] Search queries matched: {', '.join([w['title'] for w in web_results])}{CLR_RESET}")
                        context_lines = []
                        for w in web_results:
                            context_lines.append(f"[Source URL: {w['url']}]\nTitle: {w['title']}\nSnippet: {w['snippet']}")
                        context_str = "\n\n".join(context_lines)
                        context_block += (
                            "[Context retrieved from Web Search]\n"
                            "----------------------------------------\n"
                            f"{context_str}\n"
                            "----------------------------------------\n\n"
                        )
                except Exception as e:
                    print(f"{CLR_RED}⚠️ Web RAG Search Error: {e}{CLR_RESET}")

            if context_block:
                context_block += (
                    "Based on the context retrieved above, please answer the user's request.\n"
                    "User request: "
                )
                active_history = [dict(h) for h in history]
                active_history[-1]["content"] = context_block + user_input

            # Streaming response output
            print(f"{CLR_BOLD}{CLR_MAGENTA}🤖 Assistant: {CLR_RESET}", end="", flush=True)
            assistant_response = ""
            meta = None
            
            for chunk in engine.generate_stream(active_history):
                if isinstance(chunk, dict) and "__meta__" in chunk:
                    meta = chunk["__meta__"]
                else:
                    print(chunk, end="", flush=True)
                    assistant_response += chunk
            print() # Print final newline after stream completes

            if meta:
                tokens = meta.get("tokens", 0)
                elapsed = meta.get("elapsed_sec", 0.0)
                tps = meta.get("tokens_per_sec", 0.0)
                print(f"{CLR_YELLOW}⚡ [{tokens} tokens | {elapsed:.2f}s | {tps} tokens/sec]{CLR_RESET}")

            # Save response to history
            if assistant_response.strip():
                history.append({"role": "assistant", "content": assistant_response})
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully without crash
            print(f"\n{CLR_YELLOW}Interrupted. Type /exit to quit or /clear to reset history.{CLR_RESET}")
        except Exception as e:
            print(f"\n{CLR_RED}Error: {e}{CLR_RESET}")

if __name__ == "__main__":
    main()
