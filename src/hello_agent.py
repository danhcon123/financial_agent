import sys
from pathlib import Path

# Add the current directory to Python path so we can import our config
sys.path.insert

from langchain.llms import Ollama

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config.settings import settings
# Our settings file - loads Ollama URL, model name, temperature


# GLOBAL SETUP
console = Console() # printer

# CORE FUNCTIONS
def initialize_llm():
    """ 
    Connect to Ollama and create an LLM instance 
    
    What happens here:
    1. LangChain creates an Ollama client
    2. Client will connect to http://localhost:11434
    3. Every time we call llm("question"), it sends HTTP POST to Ollama
    
    Returns:
        Ollama: An LLM object ready to answer questions
        
    Raises:
        Exception: If Ollama is not running or model not found
"""

    console.print("\n[cyan]🔌 Connecting to Ollama...[/cyan]")
    console.print(f"   URL: {settings.OLLAMA_BASE_URL}")
    console.print(f"   Model: {settings.OLLAMA_MODEL}")
    console.print(f"   Temperature: {settings.OLLAMA_TEMPERATURE}")

    try:
        # Create the LLM instance
        llm = Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.OLLAMA_TEMPERATURE,
        )
        
        # Test with a simple query
        console.print("\n[yellow]🧪 Testing connection...[/yellow]")
        test_response = llm("Say 'Hello' in one word")
        
        console.print(f"[green]✅ Connection successful![/green]")
        console.print(f"  Test response: {test_response[:50]}...")
        return llm
    except Exception as e:
        console.print(f"\n[red]❌ Failed to connect to Ollama[/red]")        
        console.print(f"[red]Error: {e}[/red]\n")