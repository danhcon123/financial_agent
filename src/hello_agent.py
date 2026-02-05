import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ollama import chat  # ← Native ollama library
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config.settings import settings

# GLOBAL SETUP
console = Console()


def initialize_ollama():
    """Test connection to Ollama"""
    
    console.print("\n[cyan]🔌 Connecting to Ollama...[/cyan]")
    console.print(f"   Model: {settings.OLLAMA_MODEL}")
    console.print(f"   Temperature: {settings.OLLAMA_TEMPERATURE}")

    try:
        console.print("\n[yellow]🧪 Testing connection...[/yellow]")
        
        # Simple test without thinking
        response = chat(
            model=settings.OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': 'Say Hello'}],
        )
        
        console.print(f"[green]✅ Connection successful![/green]")
        console.print(f"  Test response: {response.message.content[:50]}...")
        return True
    except Exception as e:
        console.print(f"\n[red]❌ Failed to connect to Ollama[/red]")        
        console.print(f"[red]Error: {e}[/red]\n")
        return False


def chat_with_thinking(user_message, model):
    """
    Send a message and stream both thinking and response
    """
    stream = chat(
        model=model,
        messages=[{'role': 'user', 'content': user_message}],
        stream=True,
    )
    
    in_thinking = False
    content = ''
    thinking = ''
    
    for chunk in stream:
        # Handle thinking process
        if chunk.message.thinking:
            if not in_thinking:
                in_thinking = True
                console.print('\n[yellow]💭 Thinking:[/yellow]', flush=True)
            console.print(chunk.message.thinking, end='', style="dim yellow", flush=True)
            thinking += chunk.message.thinking
        
        # Handle actual response
        elif chunk.message.content:
            if in_thinking:
                in_thinking = False
                console.print('\n\n[bold green]Agent:[/bold green]', flush=True)
            console.print(chunk.message.content, end='', style="green", flush=True)
            content += chunk.message.content
    
    console.print('\n')  # New line at end
    return {'thinking': thinking, 'content': content}


def main():
    """Main entry point for the hello agent"""
    
    console.print(Panel.fit(
        "[bold cyan]🤖 Hello Agent - Ollama with Thinking[/bold cyan]\n"
        "[dim]See the model's reasoning process[/dim]",
        border_style="cyan"
    ))

    if not initialize_ollama():
        console.print("[red]Failed to initialize. Exiting.[/red]")
        return

    console.print("\n[green]Ready to chat! Type 'quit' or 'exit' to stop.[/green]\n")

    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")

        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print("\n[cyan]👋 Goodbye![/cyan]")
            break

        try:
            chat_with_thinking(user_input, settings.OLLAMA_MODEL)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")

        console.print()  # Add spacing


if __name__ == "__main__":
    main()