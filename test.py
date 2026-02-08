# test_ollama.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from settings import settings

def test_basic_chat():
    """Test basic chat functionality"""
    print("Testing Ollama API with LangChain...")
    print(f"Model: {settings.OLLAMA_MODEL}")
    print(f"Base URL: {settings.OLLAMA_BASE_URL}\n")
    
    # Initialize the model
    llm = ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )
    
    # Test 1: Simple message
    print("Test 1: Simple question")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 2+2? Give a brief answer.")
    ]
    
    response = llm.invoke(messages)
    print(f"Response: {response.content}\n")
    
    # Test 2: Streaming response
    print("Test 2: Streaming response")
    print("Question: Tell me a short joke.")
    print("Response: ", end="", flush=True)
    
    for chunk in llm.stream([HumanMessage(content="Tell me a short joke.")]):
        print(chunk.content, end="", flush=True)
    print("\n")
    
    print("✓ All tests completed successfully!")

if __name__ == "__main__":
    try:
        test_basic_chat()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure Ollama is running and the model is pulled:")
        print(f"  ollama pull {settings.OLLAMA_MODEL}")