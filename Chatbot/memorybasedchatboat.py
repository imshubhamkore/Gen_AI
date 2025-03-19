from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import Ollama
from langchain.chains import ConversationChain

# Load Ollama model (change model name if needed)
llm = Ollama(model="llama3.2")  # or "llama2" or any Ollama-supported model

# Initialize memory to store conversation history (window size = 3 messages)
memory = ConversationBufferWindowMemory(k=3)

# Create a conversation chain (links LLM and memory)
chatbot = ConversationChain(
    llm=llm,
    memory=memory
)

print("Chatbot is running! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot.run(user_input)
    print("Chatbot:", response)


