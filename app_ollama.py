# app.py

from flask import Flask, render_template, request, Response, stream_with_context
import ollama  # Replace this with the actual local Ollama module if different
import re
from transformers import pipeline
import time

app = Flask(__name__)

# Load knowledge document
with open("knowledge.txt", "r", encoding="utf-8") as file:
    knowledge_text = file.read()

# Initialize chat history as an empty list
chat_history = []
illocutionary_force_history = []

# Load illocutionary force classification model from Hugging Face
classifier = pipeline("text-classification", model="Godfrey2712/amf_illoc_force_intent_recognition")

# Variable to determine how much previous conversation to store
CONTEXT_WINDOW = 3  # Number of previous exchanges to include

@app.route("/")
def home():
    logo_path = "/templates/ConvoStem_NoBackground.png"
    return render_template("index.html", chat_history=chat_history, logo_path=logo_path, illocutionary_force_history=illocutionary_force_history)

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    
    # Get illocutionary force classification
    illocutionary_force = classifier(user_input)[0]['label']
    #print(f"Illocutionary Force: {illocutionary_force}")

    # Append the illocutionary force to the history
    illocutionary_force_history.append(illocutionary_force)

    # Concatenate user input into a single sentence
    user_input_single_sentence = " ".join(user_input.split(". ")).strip()

    # Generate response
    response, _ = generate_response(user_input_single_sentence, knowledge_text, illocutionary_force)

    # Append user's input and assistant's response to the chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    #return response
    return render_template("index.html", user_input=user_input, response=response, chat_history=chat_history, illocutionary_force_history=illocutionary_force_history)

def generate_response(user_input, knowledge_text, illocutionary_force):
    # Formulate the single-sentence instruction
    #instruction = f"Answer the query '{user_input}' using the knowledge '{knowledge_text}', considering the user's intent '{illocutionary_force}'. If unsure, reply 'I AM NOT SURE'."

    # Build context from chat history
    context = ""
    for entry in chat_history[-CONTEXT_WINDOW*2:]:
        if entry['role'] == 'user':
            context += f"User Query: {entry['content']}\n"
        else:
            context += f"Chat bot Response: {entry['content']}\n"

    # Formulate the single-sentence instruction
    instruction = f"""
    Respond to the user's query: '{user_input}' within the context of the following conversation: '{context}'. Use only the information provided in this document: '{knowledge_text}'.
    Carefully consider the user's communicative intent, characterized by the illocutionary force: '{illocutionary_force}'.
    If you're uncertain about the appropriate response or if the document lacks relevant information, reply with 'I AM NOT SURE'.
    Aim to respond in a manner that is as human-like as possible, enhancing the user's perception of interacting with a real person.
    """

    # Creating a single string for the prompt
    prompt = f"""
    System: You are a helpful Automated Assistance Support Agent
    {context}
    User: {instruction}
    Knowledge: {knowledge_text}
    """

    # Call Ollama local model to generate response
    response = ollama.generate(
        #model="llama2:latest",  # Specify the Ollama model
        model="tinyllama:latest",
        prompt=instruction
    )

    print('======= \n', illocutionary_force)
    print ('======= \n', context)
    response_content = response.get('response', '').strip()

    # Check if the response contains 'I AM NOT SURE' (case-insensitive)
    if re.search(r'i am not sure', response_content, re.IGNORECASE):
        # Offer the option to chat with a Human
        response_content += '<div style="text-align: center; margin-top: 10px;">'
        response_content += 'Do you want to chat with a Human?&nbsp;&nbsp;&nbsp;'
        response_content += '<br><button onclick="sendEmail()">Yes</button>&nbsp;&nbsp;&nbsp;'
        response_content += '<button onclick="continueConversation()">No</button>'
        response_content += '</div>'
    
    return (response_content, illocutionary_force)

if __name__ == "__main__":
    app.run(debug=True)