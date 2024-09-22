# app.py

from flask import Flask, render_template, request
import openai

app = Flask(__name__)

openai.api_key = 'sk-mcy9j'  # Replace 'your-api-key' with your actual OpenAI API key

# Load knowledge document
with open("knowledge.txt", "r", encoding="utf-8") as file:
    knowledge_text = file.read()


# Initialize chat history as an empty list
chat_history = []

@app.route("/")
def home():
    logo_path = "/templates/ConvoStem_NoBackground.png"
    return render_template("index.html", chat_history=chat_history, logo_path=logo_path)

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = generate_response(user_input, knowledge_text)

    # Append user's input and assistant's response separately to avoid duplication
    chat_history.append({"role": "user", "content": f"<strong>USER:</strong> {user_input}"})
    chat_history.append({"role": "assistant", "content": f"<img src='../static/ConvoStem_NoBackground.png' alt='ConvoStem Logo' style='width: 22px; height: 25px;'> <strong>BOT:</strong> {response}"})

    return render_template("index.html", user_input=user_input, response=response, chat_history=chat_history)


def generate_response(user_input, knowledge_text):
    # Combine user input and knowledge for context
    instruction = f"Use the information from {knowledge_text} only to answer {user_input}. If no information found in {knowledge_text}, just respond 'I AM NOT SURE' only. Except you need more information to understand a problem or a statement is made."

    context = [
        {"role": "system", "content": "You are a helpful Octopus Energy assistant."},
        {"role": "user", "content": f"User: {instruction}"},
        {"role": "assistant", "content": f"Assistant: Knowledge - {knowledge_text}"}
    ]

    # Call OpenAI API to generate response
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Specify the GPT engine
        messages=context,
        max_tokens=4000,
        n=1,
        stop=None,
    )

    response_content = response['choices'][0]['message']['content']

    # Check if the response is 'I DO NOT KNOW'
    if response_content.strip().lower() == 'i am not sure':
        # Offer the option to chat with a Human
        response_content += '<div style="text-align: center; margin-top: 10px;">'
        response_content += 'Do you want to chat with a Human?&nbsp;&nbsp;&nbsp;'
        response_content += '<br><button onclick="sendEmail()">Yes</button>&nbsp;&nbsp;&nbsp;'
        response_content += '<button onclick="continueConversation()">No</button>'
        response_content += '</div>'
    
    return response_content

if __name__ == "__main__":
    app.run(debug=True)