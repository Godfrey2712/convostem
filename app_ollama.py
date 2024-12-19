# app.py

from flask import Flask, render_template, request, jsonify, make_response
import ollama
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import openai
import os
import transformers
import torch
import PyPDF2
import json
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import math
import time

openai.api_key = 'sk-mcy9j'  # Replace

app = Flask(__name__)

# Load knowledge document
knowledge_file_path = "knowledge/Baby_Health_Guide.txt"
if knowledge_file_path.endswith('.pdf'):
    with open(knowledge_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        knowledge_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.getPage(page_num)
            knowledge_text += page.extract_text()
else:
    with open(knowledge_file_path, "r", encoding="utf-8") as file:
        knowledge_text = file.read()

# Initialize chat history and illocutionary force history+
chat_history = []
illocutionary_force_history = []
include_illocutionary_force = True

# Load illocutionary force classification model from Hugging Face
classifier = pipeline("text-classification", model="Godfrey2712/amf_illoc_force_intent_recognition")

# Variable to determine how much previous conversation to store
CONTEXT_WINDOW = 3  # Number of previous exchanges to include

open_ai = True
hugging_face = False
selected_model = "gpt-4"  # Default model

#######################################################
model_id = "meta-llama/Meta-Llama-3-8B"

# Initialize the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id)

#tokenizer.chat_template = {
#    "system": "You are a helpful assistant.",
#    "user": "{user_input}",
#    "assistant": "{assistant_response}"
#}

text_generation_pipeline = pipeline(
    "text-generation",
    model=model_id,
    #tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

#######################################################

# Load models for new metrics
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

qa_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_perplexity(candidate):
    """
    Calculate perplexity of a response using GPT-2.
    """
    tokens = gpt2_tokenizer(candidate, return_tensors="pt")
    with torch.no_grad():
        output = gpt2_model(**tokens, labels=tokens["input_ids"])
    loss = output.loss
    perplexity = math.exp(loss.item())
    return perplexity

def calculate_bert_score(reference, candidate):
    """
    Calculate BERTScore (Precision, Recall, F1) between reference and candidate responses.
    """
    ref_tokens = bert_tokenizer(reference, return_tensors="pt", truncation=True, padding=True)
    cand_tokens = bert_tokenizer(candidate, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        ref_emb = bert_model(**ref_tokens).pooler_output
        cand_emb = bert_model(**cand_tokens).pooler_output

    cosine_similarity = torch.nn.functional.cosine_similarity(ref_emb, cand_emb, dim=-1).item()
    return {"Precision": cosine_similarity, "Recall": cosine_similarity, "F1": cosine_similarity}

def calculate_qa_similarity(question, reference, candidate):
    """
    Calculate cosine similarity of embeddings between the question and candidate/reference responses.
    """
    question_embedding = qa_model.encode(question)
    ref_embedding = qa_model.encode(reference)
    cand_embedding = qa_model.encode(candidate)

    ref_similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(question_embedding), torch.tensor(ref_embedding), dim=0).item()
    cand_similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(question_embedding), torch.tensor(cand_embedding), dim=0).item()

    return {"QA-Ref": ref_similarity, "QA-Cand": cand_similarity}


@app.route("/")
def home():
    logo_path = "/templates/ConvoStem_NoBackground.png"
    return render_template("index.html", chat_history=chat_history, logo_path=logo_path, illocutionary_force_history=illocutionary_force_history, evaluation_scores={})

@app.route("/download_chat", methods=["GET"])
def download_chat():
    chat_type = request.args.get('type', 'all')  # Default to 'all' if no type is provided

    # Filter chat history based on the type
    if chat_type == 'user':
        chat_content = "\n".join([msg['content'] for msg in chat_history if msg['role'] == 'user'])
    elif chat_type == 'bot':
        chat_content = "\n".join([msg['content'] for msg in chat_history if msg['role'] == 'assistant'])
    else:
        # Combine user and bot chat
        chat_content = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        )

    # Create a response to download the chat as a .txt file
    response = make_response(chat_content)
    response.headers["Content-Disposition"] = f"attachment; filename={chat_type}_chat.txt"
    response.headers["Content-Type"] = "text/plain"
    return response

def evaluate_response(reference, candidate, question):
    """
    Evaluate the chatbot response quality using BLEU, ROUGE, and METEOR scores.
    :param reference: List of reference texts (ground truth responses).
    :param candidate: Generated response by the chatbot.
    :return: Dictionary of BLEU, ROUGE, and METEOR scores.
    """
    if not candidate or not reference:
        return {
            "BLEU": 0, "ROUGE-1": 0, "ROUGE-2": 0, "ROUGE-L": 0, "METEOR": 0,
            "Perplexity": float('inf'), "BERT-Precision": 0, "BERT-Recall": 0, "BERT-F1": 0,
            "QA-Ref": 0, "QA-Cand": 0
        }
    
    # Tokenize candidate and references
    tokenized_candidate = word_tokenize(candidate)
    tokenized_references = [word_tokenize(ref) for ref in reference]

    # BLEU Score
    smooth_fn = SmoothingFunction().method4
    bleu = sentence_bleu(
        tokenized_references,
        tokenized_candidate,
        smoothing_function=smooth_fn
    )

    # ROUGE Scores (aggregate across references)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [
        rouge.score(ref, candidate) for ref in reference
    ]
    aggregated_rouge = {
        "rouge1": max(score['rouge1'].fmeasure for score in rouge_scores),
        "rouge2": max(score['rouge2'].fmeasure for score in rouge_scores),
        "rougeL": max(score['rougeL'].fmeasure for score in rouge_scores),
    }

    # METEOR Score
    meteor = max(meteor_score([ref], tokenized_candidate) for ref in tokenized_references)

    # Perplexity
    perplexity = calculate_perplexity(candidate)

    # BERTScore
    bert_scores = calculate_bert_score(reference[0], candidate)

    # QA Similarity
    qa_similarity = {"QA-Ref": 0, "QA-Cand": 0}
    if question:
        qa_similarity = calculate_qa_similarity(question, reference[0], candidate)

    print(f"Evaluation Scores - BLEU: {bleu}, ROUGE-1: {aggregated_rouge['rouge1']}, "
          f"ROUGE-2: {aggregated_rouge['rouge2']}, ROUGE-L: {aggregated_rouge['rougeL']}, METEOR: {meteor}, "
          f"Perplexity: {perplexity}, BERT-Precision: {bert_scores['Precision']}, "
          f"BERT-Recall: {bert_scores['Recall']}, BERT-F1: {bert_scores['F1']}, "
          f"QA-Ref: {qa_similarity['QA-Ref']}, QA-Cand: {qa_similarity['QA-Cand']}")

    return {
        "BLEU": bleu,
        "ROUGE-1": aggregated_rouge['rouge1'],
        "ROUGE-2": aggregated_rouge['rouge2'],
        "ROUGE-L": aggregated_rouge['rougeL'],
        "METEOR": meteor,
        "Perplexity": perplexity,
        "BERT-Precision": bert_scores["Precision"],
        "BERT-Recall": bert_scores["Recall"],
        "BERT-F1": bert_scores["F1"],
        "QA-Ref": qa_similarity["QA-Ref"],
        "QA-Cand": qa_similarity["QA-Cand"]
    }

def generate_references(response, knowledge_text):
    """
    Generate relevant references from the knowledge base based on the model's response.
    This function searches the knowledge base for sections related to the content of the response.
    """
    sentences = knowledge_text.split("\n")  # Split the knowledge text into sentences or paragraphs
    
    # Simple keyword-based matching (you can enhance this with semantic similarity or other methods)
    relevant_references = []
    for sentence in sentences:
        if any(word.lower() in sentence.lower() for word in response.split()):
            relevant_references.append(sentence)
    
    # Fallback if no references are found
    if not relevant_references:
        relevant_references = ["I could not find relevant information in the knowledge base."]
    
    return relevant_references

@app.route("/get_response", methods=["POST"])
def get_response():
    start_time = time.time()  # Measure the start time

    user_input = request.form["user_input"]
    
    # Get illocutionary force classification
    illocutionary_force = classifier(user_input)[0]['label']

    # Conditionally append the illocutionary force to the history
    if include_illocutionary_force:
        illocutionary_force_history.append(illocutionary_force)

    # Concatenate user input into a single sentence
    user_input_single_sentence = " ".join(user_input.split(". ")).strip()

    # Generate response
    raw_response, _ = generate_response(user_input_single_sentence, knowledge_text, illocutionary_force)

    # Generate relevant references from the knowledge base
    references = generate_references(raw_response, knowledge_text)

    evaluation_scores = evaluate_response(references, raw_response, user_input_single_sentence)

    # Append to chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": raw_response})

    end_time = time.time()  # Measure the end time
    total_time = end_time - start_time  # Calculate the total time taken
    print(f"Total time taken for /get_response: {total_time:.2f} seconds")  # Log the total time

    return render_template(
        "index.html",
        user_input=user_input,
        response=raw_response,
        chat_history=chat_history,
        illocutionary_force_history=illocutionary_force_history,
        evaluation_scores=evaluation_scores,  # Pass scores to the template
    )

@app.route("/update_api_selection", methods=["POST"])
def update_api_selection():
    global open_ai
    data = request.get_json()
    api_selection = data.get('api')
    if api_selection == 'openai':
        open_ai = True
    elif api_selection == 'ollama':
        open_ai = False
    return jsonify({"success": True, "api": api_selection})

@app.route("/update_model_selection", methods=["POST"])
def update_model_selection():
    global selected_model
    data = request.get_json()
    selected_model = data.get('model')
    return jsonify({"success": True, "model": selected_model})

@app.route("/upload_knowledge_file", methods=["POST"])
def upload_knowledge_file():
    file = request.files['file']
    file.save(os.path.join("knowledge", file.filename))
    return jsonify({"success": True, "file": file.filename})

@app.route("/get_knowledge_files", methods=["GET"])
def get_knowledge_files():
    files = os.listdir("knowledge")
    return jsonify({"files": files})

@app.route("/update_knowledge_file", methods=["POST"])
def update_knowledge_file():
    global knowledge_file_path, knowledge_text
    data = request.get_json()
    knowledge_file_path = os.path.join("knowledge", data.get('file'))
    
    if knowledge_file_path.endswith('.pdf'):
        with open(knowledge_file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            knowledge_text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                knowledge_text += page.extract_text()
                #print("=============.pdf", knowledge_text)
    else:
        with open(knowledge_file_path, "r", encoding="utf-8") as file:
            knowledge_text = file.read()
            #print("=============.txt", knowledge_text)
    
    return jsonify({"success": True, "file": knowledge_file_path})

@app.route("/update_illocutionary_force_option", methods=["POST"])
def update_illocutionary_force_option():
    global include_illocutionary_force
    data = request.get_json()
    #print(f"Received illocutionary force update request: {data}")  # Debug log
    include_illocutionary_force = data.get('includeIllocutionaryForce') == 'yes'
    #print(f"Updated includeIllocutionaryForce to: {include_illocutionary_force}")  # Debug log
    return jsonify({"success": True, "includeIllocutionaryForce": include_illocutionary_force})

def generate_response(user_input, knowledge_text, illocutionary_force):

    # Build context from chat history
    context = ""
    for entry in chat_history[-CONTEXT_WINDOW*2:]:
        if entry['role'] == 'user':
            context += f"user: {entry['content']}\n"
        else:
            context += f"assistant: {entry['content']}\n"

    # Use OpenAI models
    if open_ai:
        instruction = f"""
        Use the information from the provided document to answer the user's query: '{user_input}'.
        Context of the conversation: '{context}'
        """
        if include_illocutionary_force:
            instruction += f"Illocutionary force: '{illocutionary_force}'\n"
        instruction += "If no relevant information is found in the document, respond with 'I AM NOT SURE'."

        model_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": f"Knowledge: {knowledge_text}"}
        ]

        # Call OpenAI API to generate response
        response = openai.ChatCompletion.create(
            model=selected_model,  # Specify the GPT engine
            messages=model_prompt,
            max_tokens=4000,
            n=1,
            stop=None,
        )

        response_content = response['choices'][0]['message']['content']

    elif hugging_face:
        input_text = (
            f"Respond to the user's query: '{user_input}' within the context of the following conversation: '{context}'. "
            f"Use only the information provided in this document: '{knowledge_text}'. "
            f"Carefully consider the user's communicative intent, characterized by the illocutionary force: '{illocutionary_force}'. "
            f"If you're uncertain about the appropriate response or if the document lacks relevant information, reply with 'I AM NOT SURE'."
        )

        response = text_generation_pipeline(
            input_text, 
            #model=model_id, 
            max_new_tokens=512,
            #max_length=356,  # Set a reasonable total length if needed
            #chat_template="assistant",
        )
        response_content = response[0]['generated_text'][-1]

    else:
        # Formulate the updated instruction
        instruction = f"""
        Respond to the user's query: "{user_input}" while considering the relevant context from previous conversations.
        Please focus on providing a response based on the latest exchange, without repeating the entire conversation history.
        If needed, use the context provided below for reference.

        Relevant context: "{context}"

        Use only the information provided in this document to respond: '{knowledge_text}'.        
        """
        #f"Use only the information provided in this document: '{knowledge_text}'. "

        if include_illocutionary_force:
            instruction += f"Consider the user's communicative intent while responding, characterized by the illocutionary force: '{illocutionary_force}'."
        #instruction += "If the document lacks sufficient information or if you are unsure about the response, explicitly state 'I AM NOT SURE' and encourage further exploration."
        #print("=============User Input===========\n", user_input)
        #print("=============Context===========\n", context)
        # Call Ollama local model to generate response
        response = ollama.generate(
            model=selected_model,
            prompt=instruction,
            options={
                "temperature": 1,    # No randomness
                "top_p": 1,          # Consider all possible tokens
                "top_k": 1,          # Choose the highest-probability token
                "num_predict": 300,   # Set a fixed length for output
                "seed": 42,           # Ensure reproducibility (if seed is supported)
                "num_ctx": 4096,           # Maximize context window for tracking history
                "repeat_last_n": -1,      
                "repeat_penalty": 1.5,     
                "mirostat_tau": 1.0   
            },
            stream=False,            # Disable streaming for consistent output
            raw=False                # Enable prompt formatting for structured input
        )
        #print ("=============Response===========\n", response)
        response_content = response.get('response', '').strip()

    return (response_content, illocutionary_force)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

    '''
    # Check if the response contains 'I AM NOT SURE' (case-insensitive)
    if re.search(r'i am not sure', response_content, re.IGNORECASE):
        # Offer the option to chat with a Human
        response_content += '<div style="text-align: center; margin-top: 10px;">'
        response_content += 'Do you want to chat with a Human?&nbsp;&nbsp;&nbsp;'
        response_content += '<br><button onclick="sendEmail()">Yes</button>&nbsp;&nbsp;&nbsp;'
        response_content += '<button onclick="continueConversation()">No</button>'
        response_content += '</div>'
    '''