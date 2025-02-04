# ConvoStem

ConvoStem is an advanced natural language processing application designed to respond to user queries using various language models and the user's inherent illocutionary forces.

## Prerequisites

Before using ConvoStem, ensure you have the following installed on your system:

### Python

Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Ollama SDK

Install the Ollama SDK on your computer by following the installation instructions provided by [Ollama](https://ollama.com/).

## Installation

### Clone the Repository

To get started, clone the ConvoStem repository and navigate to the project directory:

```bash
git clone https://github.com/Godfrey2712/convostem.git
cd convostem
```

### Install Required Python Packages

Once inside the project directory, install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

### Install the Ollama SDK and Pull the Required Model

To use the Ollama SDK, install it and pull the required model using the following commands:

```bash
ollama install
ollama pull <model_name>
```

Replace `<model_name>` with the desired model name.

## Configuration

To specify the model used by ConvoStem, modify the `app_ollama.py` file and set the desired model:

```python
model = "<model_name>"
```

Replace `<model_name>` with the actual model you want to use.

## Usage

### Running the Application

To start ConvoStem, execute the following command:

```bash
python app_ollama.py
```

Once running, access the application in your browser at [http://localhost:5000](http://localhost:5000).

## Application Features

### User Interface

The application provides the following features for user interaction:

- **Bot Name Customization**: Users can change the bot's name via an input field and an update button.
- **Knowledge File Management**: Users can upload and select knowledge files that the bot can reference.
- **API and Model Selection**: Users can choose between OpenAI and Ollama APIs and select their preferred model.
- **Illocutionary Force Inclusion**: Users can decide whether to include illocutionary force in the bot's responses.
- **Download Chat**: Users can download the chat history for user, bot, or both.
- **Evaluation Scores**: The bot displays evaluation scores such as BLEU, ROUGE, METEOR, Perplexity, and BERT scores to assess response quality.

### Backend Functionality

The backend of ConvoStem provides several core functionalities:

- **Chat History**: Maintains the history of user-bot interactions.
- **Illocutionary Force Classification**: Classifies the illocutionary force of the user's input using a model from Hugging Face.
- **Response Generation**: Generates responses using the selected model and API (OpenAI or Ollama).
- **Evaluation**: Evaluates bot responses using various metrics, including:
  - **BLEU** (Bilingual Evaluation Understudy)
  - **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
  - **METEOR** (Metric for Evaluation of Translation with Explicit ORdering)
  - **Perplexity** (Measures how well the model predicts a sample)
  - **BERT Scores** (Evaluates semantic similarity between responses)
- **Knowledge File Handling**: Loads and updates knowledge files for use in generating responses.

## Scripts

The project includes several JavaScript scripts to handle UI interactions:

- **scripts.js**: Handles UI functionalities such as:
  - Updating API selection
  - Managing model options
  - Uploading files
  - Generating pie charts for illocutionary force distribution

## Templates

- **index.html**: The main HTML file that structures the application's user interface.

For more details, refer to the source code in the repository.

---

**Repository:** [GitHub - ConvoStem](https://github.com/Godfrey2712/convostem)
