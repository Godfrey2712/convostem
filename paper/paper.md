---
title: 'Intentifier: An Open-Source Tool for Evaluating Illocutionary Forces in Dialogue using Open-Source Large Language Models'
tags:
  - Python
  - linuistics
  - dialogue
  - large language models
  - illocutionary force
authors:
  - name: Godfrey Inyama
    orcid: 0009-0007-6272-3879
date: 3 April 2026
bibliography: paper.bib
---

# Summary

Intentifier is an advanced natural language processing research tool that integrates dialogue analysis with illocutionary force detection. 
The Tool responds to user queries using various language models while simultaneously classifying the underlying communicative intent (illocutionary force) embedded in user inputs. 
It is built with Python using the Flask web framework and supports integration with open-source (Ollama) language models.

# Statement of need

Traditional conversational AI systems focus exclusively on generating appropriate responses without 
accounting for the illocutionary force which is the underlying communicative intent or speech act encoded in user utterances. 
Understanding illocutionary forces like directive, question, assertion, or commissive
is fundamental to pragmatic language understanding and more nuanced dialogue interaction.

# State of the field                                                                                                                  

Most chatbots ignore the pragmatic dimension of language, treating all inputs uniformly regardless of communicative intent. 
Existing systems seldom combine multiple evaluation metrics to assess both semantic accuracy and pragmatic appropriateness of responses.
There is limited support for dynamic knowledge base management where users can easily upload and switch between knowledge sources during interaction.
Users typically cannot observe or analyze the system's classification of speaker intent, limiting educational and research applications.
Austin introduced the foundational theory [@austin1962speech] whereas Searle carried out the formalization of illocutionary forces [@searle1969speech]

For some dialogue systems, evaluation Metrics still uses bilingual evaluation (BLEU), ROUGE [@lin-2004-rouge],
METEOR [@banerjee-lavie-2005-meteor], and BERTScore [@devlin2018bert] which we adopt for benchmarking. 

The current approach leverages using LLMs [@chavan2022large, @r-etal-2024-shot, @yu2024breakingceilingllmcommunity]

# Software design

## 1. Web Interface (Frontend)
Technologies: HTML5, CSS3, JavaScript, Chart.js

Key Features:

- Chat Container: Real-time bidirectional messaging
- Control Panel: Dropdowns for API/model selection
- Knowledge Management: File upload and selection interface
- Illocutionary Force Visualization: Pie chart showing distribution of speech acts
- Evaluation Display: Metrics dropdown showing BLEU, ROUGE, METEOR, Perplexity, BERT scores
- Input Methods: Text input, speech recognition (Web Speech API), text-to-speech output
- Customization: Bot name modification with real-time updates

## 2. Backend Application (app_ollama.py)
`Intentifier` is implemented as a Flask-based Python framework (≈18 KB core application logic) that enables dialogue-driven interaction through a structured, end-to-end processing pipeline: upon startup, it initializes multiple pre-trained models including a Hugging Face BERT-based intent classifier fine-tuned for illocutionary force recognition, a GPT-2 model for perplexity computation, a BERT model for BERTScore evaluation, and Sentence Transformers for semantic similarity; user requests are handled via a `/get_response` POST route that captures input, classifies intent such as asserting, pure questioning etc, and can optionally integrates external knowledge from PDF or TXT files, generates responses through a flexible API abstraction layer supporting Ollama-hosted models locally like LLaMA, Mistral with runtime switching, evaluates responses using a comprehensive suite of metrics (BLEU, ROUGE-1/2/L, METEOR, perplexity, BERTScore, and cosine-based QA similarity), maintains structured chat and illocutionary force histories, updates real-time visualizations, and returns enriched outputs to the frontend; additionally, the system includes knowledge base management endpoints for uploading and selecting documents, as well as chat export functionality, all built on a stack of key dependencies including Flask, transformers, torch, nltk, rouge-score, sentence-transformers, PyPDF2, Ollama, and the OpenAI API, enabling a unified platform for analysing, generating, and evaluating mixed-initiative dialogue in human-AI interaction systems.

# Research impact statement

`Intentifier` has been able to effect the use of a fine-tuned illocutionary force classification model with over 1k downloads [@godfrey_inyama_2026_intent].
A paper have being deposited on arxiv which shows the impact of using the tool on two custom documents and the results analysed from the tool [@i2025opensourcewebbasedtoolevaluating]


# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.

# References
