function sendEmail() {
    alert("Please send an email to support@updraftplus.com");
}

function continueConversation() {
    alert("Then please ask me something meaningful");
}

function scrollToBottom() {
    var chatContainer = document.getElementById("chat-container");
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

window.onload = scrollToBottom;

function speak(text) {
    var tempElement = document.createElement("div");
    tempElement.innerHTML = text;
    var strippedText = tempElement.textContent || tempElement.innerText || "";
    responsiveVoice.speak(strippedText, "UK English Male");
}

function startRecognition() {
    var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onresult = function(event) {
        var result = event.results[0][0].transcript;
        document.getElementById('user_input').value = result;
    };

    recognition.onerror = function(event) {
        console.error('Recognition error:', event.error);
    };
}

function toggleDropdown() {
    document.getElementById("myDropdown").classList.toggle("show");
}

// Close the dropdown if the user clicks outside of it
window.onclick = function(event) {
    if (!event.target.matches('.dropbtn')) {
        var dropdowns = document.getElementsByClassName("dropdown-content");
        for (var i = 0; i < dropdowns.length; i++) {
            var openDropdown = dropdowns[i];
            if (openDropdown.classList.contains('show')) {
                openDropdown.classList.remove('show');
            }
        }
    }
}

function updateBotName() {
    var botNameInput = document.getElementById("botNameInput").value;
    var botNameHeader = document.getElementById("botName");
    botNameHeader.textContent = botNameInput;
}

// Function to create pie chart for illocutionary forces
function createPieChart(labels, data) {
    var ctx = document.getElementById('forcePieChart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                label: 'Illocutionary Force Distribution',
                data: data,
                backgroundColor: [
                    '#FF6384',
                    '#36A2EB',
                    '#FFCE56',
                    '#4BC0C0',
                    '#9966FF',
                    '#FF9F40'
                ],
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(tooltipItem) {
                            var value = tooltipItem.raw;
                            return labels[tooltipItem.dataIndex] + ': ' + value;
                        }
                    }
                }
            }
        }
    });
}

function updateApiSelection() {
    var apiSelection = document.getElementById("apiSelection").value;
    fetch('/update_api_selection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ api: apiSelection })
    })
    .then(response => response.json())
    .then(data => {
        console.log('API selection updated:', data);
        updateModelOptions();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function updateModelOptions() {
    var apiSelection = document.getElementById("apiSelection").value;
    var modelSelection = document.getElementById("modelSelection");
    modelSelection.innerHTML = '';

    var models = [];
    if (apiSelection === 'openai') {
        models = ['', 'gpt-3.5-turbo', 'gpt-4', 'davinci', 'curie', 'babbage', 'ada'];
    } else if (apiSelection === 'ollama') {
        models = ['', 'tinyllama:latest', 'gemma2:9b', 'mixtral:latest', 'llama3-chatqa:latest', 'llama3:latest', 'llama2:13b', 'mistral:latest', 'llama2:latest'];
    }

    models.forEach(function(model) {
        var option = document.createElement("option");
        option.value = model;
        option.text = model;
        modelSelection.appendChild(option);
    });
}

function updateModelSelection() {
    var modelSelection = document.getElementById("modelSelection").value;
    fetch('/update_model_selection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: modelSelection })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Model selection updated:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function uploadFile() {
    var fileInput = document.getElementById('fileUpload');
    var file = fileInput.files[0];
    var formData = new FormData();
    formData.append('file', file);

    fetch('/upload_knowledge_file', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('File uploaded:', data);
        updateKnowledgeOptions();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function updateKnowledgeOptions() {
    fetch('/get_knowledge_files')
    .then(response => response.json())
    .then(data => {
        var knowledgeSelection = document.getElementById("knowledgeSelection");
        knowledgeSelection.innerHTML = '';

        data.files.forEach(function(file) {
            var option = document.createElement("option");
            option.value = file;
            option.text = file;
            knowledgeSelection.appendChild(option);
        });
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function updateKnowledgeFile() {
    var knowledgeSelection = document.getElementById("knowledgeSelection").value;
    fetch('/update_knowledge_file', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file: knowledgeSelection })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Knowledge file updated:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function updateIllocutionaryForceOption() {
    var includeIllocutionaryForce = document.getElementById("includeIllocutionaryForce").value;
    fetch('/update_illocutionary_force_option', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ includeIllocutionaryForce: includeIllocutionaryForce })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Illocutionary force option updated:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function toggleMenu() {
    var toolbar = document.getElementById("toolbar");
    if (toolbar.style.display === "block") {
        toolbar.style.display = "none";
    } else {
        toolbar.style.display = "block";
    }
}

function downloadChat(type) {
    fetch(`/download_chat?type=${type}`)
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;

            // Set the filename based on the type
            const filename = type === 'user' ? 'user_chat.txt' :
                             type === 'bot' ? 'bot_chat.txt' : 'all_chat.txt';

            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => console.error('Error downloading chat:', error));
}