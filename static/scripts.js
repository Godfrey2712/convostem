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