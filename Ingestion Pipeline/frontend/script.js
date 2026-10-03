const API_URL = "https://agentic-ai-ativ.onrender.com";

const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-btn");
const chatBox = document.getElementById("chat-box");


async function sendMessage() {

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    // Show user's message
    addMessage(question, "user");

    // Clear input
    questionInput.value = "";

    // Disable button while waiting
    sendButton.disabled = true;
    sendButton.textContent = "Thinking...";

    // Render's free tier can take up to a minute to wake up
    const wakeHint = setTimeout(() => {
        sendButton.textContent = "Waking up server...";
    }, 8000);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000);

    try {

        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            }),

            signal: controller.signal
        });

        if (!response.ok) {
            throw new Error("Server error: " + response.status);
        }

        const data = await response.json();

        // Show RAG answer
        addMessage(data.answer, "bot");

    } catch (error) {

        console.error(error);

        const msg = error.name === "AbortError"
            ? "The server took too long to respond. Please try again."
            : "Sorry, I couldn't connect to the server.";

        addMessage(msg, "bot");

    } finally {

        clearTimeout(wakeHint);
        clearTimeout(timeout);

        sendButton.disabled = false;
        sendButton.textContent = "Send";
    }
}


function addMessage(text, type) {

    const message = document.createElement("div");

    message.classList.add("message", type);

    message.textContent = text;

    chatBox.appendChild(message);

    // Automatically scroll to newest message
    chatBox.scrollTop = chatBox.scrollHeight;
}


// Send when button is clicked
sendButton.addEventListener("click", sendMessage);


// Send when Enter is pressed
questionInput.addEventListener("keydown", function(event) {

    if (event.key === "Enter") {
        sendMessage();
    }

});