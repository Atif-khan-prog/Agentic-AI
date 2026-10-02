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

    try {

        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })
        });

        if (!response.ok) {
            throw new Error("Server error");
        }

        const data = await response.json();

        // Show RAG answer
        addMessage(data.answer, "bot");

    } catch (error) {

        console.error(error);

        addMessage(
            "Sorry, I couldn't connect to the FastAPI server.",
            "bot"
        );

    } finally {

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