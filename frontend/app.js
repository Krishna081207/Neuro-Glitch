// NeuroGlitch Frontend (vanilla JS placeholder)
document.getElementById('send-btn').onclick = function() {
    const input = document.getElementById('journal-input');
    const chat = document.getElementById('chat-container');
    if (input.value.trim()) {
        const entry = document.createElement('div');
        entry.textContent = input.value;
        chat.appendChild(entry);
        input.value = '';
        // TODO: Send entry to backend for sentiment analysis
    }
};
