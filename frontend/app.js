// UI Enhancement: Handle journal analysis and display risk
document.getElementById('mood-send-btn').addEventListener('click', async () => {
    const journalText = document.getElementById('journal-input').value;
    if (!journalText.trim()) {
        document.getElementById('risk-result').textContent = 'Please enter your journal text.';
        return;
    }
    document.getElementById('risk-result').textContent = 'Analyzing...';
    const data = await sendJournalEntry(journalText);
    document.getElementById('risk-result').textContent = `Risk Level: ${data.risk_level}`;
});
// Sample function to send journal text to backend and display risk level
async function sendJournalEntry(journalText) {
    const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: journalText })
    });
    const data = await response.json();
    console.log('Risk Level:', data.risk_level);
    // Display risk level in the UI as needed
    return data;
}

// Example usage:
// sendJournalEntry("I feel like I'm drowning in my work and can't see a way out.");
// --- Navigation ---
const sections = ['chat', 'mood', 'tools', 'resources', 'auth', 'profile'];
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.onclick = () => {
        sections.forEach(sec => {
            document.getElementById(sec + '-section').style.display = 'none';
        });
        document.getElementById(btn.dataset.section + '-section').style.display = '';
    };
});

// --- Theme Switcher ---
const themeMap = {
    'light-theme': 'light',
    'dark-theme': 'dark',
    'mint-theme': 'mint'
};
Object.keys(themeMap).forEach(id => {
    document.getElementById(id).onclick = () => {
        document.body.className = themeMap[id] + '-mode';
    };
});

const API_BASE = window.__API_BASE__ || 'http://localhost:8000';
const chatInput = document.getElementById('chat-input');
const chatContainer = document.getElementById('chat-container');
const personalitySelector = document.getElementById('personality-selector');
const chatSendBtn = document.getElementById('chat-send-btn');

function getSessionId() {
    let sessionId = localStorage.getItem('ng_session_id');
    if (!sessionId) {
        sessionId = 'session-' + Math.random().toString(36).slice(2, 10);
        localStorage.setItem('ng_session_id', sessionId);
    }
    return sessionId;
}

async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) {
        return;
    }

    const personality = personalitySelector.value;
    const session_id = getSessionId();

    const userMsg = document.createElement('div');
    userMsg.className = 'chat-user';
    userMsg.textContent = 'You: ' + message;
    chatContainer.appendChild(userMsg);

    const typingMsg = document.createElement('div');
    typingMsg.className = 'chat-ai';
    typingMsg.textContent = 'AI: ...';
    chatContainer.appendChild(typingMsg);

    chatInput.value = '';

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, personality, session_id })
        });

        let data = {};
        try {
            data = await res.json();
        } catch (err) {
            data = {};
        }

        if (!res.ok) {
            typingMsg.textContent = 'AI error: ' + (data.detail || res.statusText || 'Unknown error');
            return;
        }

        typingMsg.textContent = 'AI: ' + (data.response || 'No response');
    } catch (err) {
        typingMsg.textContent = 'AI error: Network error';
    }
}

// --- Chat Feature ---
chatSendBtn.onclick = sendChatMessage;
chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
});

// --- Mood Dashboard ---
const moodScale = document.getElementById('mood-scale');
const moodValue = document.getElementById('mood-value');
moodScale.oninput = () => { moodValue.textContent = moodScale.value; };
document.getElementById('mood-send-btn').onclick = async function() {
    const mood = moodScale.value;
    const journal = document.getElementById('journal-input').value;
    const res = await fetch('http://localhost:8000/api/mood', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mood, journal })
    });
    const data = await res.json();
    updateMoodHistory();
};
async function updateMoodHistory() {
    const res = await fetch('http://localhost:8000/api/mood');
    const data = await res.json();
    const hist = document.getElementById('mood-history');
    hist.innerHTML = '';
    data.moods.forEach(entry => {
        const div = document.createElement('div');
        div.textContent = `Mood: ${entry.mood}, Journal: ${entry.journal}`;
        hist.appendChild(div);
    });
}
updateMoodHistory();

// --- Self-Help Tools ---
document.getElementById('yoga-btn').onclick = async function() {
    const mood = moodScale.value;
    const res = await fetch(`http://localhost:8000/api/yoga?mood=${mood}`);
    const data = await res.json();
    document.getElementById('tools-output').textContent = 'Yoga Poses: ' + data.poses.join(', ');
};
document.getElementById('focus-btn').onclick = async function() {
    const res = await fetch('http://localhost:8000/api/focus');
    const data = await res.json();
    document.getElementById('tools-output').textContent = `Focus Session: ${data.duration} min, Quote: ${data.quote}`;
};
document.getElementById('breathing-btn').onclick = async function() {
    const res = await fetch('http://localhost:8000/api/breathing');
    const data = await res.json();
    document.getElementById('tools-output').textContent = `Breathing: ${data.pattern}, ${data.instructions}`;
};
document.getElementById('coping-btn').onclick = async function() {
    const res = await fetch('http://localhost:8000/api/coping');
    const data = await res.json();
    document.getElementById('tools-output').textContent = 'Coping Cards: ' + data.cards.join(', ');
};
document.getElementById('wellness-btn').onclick = async function() {
    const res = await fetch('http://localhost:8000/api/wellness');
    const data = await res.json();
    document.getElementById('tools-output').textContent = 'Wellness Tips: ' + JSON.stringify(data.tips);
};

// --- Resource & Crisis Help ---
document.getElementById('hotlines-btn').onclick = async function() {
    const res = await fetch('http://localhost:8000/api/hotlines');
    const data = await res.json();
    document.getElementById('resources-output').textContent = 'Hotlines: ' + data.hotlines.join(', ');
};
document.getElementById('doctor-btn').onclick = async function() {
    const symptoms = prompt('Enter symptoms (comma separated):').split(',').map(s => s.trim());
    const res = await fetch('http://localhost:8000/api/doctor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms })
    });
    const data = await res.json();
    document.getElementById('resources-output').textContent = `Specialist: ${data.specialist}, Symptoms: ${data.symptoms.join(', ')}`;
};
document.getElementById('quizzes-btn').onclick = async function() {
    const res = await fetch('http://localhost:8000/api/quizzes');
    const data = await res.json();
    document.getElementById('resources-output').textContent = 'Quizzes: ' + data.quizzes.join(', ');
};

// --- Authentication ---
document.getElementById('login-btn').onclick = async function() {
    const username = document.getElementById('auth-username').value;
    const password = document.getElementById('auth-password').value;
    const res = await fetch('http://localhost:8000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    document.getElementById('auth-status').textContent = data.status + ' as ' + data.user;
};
document.getElementById('oauth-btn').onclick = async function() {
    const provider = prompt('Enter OAuth provider (Google, GitHub, Microsoft):');
    const res = await fetch('http://localhost:8000/api/oauth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(provider)
    });
    const data = await res.json();
    document.getElementById('auth-status').textContent = data.status + ' with ' + data.provider;
};
document.getElementById('guest-btn').onclick = async function() {
    const res = await fetch('http://localhost:8000/api/guest');
    const data = await res.json();
    document.getElementById('auth-status').textContent = 'Guest access: ' + data.features;
};

// --- User Profile ---
document.getElementById('save-profile-btn').onclick = async function() {
    const name = document.getElementById('profile-name').value;
    const picture_url = document.getElementById('profile-picture').value;
    const font_size = document.getElementById('profile-fontsize').value;
    const res = await fetch('http://localhost:8000/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, picture_url, font_size })
    });
    const data = await res.json();
    document.getElementById('profile-output').textContent = 'Profile saved!';
};
