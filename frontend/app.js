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

// --- Chat Feature ---
document.getElementById('chat-send-btn').onclick = async function() {
    const input = document.getElementById('chat-input');
    const chat = document.getElementById('chat-container');
    const personality = document.getElementById('personality-selector').value;
    if (input.value.trim()) {
        const userMsg = document.createElement('div');
        userMsg.className = 'chat-user';
        userMsg.textContent = 'You: ' + input.value;
        chat.appendChild(userMsg);
        // Send to backend
        const res = await fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: input.value, personality })
        });
        const data = await res.json();
        const aiMsg = document.createElement('div');
        aiMsg.className = 'chat-ai';
        aiMsg.textContent = 'AI: ' + data.response;
        chat.appendChild(aiMsg);
        input.value = '';
    }
};

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
