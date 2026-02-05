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

// Dashboard: Model Training Form Submission
const trainForm = document.getElementById('train-form');
const trainStatus = document.getElementById('train-status');
if (trainForm) {
    trainForm.onsubmit = async function(e) {
        e.preventDefault();
        trainStatus.textContent = 'Submitting training job...';
        const name = document.getElementById('model-name').value;
        const description = document.getElementById('model-desc').value;
        const dataset_url = document.getElementById('dataset-url').value;
        const model_type = document.getElementById('model-type').value;
        try {
            const response = await fetch('http://localhost:8000/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, description, dataset_url, model_type })
            });
            const data = await response.json();
            if (response.ok) {
                trainStatus.textContent = 'Training job submitted!';
                // Optionally show job details
                trainStatus.textContent += '\nJob: ' + JSON.stringify(data.job);
            } else {
                trainStatus.textContent = 'Error: ' + (data.detail || 'Failed to submit job');
            }
        } catch (err) {
            trainStatus.textContent = 'Network error: ' + err.message;
        }
    };
}

// API call example: send data to backend
function sendDataToBackend(data) {
    fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        console.log('Backend response:', result);
        // Handle result (update UI, show recommendations, etc.)
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
