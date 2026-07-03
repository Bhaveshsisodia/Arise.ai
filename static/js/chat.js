const messages = document.getElementById('messages');
const input = document.getElementById('question');
const button = document.getElementById('send');

function addMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'message ' + role;

    const label = document.createElement('strong');
    label.textContent = role === 'user' ? 'You' : role === 'bot' ? 'Assistant' : 'Error';
    el.appendChild(label);

    const content = document.createElement('div');
    content.className = 'message-text';
    content.textContent = text;
    el.appendChild(content);

    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
}

function addSources(sources) {
    const container = document.createElement('div');
    container.className = 'sources';
    container.innerHTML = '<strong>Sources</strong>';
    sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';
        item.innerHTML = `<div class="meta">Section: ${src.section} • Pages: ${src.pages}</div><div>${src.snippet}</div>`;
        container.appendChild(item);
    });
    messages.appendChild(container);
    messages.scrollTop = messages.scrollHeight;
}

async function sendQuestion() {
    const question = input.value.trim();
    if (!question) return;
    addMessage('user', question);
    input.value = '';
    button.disabled = true;
    addMessage('bot', 'Thinking...');

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        if (!response.ok) {
            const errorText = await response.text();
            addMessage('error', errorText);
        } else {
            const data = await response.json();
            const botText = data.answer || 'No answer returned.';
            addMessage('bot', botText);
            if (data.sources && data.sources.length) {
                addSources(data.sources);
            }
        }
    } catch (err) {
        addMessage('error', err.message);
    } finally {
        button.disabled = false;
    }
}

button.addEventListener('click', sendQuestion);
input.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') sendQuestion();
});
