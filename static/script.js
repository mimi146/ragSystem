document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const fileUpload = document.getElementById('file-upload');
    const uploadStatus = document.getElementById('upload-status');
    const chatHistoryList = document.getElementById('chat-history-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const currentChatTitle = document.getElementById('current-chat-title');
    const modelSelect = document.getElementById('model-select');

    const DEFAULT_MODEL = 'gpt-4o';
    const savedModel = localStorage.getItem('weather_rag_model');
    if (savedModel && modelSelect.querySelector(`option[value="${savedModel}"]`)) {
        modelSelect.value = savedModel;
    } else {
        modelSelect.value = DEFAULT_MODEL;
        localStorage.setItem('weather_rag_model', DEFAULT_MODEL);
    }

    modelSelect.addEventListener('change', () => {
        localStorage.setItem('weather_rag_model', modelSelect.value);
    });

    let activeChatId = null;
    let chats = JSON.parse(localStorage.getItem('weather_rag_chats') || '{}');

    // Initialize marked options
    marked.setOptions({
        breaks: true,
        gfm: true
    });

    function saveChats() {
        localStorage.setItem('weather_rag_chats', JSON.stringify(chats));
    }

    function renderHistory() {
        chatHistoryList.innerHTML = '';
        const sortedChatIds = Object.keys(chats).sort((a, b) => chats[b].updatedAt - chats[a].updatedAt);
        
        sortedChatIds.forEach(id => {
            const li = document.createElement('li');
            li.className = `history-item ${id === activeChatId ? 'active' : ''}`;
            li.textContent = chats[id].title || 'New Conversation';
            li.onclick = () => switchChat(id);
            chatHistoryList.appendChild(li);
        });
    }

    function createNewChat() {
        const id = (typeof crypto.randomUUID === 'function') 
            ? crypto.randomUUID() 
            : ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
                (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
              );
        chats[id] = {
            id,
            title: 'New Conversation',
            messages: [],
            updatedAt: Date.now()
        };
        activeChatId = id;
        saveChats();
        renderHistory();
        renderMessages();
        currentChatTitle.textContent = 'New Conversation';
        userInput.focus();
    }

    function switchChat(id) {
        activeChatId = id;
        renderHistory();
        renderMessages();
        currentChatTitle.textContent = chats[id].title;
    }

    function renderMessages() {
        chatMessages.innerHTML = '';
        const messages = chats[activeChatId].messages;

        if (messages.length === 0) {
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <i class="fas fa-cloud-sun fa-3x"></i>
                    <h2>Weather RAG Assistant</h2>
                    <p>Upload documents to the knowledge base and ask me anything about them.</p>
                </div>
            `;
            return;
        }

        messages.forEach(msg => {
            appendMessage(msg.role, msg.content, false);
        });
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendMessage(role, content, save = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.innerHTML = role === 'ai' ? marked.parse(content) : `<p>${escapeHtml(content)}</p>`;
        
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        if (save && activeChatId) {
            chats[activeChatId].messages.push({ role, content });
            chats[activeChatId].updatedAt = Date.now();
            if (role === 'user' && chats[activeChatId].messages.length === 1) {
                chats[activeChatId].title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
                currentChatTitle.textContent = chats[activeChatId].title;
            }
            saveChats();
            renderHistory();
        }

        return bubbleDiv;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function escapeMarkdown(text) {
        return String(text).replace(/([\\`*_{}[\]()#+\-.!])/g, '\\$1');
    }

    function buildSourcesMarkdown(sources) {
        const sourceLines = sources
            .filter(source => source?.url && source?.filename)
            .map(source => {
                const chunkLabel = Number.isInteger(source.chunk) ? ` (chunk ${source.chunk})` : '';
                const label = `${escapeMarkdown(source.filename)}${chunkLabel}`;
                return `- [${label}](${source.url})`;
            });

        return sourceLines.length > 0 ? `---\n**Sources**\n${sourceLines.join('\n')}` : '';
    }

    function buildAiMarkdown(rawText, sources = []) {
        const sections = [];
        sections.push(rawText || '_No answer generated._');

        const sourcesSection = buildSourcesMarkdown(sources);
        if (sourcesSection) {
            sections.push(sourcesSection);
        }

        return sections.join('\n\n');
    }

    // Handle File Upload
    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.textContent = 'Uploading...';
        uploadStatus.style.color = 'var(--text-muted)';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/weather/upload-document', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                uploadStatus.textContent = `Success: ${data.total_chunks} chunks added.`;
                uploadStatus.style.color = 'green';
            } else {
                const error = await response.json();
                uploadStatus.textContent = `Error: ${error.detail || 'Upload failed'}`;
                uploadStatus.style.color = 'red';
            }
        } catch (err) {
            uploadStatus.textContent = 'Error: Connection failed';
            uploadStatus.style.color = 'red';
        }
    });

    // Handle Chat Submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        if (!activeChatId) createNewChat();

        userInput.value = '';
        appendMessage('user', query);

        // Create AI message placeholder and start typing indicator
        const aiBubble = appendMessage('ai', '', false);
        let rawResponse = '';
        let typingTimer = null;
        let typingDots = 0;
        // Start a lightweight typing animation while waiting for the stream
        typingTimer = setInterval(() => {
            typingDots = (typingDots + 1) % 4;
            const dots = '.'.repeat(typingDots);
            aiBubble.innerHTML = `<p>Waiting for response${dots}</p>`;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 350);

        try {
            const response = await fetch('/weather/ask-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    user_id: activeChatId, // Use activeChatId as user_id for session-based memory
                    n_results: 3,
                    model: modelSelect.value
                })
            });

            if (!response.ok) {
                let errorMessage = 'Failed to get response';
                try {
                    const error = await response.json();
                    errorMessage = error.detail || errorMessage;
                } catch (err) {
                    // Ignore JSON parsing failures for non-JSON responses.
                }
                throw new Error(errorMessage);
            }

            const sourcesHeader = response.headers.get('x-weather-sources');
            let sources = [];
            if (sourcesHeader) {
                try {
                    sources = JSON.parse(sourcesHeader);
                } catch (err) {
                    console.warn('Failed to parse sources header:', err);
                }
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            // Stop the typing indicator once the stream starts
            if (typingTimer) {
                clearInterval(typingTimer);
                typingTimer = null;
            }
            aiBubble.innerHTML = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const token = decoder.decode(value, { stream: true });
                rawResponse += token;
                aiBubble.innerHTML = marked.parse(buildAiMarkdown(rawResponse));
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            const finalMessage = buildAiMarkdown(rawResponse, sources);
            aiBubble.innerHTML = marked.parse(finalMessage);

            // Save the complete AI message
            chats[activeChatId].messages.push({ role: 'ai', content: finalMessage });
            chats[activeChatId].updatedAt = Date.now();
            saveChats();

        } catch (err) {
            if (typingTimer) {
                clearInterval(typingTimer);
                typingTimer = null;
            }
            aiBubble.textContent = 'Error: ' + err.message;
            aiBubble.style.color = 'red';
        }
    });

    newChatBtn.addEventListener('click', createNewChat);

    // Initial state
    if (Object.keys(chats).length > 0) {
        const sortedChatIds = Object.keys(chats).sort((a, b) => chats[b].updatedAt - chats[a].updatedAt);
        switchChat(sortedChatIds[0]);
    } else {
        createNewChat();
    }
});
