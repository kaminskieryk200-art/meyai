// Mey AI Frontend Application

// State variables
let currentScreen = 'chat';
let messageHistory = [];
let defaultSystemPrompts = {
    'mey:1.0': 'Jesteś Mey 1.0, zaawansowanym i niezwykle inteligentnym asystentem AI stworzonym przez programistę Eryka. Twoim celem jest pomagać mu w kodowaniu, kreatywnym pisaniu, analizie oraz odpowiadać na pytania w mądry i wyczerpujący sposób. Odpowiadasz zawsze w języku polskim w sposób naturalny i profesjonalny.',
    'mey:1.0-nano': 'Jesteś Mey 1.0 Nano, szybką, lekką i zoptymalizowaną wersją asystenta AI Mey stworzoną przez programistę Eryka. Odpowiadasz po polsku w sposób zwięzły, konkretny, celny i pomocny, idealnie sprawdzając się na słabszym sprzęcie.',
    'mey:pro': 'Jesteś Mey 1.0 Pro, zaawansowanym i niezwykle inteligentnym asystentem AI wyposażonym w dostęp do wyszukiwarki internetowej. Twoim celem jest pomagać Erykowi, wykorzystując najnowsze informacje z sieci. Odpowiadasz zawsze w języku polskim, powołując się na źródła i podając linki z wyników wyszukiwania.',
    'mey_custom': 'Jesteś Mey Custom, modelem Transformer wyszkolonym lokalnie przez Eryka.'
};
let customSystemPrompt = '';
let systemConnected = false;
let trainingPollInterval = null;
let lossChart = null;

// Speech synthesis
const synth = window.speechSynthesis;
let currentUtterance = null;

// DOM Elements
const screenTitle = document.getElementById('screen-title');
const systemStatusIndicator = document.getElementById('system-status-indicator');
const systemStatusText = document.getElementById('system-status-text');
const modelSelect = document.getElementById('model-select');
const tempSlider = document.getElementById('temp-slider');
const tempVal = document.getElementById('temp-val');
const ragToggle = document.getElementById('rag-toggle');
const activeModelDisplay = document.getElementById('active-model-display');

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initChat();
    initTraining();
    initKnowledge();
    initPromptSettings();
    initChart();
    
    // Initial status check and data loading
    checkModelsStatus();
    loadDocuments();
    
    // Check status every 5 seconds
    setInterval(checkModelsStatus, 5000);
});

// 1. NAVIGATION CONTROL
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const screens = document.querySelectorAll('.app-screen');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetScreen = item.getAttribute('data-screen');
            
            // Update active nav class
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // Switch screen
            screens.forEach(s => s.classList.remove('active'));
            document.getElementById(`${targetScreen}-screen`).classList.add('active');
            
            currentScreen = targetScreen;
            
            // Update Header Title
            updateHeaderTitle();
            
            // Actions when entering screen
            if (targetScreen === 'training') {
                checkTrainingStatus();
            } else if (targetScreen === 'knowledge') {
                loadDocuments();
            }
        });
    });
}

function updateHeaderTitle() {
    switch (currentScreen) {
        case 'chat':
            const modelName = modelSelect.options[modelSelect.selectedIndex].text;
            screenTitle.innerText = `Czat z ${modelName}`;
            break;
        case 'training':
            screenTitle.innerText = 'Panel Treningowy Mey (PyTorch)';
            break;
        case 'knowledge':
            screenTitle.innerText = 'Baza Wiedzy (RAG)';
            break;
        case 'prompt':
            screenTitle.innerText = 'Osobowość Modelu (Prompt)';
            break;
    }
}

// Update model select change listener
modelSelect.addEventListener('change', () => {
    updateHeaderTitle();
    const model = modelSelect.value;
    activeModelDisplay.innerText = `Model: ${modelSelect.options[modelSelect.selectedIndex].text}`;
    
    // Set default prompt in textarea
    const promptInput = document.getElementById('system-prompt-input');
    if (promptInput) {
        promptInput.value = customSystemPrompt || defaultSystemPrompts[model] || '';
    }
});

// Temperature slider change listener
tempSlider.addEventListener('input', () => {
    tempVal.innerText = tempSlider.value;
});

// 2. MODEL STATUS CHECKS
async function checkModelsStatus() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) throw new Error();
        const data = await response.json();
        
        systemConnected = data.system_connected;
        
        // Update header indicator
        if (systemConnected) {
            systemStatusIndicator.className = 'status-indicator';
            systemStatusText.innerText = 'Aktywny';
        } else {
            systemStatusIndicator.className = 'status-indicator offline';
            systemStatusText.innerText = 'Rozłączony';
        }
        
        // Dynamic status update in model selection dropdown
        const customModel = data.models.find(m => m.id === 'mey_custom');
        const customOption = modelSelect.querySelector('option[value="mey_custom"]');
        if (customOption && customModel) {
            customOption.text = `Mey Custom (PyTorch) - ${customModel.status}`;
        }
        
    } catch (e) {
        systemStatusIndicator.className = 'status-indicator offline';
        systemStatusText.innerText = 'Brak połączenia z backendem';
    }
}

// 3. CHAT LOGIC
function initChat() {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-chat-btn');
    const voiceBtn = document.getElementById('voice-input-btn');
    
    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
    });
    
    // Send message on Enter, new line on Shift+Enter
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
    
    // Voice recognition (Speech to Text)
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.lang = 'pl-PL';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        let isListening = false;
        
        voiceBtn.addEventListener('click', () => {
            if (isListening) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
        
        recognition.onstart = () => {
            isListening = true;
            voiceBtn.classList.add('listening');
            voiceBtn.setAttribute('title', 'Słucham... Kliknij by wyłączyć');
        };
        
        recognition.onend = () => {
            isListening = false;
            voiceBtn.classList.remove('listening');
            voiceBtn.setAttribute('title', 'Dyktowanie (Mowa na tekst)');
        };
        
        recognition.onresult = (event) => {
            const speechResult = event.results[0][0].transcript;
            chatInput.value += (chatInput.value ? ' ' : '') + speechResult;
            // Trigger input event to resize textarea
            chatInput.dispatchEvent(new Event('input'));
        };
        
        recognition.onerror = (e) => {
            console.error('Speech recognition error:', e);
            voiceBtn.classList.remove('listening');
        };
    } else {
        voiceBtn.style.display = 'none'; // Speech recognition not supported
    }
}

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const chatHistory = document.getElementById('chat-history');
    const text = chatInput.value.trim();
    
    if (!text) return;
    
    // Hide welcome panel
    const welcome = document.getElementById('chat-welcome');
    if (welcome) welcome.style.display = 'none';
    
    // Append User Message to UI
    appendMessage('user', text);
    
    // Clear Input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Prepare message payload
    messageHistory.push({ role: 'user', content: text });
    
    // Append empty Assistant Message to UI (loading state)
    const assistantBubble = appendMessage('assistant', '');
    const contentElement = assistantBubble.querySelector('.message-content');
    contentElement.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Mey myśli...';
    
    // Query API
    try {
        const payload = {
            messages: messageHistory,
            model: modelSelect.value,
            temperature: parseFloat(tempSlider.value),
            use_rag: ragToggle.checked,
            system_prompt: customSystemPrompt || defaultSystemPrompts[modelSelect.value]
        };
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`Wystąpił błąd serwera (kod ${response.status})`);
        }
        
        contentElement.innerHTML = ''; // Clear loading spinner
        
        // Handle stream reader
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let completeText = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6).trim();
                    
                    if (dataStr === '[DONE]') {
                        break;
                    }
                    
                    try {
                        const parsed = JSON.parse(dataStr);
                        const contentChunk = parsed.choices[0].delta.content || '';
                        completeText += contentChunk;
                        
                        // Parse markdown on-the-fly
                        contentElement.innerHTML = marked.parse(completeText);
                        // Highlight syntax inside codes
                        Prism.highlightAllUnder(contentElement);
                        
                        // Auto scroll to bottom
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    } catch (e) {
                        // ignore malformed JSON chunks
                    }
                }
            }
        }
        
        // Save assistant answer in history
        messageHistory.push({ role: 'assistant', content: completeText });
        
        // Add speech actions
        addSpeechActions(assistantBubble, completeText);
        
    } catch (e) {
        contentElement.innerHTML = `<span style="color:var(--error);"><i class="fa-solid fa-triangle-exclamation"></i> Błąd: ${e.message}</span>`;
    }
}

function appendMessage(role, text) {
    const chatHistory = document.getElementById('chat-history');
    
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerText = role === 'user' ? 'E' : 'M';
    
    const contentWrapper = document.createElement('div');
    contentWrapper.style.display = 'flex';
    contentWrapper.style.flexDirection = 'column';
    contentWrapper.style.maxWidth = 'calc(100% - 50px)';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = text ? marked.parse(text) : '';
    
    contentWrapper.appendChild(content);
    bubble.appendChild(avatar);
    bubble.appendChild(contentWrapper);
    chatHistory.appendChild(bubble);
    
    if (text) {
        Prism.highlightAllUnder(content);
    }
    
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return bubble;
}

function addSpeechActions(bubble, text) {
    const wrapper = bubble.querySelector('div:last-child');
    
    const actions = document.createElement('div');
    actions.className = 'message-bubble-actions';
    
    const speakBtn = document.createElement('span');
    speakBtn.className = 'speech-btn';
    speakBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Czytaj na głos';
    speakBtn.addEventListener('click', () => {
        toggleSpeech(text, speakBtn, bubble);
    });
    
    actions.appendChild(speakBtn);
    wrapper.appendChild(actions);
}

function toggleSpeech(text, btnElement, bubbleElement) {
    if (synth.speaking) {
        synth.cancel();
        if (currentUtterance && currentUtterance.btn === btnElement) {
            btnElement.innerHTML = '<i class="fa-solid fa-volume-high"></i> Czytaj na głos';
            bubbleElement.classList.remove('speaking-active');
            currentUtterance = null;
            return;
        }
    }
    
    // Remove active styles from other elements
    document.querySelectorAll('.message-bubble').forEach(b => b.classList.remove('speaking-active'));
    document.querySelectorAll('.speech-btn').forEach(btn => btn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Czytaj na głos');
    
    // Clean text from markdown characters for speech
    const cleanText = text.replace(/[*_`#\-]/g, '');
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = 'pl-PL';
    
    utterance.onend = () => {
        btnElement.innerHTML = '<i class="fa-solid fa-volume-high"></i> Czytaj na głos';
        bubbleElement.classList.remove('speaking-active');
        currentUtterance = null;
    };
    
    utterance.onerror = () => {
        btnElement.innerHTML = '<i class="fa-solid fa-volume-high"></i> Czytaj na głos';
        bubbleElement.classList.remove('speaking-active');
        currentUtterance = null;
    };
    
    btnElement.innerHTML = '<i class="fa-solid fa-stop"></i> Zatrzymaj';
    bubbleElement.classList.add('speaking-active');
    
    currentUtterance = { utterance, btn: btnElement };
    synth.speak(utterance);
}

// 4. PYTORCH MODEL TRAINING LOGIC
function initTraining() {
    const startBtn = document.getElementById('start-train-btn');
    const stopBtn = document.getElementById('stop-train-btn');
    
    startBtn.addEventListener('click', startTrainingProcess);
    stopBtn.addEventListener('click', stopTrainingProcess);
}

function initChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    
    // Design matching gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.3)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
    
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointBackgroundColor: '#8b5cf6'
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#06b6d4',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3,
                    pointBackgroundColor: '#06b6d4'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 12 }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#64748b' },
                    title: {
                        display: true,
                        text: 'Wartość Straty (Loss)',
                        color: '#94a3b8'
                    }
                }
            }
        }
    });
}

async function startTrainingProcess() {
    const startBtn = document.getElementById('start-train-btn');
    const stopBtn = document.getElementById('stop-train-btn');
    const progressContainer = document.getElementById('training-progress-container');
    const logTerminal = document.getElementById('log-terminal');
    
    // Collect parameters
    const formData = new FormData();
    formData.append('iters', document.getElementById('train-iters').value);
    formData.append('batch_size', document.getElementById('train-batch').value);
    formData.append('block_size', document.getElementById('train-block').value);
    formData.append('embed_dim', document.getElementById('train-embed').value);
    formData.append('num_layers', document.getElementById('train-layers').value);
    formData.append('dataset', document.getElementById('train-dataset').value);
    
    startBtn.disabled = true;
    stopBtn.style.display = 'flex';
    progressContainer.style.display = 'block';
    
    logTerminal.innerHTML = '<div class="log-line info">Nawiązywanie połączenia z serwerem i inicjacja parametrów...</div>';
    
    // Clear chart
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.data.datasets[1].data = [];
    lossChart.update();
    
    try {
        const response = await fetch('/api/train/start', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || 'Błąd inicjacji treningu');
        
        appendLog('Inicjacja procesu pomyślna. Rozpoczynanie pętli treningowej.', 'success');
        
        // Start polling status
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        trainingPollInterval = setInterval(checkTrainingStatus, 1500);
        
    } catch (e) {
        appendLog(`Błąd krytyczny: ${e.message}`, 'error');
        startBtn.disabled = false;
        stopBtn.style.display = 'none';
    }
}

async function stopTrainingProcess() {
    try {
        const response = await fetch('/api/train/stop', { method: 'POST' });
        const data = await response.json();
        appendLog(data.message, 'info');
    } catch (e) {
        appendLog(`Błąd zatrzymywania: ${e.message}`, 'error');
    }
}

async function checkTrainingStatus() {
    try {
        const response = await fetch('/api/train/status');
        const data = await response.json();
        
        const startBtn = document.getElementById('start-train-btn');
        const stopBtn = document.getElementById('stop-train-btn');
        const progressContainer = document.getElementById('training-progress-container');
        const progressBarFill = document.getElementById('progress-bar-fill');
        const trainPercent = document.getElementById('train-percent');
        const trainLossVal = document.getElementById('train-loss-val');
        
        if (data.status === 'training' || data.status === 'starting') {
            startBtn.disabled = true;
            stopBtn.style.display = 'flex';
            progressContainer.style.display = 'block';
            
            // Update progress bar
            const percent = data.total_steps > 0 ? Math.round((data.step / data.total_steps) * 100) : 0;
            progressBarFill.style.width = `${percent}%`;
            trainPercent.innerText = `${percent}% (${data.step}/${data.total_steps})`;
            trainLossVal.innerText = `Loss: ${data.loss.toFixed(4)}`;
            
            // Append message
            if (data.message) {
                appendLog(data.message, 'info');
            }
            
            // Update chart if history available
            if (data.history && data.history.length > 0) {
                const steps = data.history.map(h => h.step);
                const trainLosses = data.history.map(h => h.train_loss);
                const valLosses = data.history.map(h => h.val_loss);
                
                lossChart.data.labels = steps;
                lossChart.data.datasets[0].data = trainLosses;
                lossChart.data.datasets[1].data = valLosses;
                lossChart.update();
            }
            
            // Set polling if not already running
            if (!trainingPollInterval) {
                trainingPollInterval = setInterval(checkTrainingStatus, 1500);
            }
        } else {
            // Idle, Completed, Stopped or Error
            startBtn.disabled = false;
            stopBtn.style.display = 'none';
            
            if (trainingPollInterval) {
                clearInterval(trainingPollInterval);
                trainingPollInterval = null;
            }
            
            if (data.status === 'completed') {
                progressBarFill.style.width = '100%';
                trainPercent.innerText = '100% (Ukończono)';
                trainLossVal.innerText = `Final Loss: ${data.loss.toFixed(4)}`;
                appendLog('Trening zakończony sukcesem! Model został zapisany na dysku.', 'success');
                appendLog('Możesz teraz wybrać model "Mey Custom (PyTorch)" w konfiguracji i przetestować jego generacje.', 'success');
                
                // Force check models status to update dropdown
                checkModelsStatus();
            } else if (data.status === 'stopped') {
                appendLog('Trening zatrzymany przez użytkownika.', 'error');
            } else if (data.status === 'error') {
                appendLog(`Błąd treningu: ${data.message}`, 'error');
            }
        }
    } catch (e) {
        console.error('Error fetching training status:', e);
    }
}

function appendLog(message, type = 'info') {
    const logTerminal = document.getElementById('log-terminal');
    
    // Avoid appending exact duplicates of logs to keep terminal clean
    const lastLine = logTerminal.lastElementChild;
    if (lastLine && lastLine.innerText === message) return;
    
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.innerText = `[${new Date().toLocaleTimeString()}] ${message}`;
    logTerminal.appendChild(line);
    
    // Auto-scroll terminal
    logTerminal.scrollTop = logTerminal.scrollHeight;
}

// 5. RAG KNOWLEDGE BASE
function initKnowledge() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('doc-file-input');
    
    uploadZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });
    
    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = 'var(--secondary)';
            uploadZone.style.backgroundColor = 'rgba(6, 182, 212, 0.05)';
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = 'rgba(139, 92, 246, 0.3)';
            uploadZone.style.backgroundColor = 'rgba(139, 92, 246, 0.02)';
        }, false);
    });
    
    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const uploadZone = document.getElementById('upload-zone');
    const originalHTML = uploadZone.innerHTML;
    
    uploadZone.innerHTML = '<i class="fa-solid fa-spinner fa-spin upload-icon"></i><h4>Analizowanie i wektoryzowanie...</h4><p>Generowanie osadzeń dla fragmentów tekstu.</p>';
    
    try {
        const response = await fetch('/api/knowledge/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Wystąpił błąd podczas wgrywania pliku.');
        
        loadDocuments(); // Refresh documents table
    } catch (e) {
        alert(e.message);
    } finally {
        uploadZone.innerHTML = originalHTML;
    }
}

async function loadDocuments() {
    try {
        const response = await fetch('/api/knowledge/documents');
        const data = await response.json();
        
        // Update stats
        document.getElementById('stats-doc-count').innerText = data.documents.length;
        document.getElementById('stats-chunk-count').innerText = data.total_chunks;
        document.getElementById('stats-search-type').innerText = 'Lokalny TF-IDF';
        
        const tableBody = document.getElementById('documents-table-body');
        
        if (data.documents.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted);">Brak dokumentów w bazie wiedzy. Wgraj plik powyżej.</td></tr>';
            return;
        }
        
        tableBody.innerHTML = '';
        data.documents.forEach(doc => {
            const sizeKB = (doc.size / 1024).toFixed(1);
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>
                    <div class="doc-name">
                        <i class="fa-solid fa-file-lines"></i>
                        <span>${doc.filename}</span>
                    </div>
                </td>
                <td>${sizeKB} KB</td>
                <td>${doc.chunks_count}</td>
                <td>
                    <button class="doc-action-btn" title="Usuń dokument" onclick="deleteDocument('${doc.filename}')">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    } catch (e) {
        console.error('Error loading documents:', e);
    }
}

// Attach deleteDocument to window so it is accessible from onclick
window.deleteDocument = async function(filename) {
    if (!confirm(`Czy na pewno chcesz usunąć plik "${filename}" z bazy wiedzy?`)) return;
    
    try {
        const response = await fetch(`/api/knowledge/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error('Błąd usuwania dokumentu');
        
        loadDocuments();
    } catch (e) {
        alert(e.message);
    }
};

// 6. SYSTEM PROMPT / CHARACTERISTIC EDIT
function initPromptSettings() {
    const saveBtn = document.getElementById('save-prompt-btn');
    const promptInput = document.getElementById('system-prompt-input');
    
    // Set initial prompt based on selected model
    const currentModel = modelSelect.value;
    promptInput.value = defaultSystemPrompts[currentModel] || '';
    
    saveBtn.addEventListener('click', () => {
        customSystemPrompt = promptInput.value.trim();
        alert('Prompt systemowy został pomyślnie zaktualizowany i zastosowany do przyszłych konwersacji!');
    });
}
