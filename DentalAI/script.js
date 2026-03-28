/* ========================
   DENTAL AI — JAVASCRIPT
   ======================== */

const CALENDLY_LINK = 'https://calendly.com/bhattializa58';

let currentLang = 'ur';
let isRecording = false;
let recognition = null;

// Chat history for Groq
let chatHistory = [];

// ========================
// LANGUAGE TOGGLE
// ========================
function setLang(lang) {
  currentLang = lang;
  document.getElementById('btnUR').classList.toggle('active', lang === 'ur');
  document.getElementById('btnEN').classList.toggle('active', lang === 'en');
  document.getElementById('chatInput').placeholder =
    lang === 'ur' ? 'Apni takleef likhein...' : 'Describe your symptoms...';
}

// ========================
// MESSAGE FUNCTIONS
// ========================
function addMessage(text, isUser) {
  const msgs = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg ' + (isUser ? 'msg-user' : 'msg-bot');
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  div.innerHTML = text.replace(/\n/g, '<br>') + '<div class="msg-time">' + now + '</div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function showTyping() {
  const msgs = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg msg-bot';
  div.id = 'typingIndicator';
  div.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typingIndicator');
  if (t) t.remove();
}

// ========================
// GROQ CHAT
// ========================
async function sendToGroq(userText) {
  // Add to history
  chatHistory.push({ role: 'user', content: userText });

  try {
    const response = await fetch('http://127.0.0.1:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: chatHistory })
    });

    const data = await response.json();

    if (data.error) {
      removeTyping();
      addMessage('❌ Error: ' + data.error, false);
      return;
    }

    const reply = data.reply;

    // Add assistant reply to history
    chatHistory.push({ role: 'assistant', content: reply });

    removeTyping();

    // Check if reply mentions booking
    const hasBooking = reply.toLowerCase().includes('appointment') ||
      reply.toLowerCase().includes('book') ||
      reply.toLowerCase().includes('doctor se milein') ||
      reply.toLowerCase().includes('doctor se milna');

    const bookBtn = hasBooking
      ? '<br><br><button onclick="openCalendly()" style="background:#1565C0;color:white;border:none;padding:7px 16px;border-radius:7px;cursor:pointer;font-size:12px;font-weight:600;">📅 ' +
        (currentLang === 'ur' ? 'Appointment Book Karein' : 'Book Appointment') + '</button>'
      : '';

    addMessage(reply + bookBtn, false);

  } catch (err) {
    removeTyping();
    addMessage('❌ Server se connect nahi ho saka. Backend chalu hai?', false);
  }
}

// ========================
// SEND MESSAGE
// ========================
function sendMessage() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, true);
  input.value = '';
  showTyping();
  sendToGroq(text);
}

function handleKey(e) {
  if (e.key === 'Enter') sendMessage();
}

// ========================
// VOICE INPUT
// ========================
function toggleMic() {
  if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
    addMessage('Voice input Chrome browser mein kaam karta hai.', false);
    return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!isRecording) {
    recognition = new SR();
    recognition.lang = currentLang === 'ur' ? 'ur-PK' : 'en-US';
    recognition.onresult = (e) => {
      document.getElementById('chatInput').value = e.results[0][0].transcript;
    };
    recognition.onend = () => {
      isRecording = false;
      document.getElementById('micBtn').classList.remove('recording');
    };
    recognition.start();
    isRecording = true;
    document.getElementById('micBtn').classList.add('recording');
  } else {
    recognition.stop();
  }
}

// ========================
// X-RAY UPLOAD & PREDICT
// ========================
async function uploadXray(file) {
  const formData = new FormData();
  formData.append('file', file);

  addMessage('⏳ X-Ray analyze ho rahi hai...', false);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: 'POST',
      body: formData
    });
    const result = await response.json();

    if (result.error) {
      addMessage('❌ Error: ' + result.error, false);
      return;
    }

    if (result.detected.length > 0) {
      const detectedMsg = currentLang === 'ur'
        ? '🔍 X-Ray mein mila: ' + result.detected.join(', ')
        : '🔍 Detected: ' + result.detected.join(', ');
      addMessage(detectedMsg, false);

      // Also send to Groq for advice
      const xrayPrompt = currentLang === 'ur'
        ? `Meri X-ray mein yeh conditions detect hui hain: ${result.detected.join(', ')}. Kya advice denge?`
        : `My X-ray detected these conditions: ${result.detected.join(', ')}. What would you advise?`;

      showTyping();
      sendToGroq(xrayPrompt);
    } else {
      addMessage('✅ X-Ray mein koi issue nahi mila.', false);
    }

    // Add overlay image to chat history
    const msgs = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'msg msg-bot';
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + result.overlay;
    img.style = 'width:100%;border-radius:8px;margin-top:4px;';
    div.appendChild(img);
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;

  } catch (err) {
    addMessage('❌ Server se connect nahi ho saka. Backend chalu hai?', false);
  }
}

// ========================
// CALENDLY BOOKING
// ========================
function openCalendly() {
  window.open(CALENDLY_LINK, '_blank');
}

// ========================
// AUTO GREETING ON LOAD
// ========================
window.onload = function () {
  setTimeout(() => {
    showTyping();
    setTimeout(() => {
      removeTyping();
      const greeting = currentLang === 'ur'
        ? 'Assalam o Alaikum! 👋 Main aapka Dental AI Assistant hun. Apni takleef batayein ya "Salam" likh kar shuru karein!'
        : 'Hello! 👋 I am your Dental AI Assistant. Type "Hello" to get started!';
      addMessage(greeting, false);
    }, 800);
  }, 300);
};
