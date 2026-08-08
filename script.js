// ---- Config ----
const API_URL = "/ask";

// ---- Element references ----
const chat = document.getElementById("chat");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const micBtn = document.getElementById("mic");

// ---- Rotating placeholder examples ----
const placeholderExamples = [
  "e.g. Where is the CSS box model explained?",
  "e.g. Where is HTML taught?",
  "e.g. Where is React JS taught?",
  "e.g. Where is the TodoList app built?",
  "e.g. Where are JavaScript events covered?",
  "e.g. Where is flexbox explained?",
  "e.g. Where is grid layout taught?",
  "e.g. Where are React hooks covered?",
  "e.g. Where is state management taught?",
  "e.g. Where is API integration explained?",
  "e.g. Where is form validation covered?",
  "e.g. Where is routing explained in React?",
  "e.g. Where is SEO discussed?",
  "e.g. Where is responsive design taught?",
  "e.g. Where is Git and GitHub covered?",
  "e.g. Where is deployment explained?",
  "e.g. Where is conditional rendering taught?",
  "e.g. Where are React props explained?",
  "e.g. Where is Tailwind CSS covered?",
  "e.g. Where is DOM manipulation taught?"
];

let placeholderIndex = 1;
input.placeholder = placeholderExamples[0];

function rotatePlaceholder() {
  if (document.activeElement === input || input.value !== "") return;
  input.style.opacity = "0";
  setTimeout(() => {
    input.placeholder = placeholderExamples[placeholderIndex];
    placeholderIndex = (placeholderIndex + 1) % placeholderExamples.length;
    input.style.opacity = "1";
  }, 300);
}
setInterval(rotatePlaceholder, 4000);

// ---- Example chip click ----
function askExample(text) {
  input.value = text;
  sendQuestion();
}

// ---- Bold video numbers and timestamps in the answer text ----
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function highlightKeyInfo(text) {
  let safe = escapeHtml(text);

  // Bold timestamps like "6 min 28 sec" or "1 hr 4 min 12 sec"
  safe = safe.replace(
    /(\d+\s*hr\s*)?\d+\s*min\s*\d+\s*sec/gi,
    (match) => `<strong>${match}</strong>`
  );

  // Bold video/tutorial numbers like "Video 4", "Tutorial #114", "video number 71"
  safe = safe.replace(
    /(Video|Tutorial)\s*(number\s*)?#?\s*\d+/gi,
    (match) => `<strong>${match}</strong>`
  );

  // Turn raw YouTube URLs into clickable links with friendly text instead of the raw URL
  safe = safe.replace(
    /(https?:\/\/(?:www\.)?youtube\.com\/[^\s)]+)/gi,
    (url) => `<a href="${url}" target="_blank" rel="noopener" style="color: var(--accent); text-decoration: underline; font-weight: 600;">Click here to enroll Sigma Web Development</a>`
  );

  return safe;
}

function addMessage(role, text, autoSpeak = false) {
  const emptyState = document.getElementById("empty-state");
  if (emptyState) emptyState.remove();

  const msg = document.createElement("div");
  msg.className = "msg " + role;

  const avatar = document.createElement("div");
  avatar.className = "avatar " + role;
  avatar.textContent = role === "user" ? "You" : "AI";

  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "user") {
    bubble.textContent = text;
  }
  wrap.appendChild(bubble);

  msg.appendChild(avatar);
  msg.appendChild(wrap);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;

  if (role === "assistant") {
    if (autoSpeak && window.speechSynthesis) {
      playSyncedSpeech(text, bubble, wrap);
    } else {
      bubble.innerHTML = highlightKeyInfo(text);
      addSpeakButton(wrap, text, bubble);
    }
  }

  return bubble;
}

function addSpeakButton(wrap, text, bubble) {
  const speakBtn = document.createElement("button");
  speakBtn.className = "speak-btn";
  speakBtn.innerHTML = "&#128264; Listen";
  speakBtn.onclick = () => playSyncedSpeech(text, bubble, wrap, speakBtn);
  wrap.appendChild(speakBtn);
  return speakBtn;
}

function addThinking() {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  msg.id = "thinking-msg";

  const avatar = document.createElement("div");
  avatar.className = "avatar assistant";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

// ---- Track last assistant answer for local 'repeat' handling ----
let lastAssistantAnswer = null;

function isRepeatCommand(q) {
  const normalized = q.trim().toLowerCase().replace(/[?.!]/g, "");
  const repeatPhrases = [
    "repeat", "repeat it", "repeat that", "say that again", "say it again",
    "can you repeat", "can you repeat that", "again", "what did you say",
    "pardon", "come again"
  ];
  return repeatPhrases.includes(normalized);
}

// ---- Send question to backend ----
async function sendQuestion() {
  const q = input.value.trim();
  if (!q) return;

  addMessage("user", q);
  input.value = "";

  // Handle "repeat" locally - don't send it to the RAG backend as a new question
  if (isRepeatCommand(q)) {
    if (lastAssistantAnswer) {
      addMessage("assistant", lastAssistantAnswer, true);
    } else {
      addMessage("assistant", "I haven't said anything yet - ask me a question first!");
    }
    input.focus();
    return;
  }

  sendBtn.disabled = true;
  addThinking();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });

    const data = await res.json();
    document.getElementById("thinking-msg")?.remove();

    if (!res.ok) {
      addMessage("assistant", "Something went wrong: " + (data.detail || res.status));
    } else {
      lastAssistantAnswer = data.answer;
      addMessage("assistant", data.answer, true); // true = auto-speak with synced text reveal
    }
  } catch (err) {
    document.getElementById("thinking-msg")?.remove();
    addMessage("assistant", "Could not reach the API. Please check your connection and try again.");
  }

  sendBtn.disabled = false;
  input.focus();
}

// ---- Voice input (speech-to-text) ----
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
let isListening = false;

if (SpeechRecognition && (location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
  recognizer = new SpeechRecognition();
  recognizer.lang = "en-US";
  recognizer.interimResults = false;
  recognizer.maxAlternatives = 1;

  recognizer.onstart = () => {
    isListening = true;
    micBtn.classList.add("listening");
  };

  recognizer.onend = () => {
    isListening = false;
    micBtn.classList.remove("listening");
  };

  recognizer.onerror = () => {
    isListening = false;
    micBtn.classList.remove("listening");
  };

  recognizer.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    input.value = transcript;
    sendQuestion();
  };
} else {
  // Voice input needs a secure context (https or localhost).
  // Opening this file directly (file://) won't trigger the mic permission popup.
  micBtn.title = "Voice input needs the page served via http://localhost (not file://). See README.";
  micBtn.style.opacity = "0.35";
  micBtn.style.cursor = "not-allowed";
}

function toggleListening() {
  if (!recognizer) return;
  if (isListening) {
    recognizer.stop();
  } else {
    try {
      recognizer.start();
    } catch (err) {
      // Recognizer was stuck in a bad state (e.g. already running) - reset and retry
      recognizer.stop();
      setTimeout(() => {
        try { recognizer.start(); } catch (e) { /* ignore */ }
      }, 200);
    }
  }
}

// ---- Voice output synced with text reveal (subtitle-style) ----
if (window.speechSynthesis) {
  window.speechSynthesis.getVoices();
  window.speechSynthesis.onvoiceschanged = () => {};
}

function playSyncedSpeech(text, bubble, wrap, existingBtn) {
  if (!window.speechSynthesis) {
    bubble.innerHTML = highlightKeyInfo(text);
    return;
  }

  const btn = existingBtn || addSpeakButton(wrap, text, bubble);
  const wasSpeaking = btn.classList.contains("speaking");

  window.speechSynthesis.cancel();
  document.querySelectorAll(".speak-btn.speaking").forEach(b => {
    b.classList.remove("speaking");
    b.innerHTML = "&#128264; Listen";
  });

  // If this button was already speaking, treat click as "stop"
  if (wasSpeaking) {
    bubble.innerHTML = highlightKeyInfo(text);
    return;
  }

  // Remove URLs from what gets spoken - TTS reads links out letter by letter,
  // which sounds terrible. The visual bubble still shows the full text/links.
  // Also expand "No:" to "Number:" so TTS doesn't say "no" (negation) for the video label.
  const speakableText = text
    .replace(/https?:\/\/\S+/gi, "")
    .replace(/\bNo:/g, "Number:")
    .replace(/\n{2,}/g, "\n")
    .trim();

  const start = () => {
    const utterance = new SpeechSynthesisUtterance(speakableText);
    utterance.lang = "en-US";
    utterance.rate = 0.95;
    utterance.pitch = 1.0;

    const voices = window.speechSynthesis.getVoices();
    // Prefer higher-quality voices (Google/Microsoft Natural) over robotic defaults
    const preferredVoice =
      voices.find(v => v.name.includes("Google US English")) ||
      voices.find(v => v.name.includes("Natural") && v.lang.startsWith("en")) ||
      voices.find(v => v.name.includes("Google") && v.lang.startsWith("en")) ||
      voices.find(v => v.lang.startsWith("en"));
    if (preferredVoice) utterance.voice = preferredVoice;

    btn.classList.add("speaking");
    btn.innerHTML = "&#9209; Stop";
    bubble.textContent = ""; // start empty, reveal as it speaks

    let boundaryFired = false;

    utterance.onboundary = (event) => {
      boundaryFired = true;
      const revealedSoFar = speakableText.substring(0, event.charIndex + (event.charLength || 0));
      bubble.innerHTML = highlightKeyInfo(revealedSoFar) +
        '<span style="opacity:0.35">' + escapeHtml(speakableText.substring(revealedSoFar.length)) + '</span>';
      chat.scrollTop = chat.scrollHeight;
    };

    // Fallback: some Chrome voices never fire onboundary events. If nothing
    // happened shortly after speech starts, just show the full text instead
    // of leaving the bubble blank.
    setTimeout(() => {
      if (!boundaryFired) {
        bubble.innerHTML = highlightKeyInfo(text);
      }
    }, 600);

    utterance.onend = () => {
      btn.classList.remove("speaking");
      btn.innerHTML = "&#128264; Listen";
      bubble.innerHTML = highlightKeyInfo(text);
    };

    utterance.onerror = () => {
      btn.classList.remove("speaking");
      btn.innerHTML = "&#128264; Listen";
      bubble.innerHTML = highlightKeyInfo(text);
    };

    window.speechSynthesis.speak(utterance);
  };

  if (window.speechSynthesis.getVoices().length === 0) {
    setTimeout(start, 150);
  } else {
    start();
  }
}