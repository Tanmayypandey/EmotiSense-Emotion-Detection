const inputText = document.getElementById("inputText");
const charCount = document.getElementById("charCount");
const predictBtn = document.getElementById("predictBtn");
const errorMessage = document.getElementById("errorMessage");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const loadingState = document.getElementById("loadingState");
const resultSubtitle = document.getElementById("resultSubtitle");

const EMOTIONS = {
  joy: { label: "Joy", emoji: "😄", color: "#ffc65b" },
  sadness: { label: "Sadness", emoji: "😢", color: "#50c9ff" },
  anger: { label: "Anger", emoji: "😠", color: "#ff687c" },
  fear: { label: "Fear", emoji: "😨", color: "#c37aff" },
  love: { label: "Love", emoji: "❤️", color: "#ff7bd4" },
  surprise: { label: "Surprise", emoji: "😲", color: "#a98aff" }
};

// Character counter
inputText.addEventListener("input", () => {
  charCount.textContent = inputText.value.length;
  hideError();
});

// Quick example buttons
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    inputText.value = button.dataset.example;
    charCount.textContent = inputText.value.length;
    hideError();
    inputText.focus();
  });
});

// Light / dark theme toggle
document.getElementById("themeToggle").addEventListener("click", () => {
  document.body.classList.toggle("light");

  const lightMode = document.body.classList.contains("light");
  const toggle = document.getElementById("themeToggle");

  toggle.textContent = lightMode ? "☀" : "☾";
  toggle.setAttribute(
    "aria-label",
    lightMode ? "Toggle dark mode" : "Toggle light mode"
  );
});

// Detect button
predictBtn.addEventListener("click", predictEmotion);

// Ctrl + Enter also starts prediction
inputText.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    predictEmotion();
  }
});

function hideError() {
  errorMessage.textContent = "";
  errorMessage.classList.remove("show");
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.add("show");
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;

  predictBtn.innerHTML = isLoading
    ? '<span class="sparkle">✧</span> Analyzing…'
    : '<span class="sparkle">✧</span> Detect Emotion <span class="arrow">→</span>';

  loadingState.classList.toggle("show", isLoading);

  if (isLoading) {
    resultPlaceholder.style.display = "none";
    resultContent.classList.remove("show");
  }
}

function normalizeEmotion(value) {
  return String(value || "").toLowerCase().trim();
}

function formatPercent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "0%";
  }

  return `${(number * 100).toFixed(1)}%`;
}

// Main prediction function
async function predictEmotion() {
  const text = inputText.value.trim();

  hideError();

  if (!text) {
    showError("Please enter a sentence before detecting its emotion.");
    inputText.focus();
    return;
  }

  if (text.length > 2000) {
    showError("Please keep your text within 2,000 characters.");
    return;
  }

  setLoading(true);
  resultSubtitle.textContent = "Analyzing your text…";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    let data;

    try {
      data = await response.json();
    } catch {
      throw new Error("The API returned an unreadable response.");
    }

    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : "The prediction request failed.";

      throw new Error(detail);
    }

    renderResult(data, text);

  } catch (error) {
    resultPlaceholder.style.display = "flex";
    resultContent.classList.remove("show");

    resultSubtitle.textContent = "Prediction could not be completed";

    showError(
      `${error.message || "Could not connect to the API."} ` +
      "Make sure FastAPI is running at http://127.0.0.1:8000."
    );

  } finally {
    setLoading(false);
  }
}

// Display the model response
function renderResult(data, originalText) {
  const emotionKey = normalizeEmotion(data.predicted_emotion);

  const emotion = EMOTIONS[emotionKey] || {
    label: emotionKey || "Unknown",
    emoji: "✨",
    color: "#38cfff"
  };

  document.getElementById("emotionEmoji").textContent = emotion.emoji;
  document.getElementById("emotionName").textContent = emotion.label;

  document.getElementById("confidenceValue").textContent =
    formatPercent(data.confidence);

  document.getElementById("analyzedText").textContent = originalText;

  resultSubtitle.textContent = "Your model's prediction is ready";
  resultPlaceholder.style.display = "none";

  const probabilities = data.all_probabilities || {};

  const rows = Object.entries(probabilities)
    .map(([key, value]) => ({
      key: normalizeEmotion(key),
      value: Number(value)
    }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => b.value - a.value);

  const list = document.getElementById("probabilityList");
  list.replaceChildren();

  rows.forEach(({ key, value }) => {
    const itemEmotion = EMOTIONS[key] || {
      label: key,
      color: "#38cfff"
    };

    const row = document.createElement("div");
    row.className = "probability-row";

    const name = document.createElement("span");
    name.className = "prob-name";
    name.textContent = itemEmotion.label;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";

    fill.style.background =
      `linear-gradient(90deg, ${itemEmotion.color}, ${itemEmotion.color}bb)`;

    fill.style.width = "0%";

    track.appendChild(fill);

    const percent = document.createElement("span");
    percent.className = "prob-value";
    percent.textContent = formatPercent(value);

    row.append(name, track, percent);
    list.appendChild(row);

    requestAnimationFrame(() => {
      fill.style.width =
        `${Math.max(0, Math.min(100, value * 100))}%`;
    });
  });

  resultContent.classList.add("show");
}

// Optional API health check
fetch("/health").catch(() => {
  console.warn("FastAPI health endpoint is currently unavailable.");
});