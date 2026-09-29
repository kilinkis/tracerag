import {fetchAnswer, fetchCorpusStatus, RequestError} from "/assets/api.js";
import {renderAnswer, renderError, renderLoading} from "/assets/render.js";

const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-question");
const characterCount = document.querySelector("#character-count");
const resultRegion = document.querySelector("#result-region");
const announcement = document.querySelector("#announcement");

let lastQuestion = "";

function announce(message) {
  announcement.textContent = "";
  window.requestAnimationFrame(() => {
    announcement.textContent = message;
  });
}

function setSubmitting(isSubmitting) {
  questionInput.disabled = isSubmitting;
  submitButton.disabled = isSubmitting;
  submitButton.querySelector("span").textContent = isSubmitting ? "Reviewing evidence" : "Send to review";
}

function updateCharacterCount() {
  characterCount.textContent = String(questionInput.value.length);
}

async function submitQuestion(question) {
  lastQuestion = question;
  setSubmitting(true);
  renderLoading(resultRegion);
  announce("Reviewing retrieved evidence.");
  resultRegion.scrollIntoView({behavior: "smooth", block: "start"});

  try {
    const result = await fetchAnswer(question);
    renderAnswer(resultRegion, result);
    announce(result.abstained ? "The system abstained from ruling." : "Grounded ruling ready.");
  } catch (error) {
    const requestError = error instanceof RequestError ? error : new RequestError("TraceRAG is temporarily unavailable.");
    renderError(resultRegion, requestError, () => submitQuestion(lastQuestion));
    announce("The ruling could not be completed.");
  } finally {
    setSubmitting(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }
  submitQuestion(question);
});

questionInput.addEventListener("input", updateCharacterCount);

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    updateCharacterCount();
    questionInput.focus();
  });
});

fetchCorpusStatus()
  .then((status) => {
    document.querySelector("#stat-season").textContent = String(status.season);
    document.querySelector("#stat-documents").textContent = String(status.documents);
    document.querySelector("#stat-chunks").textContent = String(status.chunks);
    document.querySelector("#stat-rules").textContent = String(status.rule_references);
  })
  .catch(() => {
    document.querySelector(".corpus-stats").setAttribute("aria-label", "Corpus status unavailable");
  });

updateCharacterCount();
