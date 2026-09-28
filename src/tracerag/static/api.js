const REQUEST_TIMEOUT_MS = 35_000;

async function requestJson(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = payload?.detail;
      const message = typeof detail === "string" ? detail : "The request could not be completed.";
      throw new RequestError(message, response.status);
    }

    return payload;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new RequestError("The request took too long. Please try again.", 408);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export class RequestError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "RequestError";
    this.status = status;
  }
}

export function fetchCorpusStatus() {
  return requestJson("/corpus/status");
}

export function fetchAnswer(question) {
  return requestJson("/answers", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question, top_k: 3}),
  });
}
