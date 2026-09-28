function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatModel(model) {
  return model.split("/").at(-1)?.replaceAll("-", " ") ?? model;
}

function traceItem(label, value) {
  const item = element("div", "trace-item");
  item.append(element("dt", null, label), element("dd", null, value));
  return item;
}

function renderTrace(trace) {
  const panel = element("aside", "trace-panel");
  panel.setAttribute("aria-labelledby", "trace-heading");
  panel.append(
    element("p", "section-number", "03 / Execution trace"),
    element("h2", null, "How this was decided"),
  );

  const metrics = element("dl", "trace-grid");
  metrics.append(
    traceItem("Retrieved", `${trace.retrieved_chunks} passages`),
    traceItem("Latency", `${(trace.latency_ms / 1000).toFixed(2)} s`),
    traceItem("Tokens", trace.total_tokens.toLocaleString()),
    traceItem("Attempts", String(trace.generation_attempts)),
    traceItem("Embedding", formatModel(trace.embedding_model)),
    traceItem("Generator", formatModel(trace.generation_model)),
  );
  panel.append(metrics);
  return panel;
}

function renderCitations(citations, evidenceIds) {
  const section = element("div", "citations");
  section.append(element("h3", null, "Cited evidence"));
  const list = element("div", "citation-list");

  citations.forEach((citation, index) => {
    const button = element("button", "citation-chip");
    button.type = "button";
    const references = citation.rule_references.join(", ");
    button.textContent = `[${index + 1}] ${citation.document_title} · ${references}`;
    button.addEventListener("click", () => {
      const target = document.getElementById(evidenceIds.get(citation.chunk_id));
      target?.scrollIntoView({behavior: "smooth", block: "center"});
      target?.querySelector("summary")?.focus();
    });
    list.append(button);
  });

  section.append(list);
  return section;
}

function renderEvidence(matches, citedChunkIds) {
  const section = element("section", "evidence-section");
  section.setAttribute("aria-labelledby", "evidence-heading");

  const heading = element("div", "evidence-heading");
  const titleGroup = element("div");
  titleGroup.append(
    element("p", "section-number", "04 / Retrieved evidence"),
    element("h2", null, "Passages considered"),
  );
  heading.append(titleGroup, element("p", "evidence-count", `${matches.length} ranked passages`));
  section.append(heading);

  const list = element("div", "evidence-list");
  matches.forEach((match, index) => {
    const details = element("details", "evidence-card");
    details.id = `evidence-${index + 1}`;
    if (index === 0) details.open = true;

    const summary = element("summary");
    const rank = element("span", "evidence-rank", String(index + 1).padStart(2, "0"));
    const label = element("span", "evidence-label");
    label.append(
      element("strong", null, match.document_title),
      element("span", null, match.heading_path.join(" / ")),
    );
    const score = element("span", "evidence-score", `${match.score.toFixed(3)} cosine`);
    summary.append(rank, label, score);

    const body = element("div", "evidence-body");
    const tags = element("div", "rule-tags");
    match.rule_references.forEach((reference) => tags.append(element("span", null, `Rule ${reference}`)));
    if (citedChunkIds.has(match.chunk_id)) tags.append(element("span", "cited-tag", "Cited"));

    const source = element("a", "source-link", "Open official rulebook ↗");
    source.href = match.source_url;
    source.target = "_blank";
    source.rel = "noreferrer";
    body.append(tags, element("p", "evidence-text", match.text), source);
    details.append(summary, body);
    list.append(details);
  });

  section.append(list);
  return section;
}

function answerPanel(result, evidenceIds) {
  const panel = element("article", `answer-panel ${result.abstained ? "is-abstention" : "is-supported"}`);
  const headingRow = element("div", "answer-heading");
  headingRow.append(
    element("p", "section-number", "02 / Ruling"),
    element("span", "result-status", result.abstained ? "Abstained" : "Evidence verified"),
  );
  panel.append(headingRow);

  if (result.abstained) {
    panel.append(
      element("p", "answer-kicker", "Insufficient evidence"),
      element("h2", null, "No supported ruling."),
      element("p", "answer-explanation", result.abstention_reason),
    );
  } else {
    panel.append(
      element("p", "answer-kicker", `2026 NFL rules · ${result.citations.length} citation${result.citations.length === 1 ? "" : "s"}`),
      element("h2", null, result.ruling),
      element("p", "answer-explanation", result.explanation),
      renderCitations(result.citations, evidenceIds),
    );
  }
  return panel;
}

export function renderAnswer(region, result) {
  const evidenceIds = new Map(result.evidence.map((match, index) => [match.chunk_id, `evidence-${index + 1}`]));
  const citedChunkIds = new Set(result.citations.map((citation) => citation.chunk_id));
  const layout = element("div", "result-layout");
  layout.append(answerPanel(result, evidenceIds), renderTrace(result.trace));

  region.removeAttribute("aria-busy");
  region.replaceChildren(layout, renderEvidence(result.evidence, citedChunkIds));
  region.querySelector(".answer-panel h2")?.setAttribute("tabindex", "-1");
  region.querySelector(".answer-panel h2")?.focus({preventScroll: true});
}

export function renderLoading(region) {
  const loading = element("div", "loading-state");
  loading.append(
    element("p", "section-number", "02 / Reviewing"),
    element("div", "skeleton skeleton-short"),
    element("div", "skeleton skeleton-title"),
    element("div", "skeleton"),
    element("div", "skeleton skeleton-long"),
  );
  region.setAttribute("aria-busy", "true");
  region.replaceChildren(loading);
}

export function renderError(region, error, retry) {
  const panel = element("div", "error-state");
  const title = error.status === 502 ? "The model provider is busy." : "The ruling could not be completed.";
  panel.append(
    element("p", "section-number", "02 / Request interrupted"),
    element("p", "error-code", error.status ? `HTTP ${error.status}` : "Connection error"),
    element("h2", null, title),
    element("p", null, error.message),
  );
  const button = element("button", "secondary-action", "Try this question again");
  button.type = "button";
  button.addEventListener("click", retry);
  panel.append(button);
  region.removeAttribute("aria-busy");
  region.replaceChildren(panel);
  panel.querySelector("h2").setAttribute("tabindex", "-1");
  panel.querySelector("h2").focus({preventScroll: true});
}
