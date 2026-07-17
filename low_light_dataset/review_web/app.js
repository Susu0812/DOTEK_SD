(() => {
  "use strict";

  const CANVAS_WIDTH = 640;
  const CANVAS_HEIGHT = 480;
  const SELECTION_RADIUS = 6;
  const AUTOSAVE_DELAY_MS = 500;
  const HISTORY_LIMIT = 50;
  const SESSION_REASON_PREFIX = "hose-review:exclusion-reason:";
  const CLASS_NAMES = {
    positive: "正样本",
    hard_negative: "困难负样本",
    excluded: "排除",
    needs_second_review: "需二次复核",
  };

  const byId = (id) => document.getElementById(id);
  const canvas = byId("hose-canvas");
  const context = canvas.getContext("2d");
  const backgroundImage = new Image();
  const state = {
    records: [],
    queue: [],
    detail: null,
    anchors: [],
    suggestionAnchors: [],
    selectedIndex: null,
    dragging: false,
    dragSnapshot: null,
    undoStack: [],
    redoStack: [],
  };
  let autosaveTimer = null;
  let pendingAutosave = Promise.resolve(true);
  let saveBlocked = false;
  let editGeneration = 0;
  let navigationToken = 0;
  const reasonDraftFallback = new Map();

  function markLocalEdit() {
    editGeneration += 1;
  }

  function reasonDraftKey(stem) {
    return `${SESSION_REASON_PREFIX}${stem}`;
  }

  function loadReasonDraft(stem) {
    if (!stem) return null;
    if (reasonDraftFallback.has(stem)) return reasonDraftFallback.get(stem);
    try {
      return window.sessionStorage.getItem(reasonDraftKey(stem));
    } catch (error) {
      return null;
    }
  }

  function storeReasonDraft(stem, reason) {
    if (!stem) return;
    reasonDraftFallback.set(stem, reason);
    try {
      window.sessionStorage.setItem(reasonDraftKey(stem), reason);
    } catch (error) {
      // The in-memory fallback still protects the current browser session.
    }
  }

  function clearReasonDraft(stem) {
    if (!stem) return;
    reasonDraftFallback.delete(stem);
    try {
      window.sessionStorage.removeItem(reasonDraftKey(stem));
    } catch (error) {
      // Storage can be disabled; the fallback has already been cleared.
    }
  }

  function cloneAnchors(anchors) {
    return anchors.map((anchor) => ({ ...anchor }));
  }

  function deduplicateAnchors(anchors) {
    const byY = new Map();
    anchors.forEach((anchor) => {
      const rawX = Number(anchor.x);
      const rawY = Number(anchor.y);
      if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) return;
      const y = Math.max(0, Math.min(479, Math.round(rawY)));
      const x = Math.max(0, Math.min(639, Number(rawX.toFixed(2))));
      byY.set(y, { y, x, confidence: 1, source: "human" });
    });
    return Array.from(byY.values()).sort((a, b) => a.y - b.y);
  }

  function pointerCoordinates(event) {
    const rect = canvas.getBoundingClientRect();
    const scaledX = (event.clientX - rect.left) * CANVAS_WIDTH / rect.width;
    const scaledY = (event.clientY - rect.top) * CANVAS_HEIGHT / rect.height;
    return {
      x: Math.max(0, Math.min(639, Number(scaledX.toFixed(2)))),
      y: Math.max(0, Math.min(479, Math.round(scaledY))),
    };
  }

  function nearestAnchorIndex(event) {
    const rect = canvas.getBoundingClientRect();
    const point = pointerCoordinates(event);
    let nearest = null;
    let nearestDistance = SELECTION_RADIUS + 1;
    state.anchors.forEach((anchor, index) => {
      const dx = (anchor.x - point.x) * rect.width / CANVAS_WIDTH;
      const dy = (anchor.y - point.y) * rect.height / CANVAS_HEIGHT;
      const distance = Math.hypot(dx, dy);
      if (distance <= SELECTION_RADIUS && distance < nearestDistance) {
        nearest = index;
        nearestDistance = distance;
      }
    });
    return nearest;
  }

  function drawCenterline(anchors, color, width, pointRadius) {
    const sorted = cloneAnchors(anchors).sort((a, b) => a.y - b.y);
    if (sorted.length) {
      context.beginPath();
      context.moveTo(sorted[0].x, sorted[0].y);
      sorted.slice(1).forEach((anchor) => context.lineTo(anchor.x, anchor.y));
      context.strokeStyle = color;
      context.lineWidth = width;
      context.lineJoin = "round";
      context.stroke();
    }
    sorted.forEach((anchor) => {
      context.beginPath();
      context.arc(anchor.x, anchor.y, pointRadius, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      context.strokeStyle = "#111827";
      context.lineWidth = 1;
      context.stroke();
    });
  }

  function renderCanvas() {
    context.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    if (backgroundImage.complete && backgroundImage.naturalWidth) {
      context.drawImage(backgroundImage, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    } else {
      context.fillStyle = "#111827";
      context.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    }
    drawCenterline(state.suggestionAnchors, "#f59e0b", 3, 4);
    drawCenterline(state.anchors, "#ef4444", 3, 5);
    if (state.selectedIndex !== null && state.anchors[state.selectedIndex]) {
      const selected = state.anchors[state.selectedIndex];
      context.beginPath();
      context.arc(selected.x, selected.y, 9, 0, Math.PI * 2);
      context.strokeStyle = "#ffffff";
      context.lineWidth = 2;
      context.stroke();
    }
  }

  function mediaUrl(stem, enhanced = false) {
    if (!stem) return "";
    const base = enhanced ? "/media/enhanced/" : "/media/original/";
    return `${base}${encodeURIComponent(stem)}.jpg`;
  }

  function loadBackground() {
    if (!state.detail) return;
    backgroundImage.src = mediaUrl(state.detail.stem, byId("preview-toggle").checked);
  }

  backgroundImage.addEventListener("load", renderCanvas);
  backgroundImage.addEventListener("error", () => {
    renderCanvas();
    showSaveResult("图像加载失败，请检查候选帧。", true);
  });

  function pushHistory(stack, anchors) {
    stack.push(cloneAnchors(anchors));
    if (stack.length > HISTORY_LIMIT) stack.shift();
  }

  function commitAnchors(nextAnchors) {
    pushHistory(state.undoStack, state.anchors);
    state.redoStack.length = 0;
    state.anchors = deduplicateAnchors(nextAnchors);
    markLocalEdit();
    state.selectedIndex = null;
    renderCanvas();
    updateHistoryButtons();
    scheduleAutosave();
  }

  function undo() {
    if (!state.undoStack.length) return;
    pushHistory(state.redoStack, state.anchors);
    state.anchors = state.undoStack.pop();
    markLocalEdit();
    state.selectedIndex = null;
    renderCanvas();
    updateHistoryButtons();
    scheduleAutosave();
  }

  function redo() {
    if (!state.redoStack.length) return;
    pushHistory(state.undoStack, state.anchors);
    state.anchors = state.redoStack.pop();
    markLocalEdit();
    state.selectedIndex = null;
    renderCanvas();
    updateHistoryButtons();
    scheduleAutosave();
  }

  function updateHistoryButtons() {
    byId("undo").disabled = state.undoStack.length === 0;
    byId("redo").disabled = state.redoStack.length === 0;
  }

  canvas.addEventListener("pointerdown", (event) => {
    canvas.focus();
    const nearest = nearestAnchorIndex(event);
    if (nearest !== null) {
      state.selectedIndex = nearest;
      state.dragging = true;
      state.dragSnapshot = cloneAnchors(state.anchors);
      canvas.setPointerCapture(event.pointerId);
      renderCanvas();
      return;
    }
    commitAnchors([...state.anchors, pointerCoordinates(event)]);
    const point = pointerCoordinates(event);
    state.selectedIndex = state.anchors.findIndex((anchor) => anchor.y === point.y);
    renderCanvas();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging || state.selectedIndex === null) return;
    const point = pointerCoordinates(event);
    const moving = cloneAnchors(state.anchors);
    moving[state.selectedIndex] = { ...moving[state.selectedIndex], ...point };
    const nextAnchors = deduplicateAnchors(moving);
    const geometryChanged = JSON.stringify(nextAnchors) !== JSON.stringify(state.anchors);
    if (!geometryChanged) return;
    state.anchors = nextAnchors;
    markLocalEdit();
    state.selectedIndex = state.anchors.findIndex((anchor) => anchor.y === point.y);
    renderCanvas();
  });

  function finishDrag(event) {
    if (!state.dragging) return;
    state.dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    if (JSON.stringify(state.dragSnapshot) !== JSON.stringify(state.anchors)) {
      pushHistory(state.undoStack, state.dragSnapshot || []);
      state.redoStack.length = 0;
      updateHistoryButtons();
      scheduleAutosave();
    }
    state.dragSnapshot = null;
  }

  canvas.addEventListener("pointerup", finishDrag);
  canvas.addEventListener("pointercancel", finishDrag);

  function isTypingTarget(target) {
    return target instanceof HTMLInputElement
      || target instanceof HTMLTextAreaElement
      || target instanceof HTMLSelectElement
      || target.isContentEditable;
  }

  function editorHasFocus() {
    return document.activeElement === canvas || byId("canvas-editor").contains(document.activeElement);
  }

  document.addEventListener("keydown", (event) => {
    if (isTypingTarget(event.target) || !editorHasFocus()) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      undo();
    } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault();
      redo();
    } else if ((event.key === "Delete" || event.key === "Backspace") && state.selectedIndex !== null) {
      event.preventDefault();
      commitAnchors(state.anchors.filter((_, index) => index !== state.selectedIndex));
    }
  });

  byId("undo").addEventListener("click", undo);
  byId("redo").addEventListener("click", redo);
  byId("clear-anchors").addEventListener("click", () => {
    if (!state.anchors.length) return;
    if (window.confirm("确认清空当前记录的全部人工锚点？")) commitAnchors([]);
  });

  function selectedTags() {
    return Array.from(byId("interference-tags").querySelectorAll("input:checked"), (input) => input.value);
  }

  function warningValues(raw = byId("warnings").value) {
    return raw.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
  }

  function captureDraft() {
    return {
      anchors: cloneAnchors(state.anchors),
      notes: byId("notes").value,
      warnings: byId("warnings").value,
      exclusionReason: byId("exclusion-reason").value,
      tags: selectedTags(),
    };
  }

  function restoreDraft(draft) {
    state.anchors = cloneAnchors(draft.anchors);
    byId("notes").value = draft.notes;
    byId("warnings").value = draft.warnings;
    byId("exclusion-reason").value = draft.exclusionReason;
    byId("interference-tags").querySelectorAll("input").forEach((input) => {
      input.checked = draft.tags.includes(input.value);
    });
    renderCanvas();
  }

  function draftPayload(
    status = null,
    action = "draft",
    anchorsOverride = null,
    draft = captureDraft(),
  ) {
    const effectiveStatus = status || state.detail.status;
    const payload = {
      revision: state.detail.revision,
      action,
      status: effectiveStatus,
      anchors: (anchorsOverride === null ? draft.anchors : anchorsOverride).map((anchor) => ({
        y: Math.round(anchor.y),
        x: Number(anchor.x),
        confidence: 1,
        source: "human",
      })),
      interference_tags: draft.tags,
      exclusion_reason: effectiveStatus === "excluded" ? draft.exclusionReason || null : null,
      notes: draft.notes,
      warnings: warningValues(draft.warnings),
    };
    delete payload.first_reviewed_at;
    delete payload.second_reviewed_at;
    delete payload.origin;
    delete payload.actor;
    delete payload.image_sha256;
    delete payload.preannotation;
    return payload;
  }

  function showSaveResult(message, isError = false) {
    const result = byId("save-result");
    result.textContent = message;
    result.classList.toggle("error", isError);
  }

  async function fetchJson(url, options = undefined) {
    const response = await fetch(url, options);
    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}`);
      error.response = response;
      throw error;
    }
    return response.json();
  }

  async function preserveRevisionConflict(stem) {
    const remote = await fetchJson(`/api/records/${encodeURIComponent(stem)}`);
    if (state.detail?.stem === stem) {
      const editableFields = new Set([
        "anchors", "interference_tags", "exclusion_reason", "notes", "warnings",
      ]);
      const safeMetadata = Object.fromEntries(
        Object.entries(remote).filter(([key]) => !editableFields.has(key)),
      );
      state.detail = { ...state.detail, ...safeMetadata };
      byId("revision-indicator").textContent = `修订 ${remote.revision}（冲突）`;
    }
    showSaveResult("修订冲突：已保留本地未保存内容，请核对后再次保存。", true);
  }

  async function performSave({ status = null, action = "draft", anchorsOverride = null, quiet = false } = {}) {
    if (!state.detail) return false;
    const saveGeneration = editGeneration;
    const saveSnapshot = captureDraft();
    const saveStem = state.detail.stem;
    const payload = draftPayload(status, action, anchorsOverride, saveSnapshot);
    let response;
    try {
      response = await fetch(`/api/records/${encodeURIComponent(saveStem)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (response.status === 409) {
        await preserveRevisionConflict(saveStem);
        saveBlocked = true;
        return false;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const saved = await response.json();
      const recordIndex = state.records.findIndex((record) => record.stem === saved.stem);
      if (recordIndex >= 0) state.records[recordIndex] = { ...state.records[recordIndex], ...saved };

      const editableFields = new Set([
        "anchors", "interference_tags", "exclusion_reason", "notes", "warnings",
      ]);
      const safeMetadata = Object.fromEntries(
        Object.entries(saved).filter(([key]) => !editableFields.has(key)),
      );
      if (state.detail && state.detail.stem === saved.stem) {
        state.detail = { ...state.detail, ...safeMetadata };
        byId("revision-indicator").textContent = `修订 ${saved.revision}`;
      }

      if (editGeneration === saveGeneration && state.detail?.stem === saved.stem) {
        state.detail = { ...state.detail, ...saved };
        state.anchors = deduplicateAnchors(saved.anchors || []);
        if (action !== "draft") {
          if (status === "excluded") storeReasonDraft(saved.stem, saved.exclusion_reason || "");
          else clearReasonDraft(saved.stem);
        }
        populateDraftFields(state.detail);
        renderCanvas();
      } else if (state.detail?.stem === saved.stem) {
        // newer edits remain pending; an older response must not replace them.
        if (action !== "draft") {
          saveBlocked = true;
          showSaveResult("服务器已保存分类，但检测到更新的本地编辑；请重新点击并确认分类。", true);
          return false;
        }
        showSaveResult("较新的本地编辑仍待保存，已安排下一次草稿保存。", false);
        return true;
      }
      saveBlocked = false;
      showSaveResult(quiet ? "草稿已自动保存。" : "保存成功。", false);
      return true;
    } catch (error) {
      saveBlocked = true;
      showSaveResult("保存失败：本地编辑仍保留，请检查服务后重试。", true);
      return false;
    }
  }

  function scheduleAutosave() {
    if (!state.detail) return;
    if (autosaveTimer !== null) window.clearTimeout(autosaveTimer);
    autosaveTimer = window.setTimeout(() => {
      autosaveTimer = null;
      pendingAutosave = pendingAutosave.then((previousSaved) => (
        previousSaved && !saveBlocked ? performSave({ quiet: true }) : false
      ));
    }, AUTOSAVE_DELAY_MS);
  }

  async function awaitPendingAutosave() {
    if (autosaveTimer !== null) {
      window.clearTimeout(autosaveTimer);
      autosaveTimer = null;
      pendingAutosave = pendingAutosave.then((previousSaved) => (
        previousSaved && !saveBlocked ? performSave({ quiet: true }) : false
      ));
    }
    if (saveBlocked) return false;
    return await pendingAutosave;
  }

  async function saveCurrentDraft() {
    const explicitRetry = saveBlocked;
    if (autosaveTimer !== null) {
      window.clearTimeout(autosaveTimer);
      autosaveTimer = null;
    }
    if (!explicitRetry && !await pendingAutosave) {
      showSaveResult("待处理草稿保存失败；请再次点击保存以重试。", true);
      return;
    }
    saveBlocked = false;
    pendingAutosave = performSave();
    await pendingAutosave;
  }

  function recordTimestamp(stem) {
    const record = state.records.find((item) => item.stem === stem);
    return record ? String(record.target_timestamp_seconds ?? "—") : "—";
  }

  function setNeighbor(kind, stem) {
    const button = byId(`${kind}-frame`);
    const timestamp = byId(`${kind}-timestamp`);
    const thumbnail = byId(`${kind}-thumbnail`);
    button.disabled = !stem;
    button.dataset.stem = stem || "";
    timestamp.textContent = stem ? recordTimestamp(stem) : "无";
    if (stem) {
      thumbnail.src = mediaUrl(stem, false);
      thumbnail.alt = `${kind === "previous" ? "上一" : "下一"}帧 ${stem}`;
      thumbnail.hidden = false;
    } else {
      thumbnail.removeAttribute("src");
      thumbnail.alt = "序列边界，无相邻帧";
      thumbnail.hidden = true;
    }
  }

  function showSuggestion(preannotation) {
    if (!preannotation) {
      byId("suggestion-context").textContent = "当前记录没有模型建议。";
      return;
    }
    const metrics = preannotation.source_metrics || {};
    const confidence = preannotation.confidence ?? metrics.confidence ?? "未提供";
    const source = preannotation.source ?? preannotation.status ?? "未知";
    byId("suggestion-context").textContent = `来源：${source}；置信度：${confidence}；锚点：${state.suggestionAnchors.length}。`;
  }

  function populateDraftFields(detail) {
    byId("notes").value = detail.notes || "";
    byId("warnings").value = (detail.warnings || []).join("\n");
    const localReason = loadReasonDraft(detail.stem);
    const reason = localReason === null ? detail.exclusion_reason || "" : localReason;
    byId("exclusion-reason").value = reason;
    if (localReason === null && detail.exclusion_reason) {
      storeReasonDraft(detail.stem, detail.exclusion_reason);
    }
    byId("interference-tags").querySelectorAll("input").forEach((input) => {
      input.checked = (detail.interference_tags || []).includes(input.value);
    });
  }

  async function loadRecord(stem, waitForSave = true) {
    if (!stem) return;
    const requestToken = ++navigationToken;
    if (waitForSave && state.detail && state.detail.stem !== stem) {
      const saved = await awaitPendingAutosave();
      if (requestToken !== navigationToken) return;
      if (!saved) {
        showSaveResult("切换已取消：请先处理当前记录的保存错误。", true);
        return;
      }
    }
    try {
      const detail = await fetchJson(`/api/records/${encodeURIComponent(stem)}`);
      if (requestToken !== navigationToken) return;
      state.detail = detail;
      state.anchors = deduplicateAnchors(detail.anchors || []);
      state.suggestionAnchors = deduplicateAnchors(detail.preannotation?.anchors || []);
      state.selectedIndex = null;
      state.undoStack.length = 0;
      state.redoStack.length = 0;
      byId("current-stem").textContent = detail.stem;
      byId("current-timestamp").textContent = String(detail.target_timestamp_seconds ?? "—");
      byId("revision-indicator").textContent = `修订 ${detail.revision}`;
      byId("record-select").value = detail.stem;
      populateDraftFields(detail);
      showSuggestion(detail.preannotation);
      setNeighbor("previous", detail.previous_stem);
      setNeighbor("next", detail.next_stem);
      byId("preview-toggle").checked = false;
      byId("current-thumbnail").src = mediaUrl(detail.stem, false);
      loadBackground();
      updateHistoryButtons();
      byId("classification-preview").textContent = `当前状态：${CLASS_NAMES[detail.status] || detail.status}`;
      showSaveResult("已加载，等待编辑。", false);
    } catch (error) {
      if (requestToken !== navigationToken) return;
      showSaveResult("记录加载失败，请刷新后重试。", true);
    }
  }

  function recordEventKind(record) {
    if (record.source === "base") return "base";
    if (record.source === "event") return "event";
    return record.event_reason ? "event" : "base";
  }

  function recordsForStatus(status) {
    if (["all", "unreviewed", "needs_second_review"].includes(status)) {
      const byStem = new Map(state.records.map((record) => [record.stem, record]));
      return state.queue.map((stem) => byStem.get(stem)).filter(Boolean);
    }
    return state.records;
  }

  function matchesFilters(record) {
    const status = byId("status-filter").value;
    const event = byId("event-filter").value;
    const warning = byId("warning-filter").value;
    if (status === "unreviewed" && record.status !== "unreviewed") return false;
    if (
      status === "needs_second_review"
      && record.status !== "needs_second_review"
      && !(record.second_review_required && !record.second_reviewed_at)
    ) return false;
    if (["positive", "hard_negative", "excluded"].includes(status) && record.status !== status) return false;
    if (event !== "all" && recordEventKind(record) !== event) return false;
    const hasWarnings = Array.isArray(record.warnings) && record.warnings.length > 0;
    if (warning === "warning" && !hasWarnings) return false;
    if (warning === "clean" && hasWarnings) return false;
    return true;
  }

  async function applyFilters() {
    navigationToken += 1;
    const status = byId("status-filter").value;
    const filtered = recordsForStatus(status).filter(matchesFilters);
    const select = byId("record-select");
    select.replaceChildren();
    filtered.forEach((record) => {
      const option = document.createElement("option");
      option.value = record.stem;
      option.textContent = `${record.target_timestamp_seconds ?? "—"} · ${record.stem} · ${record.status}`;
      select.append(option);
    });
    const currentStillVisible = filtered.some((record) => record.stem === state.detail?.stem);
    if (currentStillVisible) {
      select.value = state.detail.stem;
    } else if (filtered.length) {
      select.value = filtered[0].stem;
      if (state.detail) await loadRecord(filtered[0].stem);
    } else {
      select.value = "";
      showSaveResult("当前筛选条件没有匹配记录。", false);
    }
    return filtered;
  }

  function renderSummary(summary) {
    const statusCounts = summary.counts || summary.status_counts || summary.by_status || {};
    byId("summary-counts").textContent = `总计 ${summary.total ?? state.records.length} · 未复核 ${statusCounts.unreviewed ?? 0} · 需二审 ${statusCounts.needs_second_review ?? 0}`;
  }

  async function refreshLists() {
    const [summary, recordsPayload, queuePayload] = await Promise.all([
      fetchJson("/api/summary"),
      fetchJson("/api/records"),
      fetchJson("/api/review-queue"),
    ]);
    state.records = recordsPayload.records || [];
    state.queue = queuePayload.stems || [];
    renderSummary(summary);
    await applyFilters();
  }

  async function confirmClassification(status) {
    if (!state.detail) return;
    if (status === "positive" && state.anchors.length < 3) {
      showSaveResult("正样本至少需要 3 个锚点。", true);
      return;
    }
    if (status === "hard_negative" && selectedTags().length === 0) {
      showSaveResult("困难负样本至少需要一个干扰标签。", true);
      return;
    }
    if (status === "excluded" && !byId("exclusion-reason").value) {
      showSaveResult("排除记录必须选择排除原因。", true);
      return;
    }
    const className = CLASS_NAMES[status];
    byId("classification-preview").textContent = `正在确认分类：${className}`;
    if (!window.confirm(`确认将当前记录保存为“${className}”？`)) {
      byId("classification-preview").textContent = "已取消分类确认";
      return;
    }
    const explicitRetry = saveBlocked;
    if (explicitRetry) {
      if (autosaveTimer !== null) {
        window.clearTimeout(autosaveTimer);
        autosaveTimer = null;
      }
      saveBlocked = false;
      pendingAutosave = Promise.resolve(true);
    }
    const pendingSaved = await awaitPendingAutosave();
    if (!pendingSaved) {
      byId("classification-preview").textContent = "待重新确认分类";
      showSaveResult("待处理草稿未保存；请解决错误后重新点击并确认分类。", true);
      return;
    }
    const action = status === "needs_second_review" ? "draft" : "finalize";
    const anchorsOverride = status === "hard_negative" ? [] : null;
    pendingAutosave = performSave({ status, action, anchorsOverride });
    const saved = await pendingAutosave;
    if (saved) {
      if (anchorsOverride) state.anchors = [];
      if (status === "excluded") {
        storeReasonDraft(state.detail.stem, byId("exclusion-reason").value);
      } else {
        clearReasonDraft(state.detail.stem);
        byId("exclusion-reason").value = "";
      }
      byId("classification-preview").textContent = `已确认分类：${className}`;
      await refreshLists();
    }
  }

  byId("mark-positive").addEventListener("click", () => confirmClassification('positive'));
  byId("mark-hard-negative").addEventListener("click", () => confirmClassification('hard_negative'));
  byId("mark-excluded").addEventListener("click", () => confirmClassification('excluded'));
  byId("mark-needs-review").addEventListener("click", () => confirmClassification('needs_second_review'));
  byId("save-status").addEventListener("click", saveCurrentDraft);

  ["notes", "warnings"].forEach((id) => {
    byId(id).addEventListener("input", () => {
      markLocalEdit();
      scheduleAutosave();
    });
  });
  byId("exclusion-reason").addEventListener("change", () => {
    markLocalEdit();
    if (state.detail) storeReasonDraft(state.detail.stem, byId("exclusion-reason").value);
    scheduleAutosave();
  });
  byId("interference-tags").addEventListener("change", () => {
    markLocalEdit();
    scheduleAutosave();
  });
  byId("preview-toggle").addEventListener("change", loadBackground);

  ["status-filter", "event-filter", "warning-filter"].forEach((id) => {
    byId(id).addEventListener("change", () => { void applyFilters(); });
  });
  byId("record-select").addEventListener("change", (event) => loadRecord(event.target.value));
  byId("previous-frame").addEventListener("click", (event) => loadRecord(event.currentTarget.dataset.stem));
  byId("next-frame").addEventListener("click", (event) => loadRecord(event.currentTarget.dataset.stem));

  async function initialize() {
    try {
      await refreshLists();
      const initialStem = byId("record-select").value;
      if (initialStem) await loadRecord(initialStem, false);
      else showSaveResult("当前没有可复核记录。", false);
    } catch (error) {
      showSaveResult("工作台加载失败，请确认本地复核服务正在运行。", true);
    }
  }

  updateHistoryButtons();
  initialize();
})();
