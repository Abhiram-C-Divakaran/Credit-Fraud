/**
 * Credit Card Fraud Detection — Client-Side Inference Engine & UI Controller
 * =========================================================================
 * Evaluates transaction risk using a client-side Random Forest model exported
 * from scikit-learn. Consumes shared inference and validation logic from core.js.
 */

(function () {
  'use strict';

  // Fallback check for shared core library
  const Core = window.FraudCore;
  if (!Core) {
    console.error('[FraudDetector] FraudCore library not loaded. Ensure core.js is included.');
  }

  // ───────────────────────────────────────────────────────────────────
  // APPLICATION STATE
  // ───────────────────────────────────────────────────────────────────
  const state = {
    model: null,
    isLoading: false,
    isReady: false,
    hasError: false,
    foreign: 0,
    online: 0,
    totalChecked: 0,
    history: [],
  };

  // ───────────────────────────────────────────────────────────────────
  // THEME MANAGEMENT (Light / Dark Mode)
  // ───────────────────────────────────────────────────────────────────
  function initTheme() {
    const stored = localStorage.getItem('cc_fraud_theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = stored === 'dark' || (stored === null && prefersDark);
    if (isDark) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
    updateThemeIcon();
  }

  function updateThemeIcon() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const isDark = document.body.classList.contains('dark');
    btn.textContent = isDark ? '☀️' : '🌙';
    btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
  }

  function toggleTheme() {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('cc_fraud_theme', isDark ? 'dark' : 'light');
    updateThemeIcon();
  }

  // ───────────────────────────────────────────────────────────────────
  // MODEL LOADING & SCHEMA VALIDATION
  // ───────────────────────────────────────────────────────────────────
  async function loadModel() {
    const badge = document.getElementById('model-status');
    const alertBox = document.getElementById('model-error-alert');
    const analyzeBtn = document.getElementById('analyze-btn');

    state.isLoading = true;
    state.hasError = false;
    state.isReady = false;

    if (badge) {
      badge.textContent = '⏳ Loading Model...';
      badge.className = 'model-badge loading';
    }
    if (alertBox) alertBox.style.display = 'none';
    if (analyzeBtn) analyzeBtn.disabled = true;

    try {
      const response = await fetch('./model.json', { cache: 'no-cache' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }
      const data = await response.json();

      // Validate schema using shared Core
      const schemaCheck = Core.validateModelSchema(data);
      if (!schemaCheck.isValid) {
        throw new Error(`Invalid model structure: ${schemaCheck.error}`);
      }

      state.model = data;
      state.isReady = true;
      state.isLoading = false;

      if (badge) {
        badge.textContent = `● Demo Model Ready (${data.n_trees} Trees)`;
        badge.className = 'model-badge ready';
      }
      if (analyzeBtn) analyzeBtn.disabled = false;

      // Populate dynamic model metrics and feature importance
      renderModelStats(data);
      renderFeatureImportances(data);
    } catch (err) {
      console.error('[FraudDetector] Failed to load model.json:', err);
      state.hasError = true;
      state.isLoading = false;
      state.isReady = false;

      if (badge) {
        badge.textContent = '⚠️ Model Not Loaded';
        badge.className = 'model-badge error';
      }
      if (alertBox) alertBox.style.display = 'block';
      if (analyzeBtn) analyzeBtn.disabled = true;
    }
  }

  // ───────────────────────────────────────────────────────────────────
  // DYNAMIC RENDERING: METRICS & FEATURE IMPORTANCES
  // ───────────────────────────────────────────────────────────────────
  function renderModelStats(model) {
    const metrics = model.metrics || {};
    const accEl = document.getElementById('metric-accuracy');
    const precEl = document.getElementById('metric-precision');
    const recEl = document.getElementById('metric-recall');
    const f1El = document.getElementById('metric-f1');

    if (accEl && typeof metrics.accuracy === 'number') {
      accEl.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
    }
    if (precEl && typeof metrics.precision === 'number') {
      precEl.textContent = `${(metrics.precision * 100).toFixed(1)}%`;
    }
    if (recEl && typeof metrics.recall === 'number') {
      recEl.textContent = `${(metrics.recall * 100).toFixed(1)}%`;
    }
    if (f1El && typeof metrics.f1_score === 'number') {
      f1El.textContent = `${(metrics.f1_score * 100).toFixed(1)}%`;
    }
  }

  function renderFeatureImportances(model) {
    const container = document.getElementById('feature-bars');
    if (!container) return;

    const list = model.feature_importances;
    if (!Array.isArray(list) || list.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted);font-size:12px;">No feature importances available.</p>';
      return;
    }

    container.innerHTML = list.map((item) => {
      const pct = Math.max(0, Math.min(100, item.percentage || 0));
      return `
        <div class="feature-row">
          <span class="feature-name">${escapeHtml(item.name)}</span>
          <div class="feature-bar-bg" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${escapeHtml(item.name)} importance">
            <div class="feature-bar-fill" style="width: ${pct}%"></div>
          </div>
          <span class="feature-pct">${pct}%</span>
        </div>
      `;
    }).join('');
  }

  // ───────────────────────────────────────────────────────────────────
  // FORM INPUT VALIDATION (Powered by FraudCore)
  // ───────────────────────────────────────────────────────────────────
  function validateFieldUI(id, value) {
    const input = document.getElementById(id);
    const errorEl = document.getElementById(`${id}-error`);
    const errorText = Core.validateField(id, value);

    if (errorText) {
      if (input) {
        input.setAttribute('aria-invalid', 'true');
        input.setAttribute('aria-describedby', `${id}-error`);
      }
      if (errorEl) {
        errorEl.textContent = errorText;
        errorEl.classList.add('active');
      }
      return false;
    } else {
      if (input) {
        input.removeAttribute('aria-invalid');
      }
      if (errorEl) {
        errorEl.textContent = '';
        errorEl.classList.remove('active');
      }
      return true;
    }
  }

  function validateAllInputs() {
    let isValid = true;
    let firstInvalidInput = null;

    const amountVal = document.getElementById('amount').value.trim();
    const hourVal = document.getElementById('hour').value.trim();
    const distanceVal = document.getElementById('distance').value.trim();
    const frequencyVal = document.getElementById('frequency').value.trim();

    const fields = [
      { id: 'amount', val: amountVal },
      { id: 'hour', val: hourVal },
      { id: 'distance', val: distanceVal },
      { id: 'frequency', val: frequencyVal },
    ];

    for (const f of fields) {
      const valid = validateFieldUI(f.id, f.val);
      if (!valid && !firstInvalidInput) {
        firstInvalidInput = document.getElementById(f.id);
      }
      isValid = isValid && valid;
    }

    if (!isValid && firstInvalidInput) {
      firstInvalidInput.focus();
    }

    return isValid;
  }

  function attachFieldListeners() {
    ['amount', 'hour', 'distance', 'frequency'].forEach((id) => {
      const input = document.getElementById(id);
      if (!input) return;
      input.addEventListener('input', () => {
        const errorEl = document.getElementById(`${id}-error`);
        if (errorEl && errorEl.classList.contains('active')) {
          input.removeAttribute('aria-invalid');
          errorEl.textContent = '';
          errorEl.classList.remove('active');
        }
      });
    });
  }

  // ───────────────────────────────────────────────────────────────────
  // ACCESSIBLE TOGGLE BUTTONS (Foreign & Online)
  // ───────────────────────────────────────────────────────────────────
  function setToggle(type, isTrue) {
    if (type === 'foreign') {
      state.foreign = isTrue ? 1 : 0;
      updateToggleUI('foreign-no', 'foreign-yes', isTrue);
    } else if (type === 'online') {
      state.online = isTrue ? 1 : 0;
      updateToggleUI('online-no', 'online-yes', isTrue);
    }
  }

  function updateToggleUI(noBtnId, yesBtnId, isTrue) {
    const noBtn = document.getElementById(noBtnId);
    const yesBtn = document.getElementById(yesBtnId);
    if (!noBtn || !yesBtn) return;

    if (isTrue) {
      yesBtn.classList.add('active');
      yesBtn.setAttribute('aria-checked', 'true');
      noBtn.classList.remove('active');
      noBtn.setAttribute('aria-checked', 'false');
    } else {
      noBtn.classList.add('active');
      noBtn.setAttribute('aria-checked', 'true');
      yesBtn.classList.remove('active');
      yesBtn.setAttribute('aria-checked', 'false');
    }
  }

  // ───────────────────────────────────────────────────────────────────
  // RUN DETECTION & UPDATE UI (Powered by FraudCore)
  // ───────────────────────────────────────────────────────────────────
  function runDetection() {
    if (!state.model) {
      alert('The machine learning model is still loading or could not be loaded. Please check the status banner.');
      return;
    }

    if (!validateAllInputs()) {
      return;
    }

    const amount = parseFloat(document.getElementById('amount').value);
    const hour = parseInt(document.getElementById('hour').value, 10);
    const distance = parseFloat(document.getElementById('distance').value);
    const frequency = parseInt(document.getElementById('frequency').value, 10);
    const foreign = state.foreign;
    const online = state.online;

    const rawFeatures = [amount, hour, distance, frequency, foreign, online];
    const { fraudProbability, isFraud } = Core.runInference(rawFeatures, state.model);
    const fraudPct = (fraudProbability * 100).toFixed(1);

    // Update Result Panel
    const panel = document.getElementById('result-panel');
    const icon = document.getElementById('result-icon');
    const label = document.getElementById('result-label');
    const probText = document.getElementById('result-prob');
    const bar = document.getElementById('prob-bar');
    const needle = document.getElementById('meter-needle');
    const announcer = document.getElementById('result-announcer');

    if (panel) {
      panel.style.display = 'block';
      panel.className = `result-panel ${isFraud ? 'fraud' : 'safe'}`;
    }
    if (icon) icon.textContent = isFraud ? '🚨' : '✅';
    if (label) label.textContent = isFraud ? 'FRAUD DETECTED' : 'TRANSACTION SAFE';
    if (probText) {
      probText.textContent = `Demo Risk Score: ${fraudPct}% (Threshold: 50.0%)`;
    }
    if (bar) {
      bar.style.width = '0%';
      setTimeout(() => {
        bar.style.width = `${Math.min(100, Math.max(0, fraudPct))}%`;
      }, 50);
    }
    if (needle) {
      needle.style.left = `${Math.min(100, Math.max(0, fraudPct))}%`;
    }

    // Accessible screen reader announcement
    if (announcer) {
      announcer.textContent = `Detection result: ${isFraud ? 'Fraud detected' : 'Transaction safe'}. Demonstration risk score is ${fraudPct} percent.`;
    }

    // Update total count
    state.totalChecked += 1;
    const totalCountEl = document.getElementById('total-count');
    if (totalCountEl) totalCountEl.textContent = state.totalChecked;

    // Record into history
    state.history.unshift({
      amount,
      hour,
      distance,
      frequency,
      foreign,
      online,
      fraudProbability,
      isFraud,
      timestamp: new Date(),
    });
    renderHistory();
  }

  function renderHistory() {
    const list = document.getElementById('history-list');
    if (!list) return;

    if (state.history.length === 0) {
      list.innerHTML = `
        <div class="empty-history">
          <span class="emoji" aria-hidden="true">🔍</span>
          No transactions analyzed yet.
        </div>
      `;
      return;
    }

    list.innerHTML = state.history.slice(0, 10).map((item) => {
      const timeStr = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      return `
        <div class="history-item">
          <div class="history-dot ${item.isFraud ? 'h-fraud' : 'h-safe'}" aria-hidden="true"></div>
          <div class="history-info">
            <div class="h-amt">$${item.amount.toFixed(2)}</div>
            <div class="h-detail">${timeStr} · ${item.distance}km · Hour ${item.hour} · ${item.frequency} txns/hr</div>
          </div>
          <span class="history-tag ${item.isFraud ? 'h-fraud-tag' : 'h-safe-tag'}">
            ${item.isFraud ? 'FRAUD' : 'SAFE'} (${(item.fraudProbability * 100).toFixed(0)}%)
          </span>
        </div>
      `;
    }).join('');
  }

  // ───────────────────────────────────────────────────────────────────
  // QUICK TEST PRESETS
  // ───────────────────────────────────────────────────────────────────
  function loadPreset(type) {
    const presets = {
      normal: { amount: 65.0, hour: 14, distance: 12.0, frequency: 2, foreign: 0, online: 0 },
      suspicious: { amount: 3200.0, hour: 2, distance: 350.0, frequency: 9, foreign: 1, online: 1 },
      large: { amount: 1850.0, hour: 11, distance: 45.0, frequency: 3, foreign: 0, online: 1 },
      night: { amount: 120.0, hour: 3, distance: 15.0, frequency: 7, foreign: 0, online: 0 },
    };

    const p = presets[type];
    if (!p) return;

    const amountInput = document.getElementById('amount');
    const hourInput = document.getElementById('hour');
    const distanceInput = document.getElementById('distance');
    const freqInput = document.getElementById('frequency');

    if (amountInput) amountInput.value = p.amount;
    if (hourInput) hourInput.value = p.hour;
    if (distanceInput) distanceInput.value = p.distance;
    if (freqInput) freqInput.value = p.frequency;

    setToggle('foreign', p.foreign === 1);
    setToggle('online', p.online === 1);

    // Clear previous validation errors
    ['amount', 'hour', 'distance', 'frequency'].forEach((id) => {
      const input = document.getElementById(id);
      const err = document.getElementById(`${id}-error`);
      if (input) input.removeAttribute('aria-invalid');
      if (err) {
        err.textContent = '';
        err.classList.remove('active');
      }
    });

    runDetection();
  }


  // ───────────────────────────────────────────────────────────────────
  // BENCHMARK LEARNING LAB CONTROLLER & STATE
  // ───────────────────────────────────────────────────────────────────
  const CANDIDATE_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95];

  const benchmarkState = {
    report: null,
    isLoaded: false,
    activeModel: 'random_forest', // 'random_forest' | 'baseline_logistic_regression'
    thresholdIndex: 3,            // Default candidate 0.30 (index 3)
    activeScenario: 'catch_fraud',
    activeWindow: 'all',          // 'all' | '1' | '2' | '3'
  };

  /**
   * Load real benchmark data from reports/real_data_evaluation.json.
   * Seamlessly falls back to window.BENCHMARK_REPORT_DATA when file:/// protocol prevents fetch.
   */
  async function loadBenchmarkData() {
    const statusBadge = document.getElementById('benchmark-status');
    try {
      let data = null;
      try {
        const response = await fetch('./reports/real_data_evaluation.json', { cache: 'no-cache' });
        if (response.ok) {
          data = await response.json();
        }
      } catch (fetchErr) {
        console.warn('[FraudLab] Fetch failed (possibly local file:/// protocol), checking embedded cache...', fetchErr);
      }

      // Fallback to static cache if fetch was blocked or failed
      if (!data && typeof window.BENCHMARK_REPORT_DATA === 'object' && window.BENCHMARK_REPORT_DATA) {
        data = window.BENCHMARK_REPORT_DATA;
      }

      if (!data) {
        throw new Error('Benchmark data could not be loaded via fetch or static bundle.');
      }

      // Validate schema
      const val = Core.validateBenchmarkReportData(data);
      if (!val.isValid) {
        throw new Error(`Benchmark schema validation failed: ${val.error}`);
      }

      benchmarkState.report = data;
      benchmarkState.isLoaded = true;

      if (statusBadge) {
        statusBadge.textContent = '● Benchmark Evidence Ready';
        statusBadge.className = 'model-badge ready';
      }

      initBenchmarkUI();
    } catch (err) {
      console.error('[FraudLab] Failed to load benchmark data:', err);
      if (statusBadge) {
        statusBadge.textContent = '⚠️ Benchmark Data Offline';
        statusBadge.className = 'model-badge error';
      }
    }
  }

  function initBenchmarkUI() {
    if (!benchmarkState.report) return;

    // Set dataset provenance hash
    const shaEl = document.getElementById('bench-sha256');
    if (shaEl && benchmarkState.report.dataset_metadata) {
      const fullHash = benchmarkState.report.dataset_metadata.file_sha256 || '';
      shaEl.textContent = fullHash.substring(0, 16) + '...';
      shaEl.title = fullHash;
    }

    renderTemporalTable();
    updateBenchmarkDisplay();
  }

  function setBenchmarkModel(modelKey) {
    if (modelKey !== 'random_forest' && modelKey !== 'baseline_logistic_regression') return;
    benchmarkState.activeModel = modelKey;

    const btnRf = document.getElementById('bench-model-rf');
    const btnLr = document.getElementById('bench-model-lr');

    if (btnRf && btnLr) {
      const isRf = modelKey === 'random_forest';
      btnRf.classList.toggle('active', isRf);
      btnRf.setAttribute('aria-checked', isRf ? 'true' : 'false');
      btnLr.classList.toggle('active', !isRf);
      btnLr.setAttribute('aria-checked', !isRf ? 'true' : 'false');
    }

    // Default scenario for newly selected model
    selectScenario(benchmarkState.activeScenario || 'catch_fraud');
  }

  function onThresholdSliderInput(sliderVal) {
    const idx = Math.max(0, Math.min(CANDIDATE_THRESHOLDS.length - 1, parseInt(sliderVal, 10)));
    benchmarkState.thresholdIndex = idx;
    benchmarkState.activeScenario = null; // User custom adjusted
    updateScenarioButtonHighlight();
    updateBenchmarkDisplay();
  }

  function selectScenario(scenarioId) {
    benchmarkState.activeScenario = scenarioId;
    const modelKey = benchmarkState.activeModel;
    let targetThreshold = null;

    if (benchmarkState.report && benchmarkState.report.operating_scenarios && benchmarkState.report.operating_scenarios[modelKey]) {
      const scList = benchmarkState.report.operating_scenarios[modelKey];
      const match = scList.find((s) => s.id === scenarioId);
      if (match && typeof match.selected_threshold === 'number') {
        targetThreshold = match.selected_threshold;
      }
    }

    if (targetThreshold === null) {
      const isRf = modelKey === 'random_forest';
      if (scenarioId === 'catch_fraud') {
        targetThreshold = isRf ? 0.30 : 0.95;
      } else if (scenarioId === 'balanced_f1') {
        targetThreshold = isRf ? 0.40 : 0.95;
      } else if (scenarioId === 'reduce_alarms') {
        targetThreshold = isRf ? 0.80 : 0.95;
      } else {
        targetThreshold = 0.50;
      }
    }

    // Match closest candidate index
    let bestIdx = 0;
    let minDiff = 999;
    for (let i = 0; i < CANDIDATE_THRESHOLDS.length; i++) {
      const diff = Math.abs(CANDIDATE_THRESHOLDS[i] - targetThreshold);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = i;
      }
    }
    benchmarkState.thresholdIndex = bestIdx;

    const slider = document.getElementById('benchmark-threshold-slider');
    if (slider) {
      slider.value = benchmarkState.thresholdIndex;
    }

    updateScenarioButtonHighlight();
    updateBenchmarkDisplay();
  }

  function updateScenarioButtonHighlight() {
    ['catch_fraud', 'balanced_f1', 'reduce_alarms'].forEach((id) => {
      const btn = document.getElementById(`scenario-${id}`);
      if (btn) {
        btn.classList.toggle('active', benchmarkState.activeScenario === id);
      }
    });
  }

  function setTemporalWindow(windowKey) {
    benchmarkState.activeWindow = windowKey;

    ['all', '1', '2', '3'].forEach((k) => {
      const btn = document.getElementById(`win-btn-${k}`);
      if (btn) {
        const isActive = k === windowKey;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
      }
    });

    highlightTemporalRows();
  }

  function highlightTemporalRows() {
    const rows = document.querySelectorAll('#temporal-table-body tr');
    rows.forEach((row) => {
      const wId = row.getAttribute('data-window');
      if (!wId) return;
      if (benchmarkState.activeWindow === 'all' || wId === benchmarkState.activeWindow) {
        row.style.opacity = '1.0';
      } else {
        row.style.opacity = '0.35';
      }
    });
  }

  function updateBenchmarkDisplay() {
    if (!benchmarkState.report) return;

    const tradeoffs = benchmarkState.report.threshold_tradeoff_analysis[benchmarkState.activeModel];
    if (!Array.isArray(tradeoffs) || tradeoffs.length === 0) return;

    const currentItem = tradeoffs[benchmarkState.thresholdIndex] || tradeoffs[3];
    const baselineItem = tradeoffs.find((r) => Math.abs(r.threshold - 0.50) < 0.001) || tradeoffs[5];
    const isRf = benchmarkState.activeModel === 'random_forest';
    const modelDisplayName = isRf ? 'Random Forest' : 'Logistic Regression';

    // 1. Threshold display & slider accessibility
    const tValEl = document.getElementById('threshold-val');
    const sliderEl = document.getElementById('benchmark-threshold-slider');
    if (tValEl) tValEl.textContent = currentItem.threshold.toFixed(2);
    if (sliderEl) {
      sliderEl.setAttribute('aria-valuenow', currentItem.threshold.toFixed(2));
      sliderEl.setAttribute(
        'aria-valuetext',
        `Threshold ${currentItem.threshold.toFixed(2)}, Recall ${(currentItem.recall * 100).toFixed(1)}%, False Positives ${currentItem.confusion_matrix.false_positives}`
      );
    }

    // 2. Tuned threshold badge
    const badgeEl = document.getElementById('tuned-threshold-badge');
    if (badgeEl) {
      if (isRf) {
        badgeEl.innerHTML = 'Validation-Tuned Operating Threshold: <strong>0.399</strong> (Peak F1: 81.8%)';
      } else {
        badgeEl.innerHTML = 'Validation-Tuned Operating Threshold: <strong>0.950</strong> (F1: 51.4%, Class Imbalance Tuned)';
      }
    }

    // 3. Dynamic Narrative & Comparison
    const narrative = Core.generateThresholdNarrative(currentItem, baselineItem, modelDisplayName);
    const nTitle = document.getElementById('narrative-title');
    const nSummary = document.getElementById('narrative-summary');
    const nTradeoff = document.getElementById('narrative-tradeoff');
    const nIcon = document.getElementById('narrative-icon');

    if (benchmarkState.activeScenario === null) {
      if (nTitle) nTitle.textContent = `Retrospective Sensitivity Analysis (Cutoff: ${currentItem.threshold.toFixed(2)})`;
      if (nSummary) {
        nSummary.textContent = `Retrospective exploratory evaluation at cutoff ${currentItem.threshold.toFixed(2)} on held-out test data (${(currentItem.recall * 100).toFixed(1)}% recall, ${cm.false_positives} false alarms). Note: Slider adjustments are for retrospective sensitivity analysis only. Operating thresholds must be selected on validation data.`;
      }
      if (nTradeoff) nTradeoff.textContent = narrative.tradeOffComparison;
      if (nIcon) nIcon.textContent = '🔬';
    } else {
      if (nTitle) nTitle.textContent = narrative.headline;
      if (nSummary) nSummary.textContent = narrative.summary;
      if (nTradeoff) nTradeoff.textContent = narrative.tradeOffComparison;
      if (nIcon) {
        if (currentItem.threshold <= 0.30) nIcon.textContent = '🎯';
        else if (currentItem.threshold <= 0.50) nIcon.textContent = '⚖️';
        else nIcon.textContent = '🛡️';
      }
    }

    // 4. Evidence Metric Cards
    const recEl = document.getElementById('bench-metric-recall');
    const subRecEl = document.getElementById('bench-sub-recall');
    const precEl = document.getElementById('bench-metric-precision');
    const subPrecEl = document.getElementById('bench-sub-precision');
    const f1El = document.getElementById('bench-metric-f1');
    const subF1El = document.getElementById('bench-sub-f1');
    const fprEl = document.getElementById('bench-metric-fpr');
    const subFprEl = document.getElementById('bench-sub-fpr');

    const cm = currentItem.confusion_matrix;
    const totalFrauds = cm.true_positives + cm.false_negatives;
    const totalLegit = cm.true_negatives + cm.false_positives;

    if (recEl) recEl.textContent = `${(currentItem.recall * 100).toFixed(1)}%`;
    if (subRecEl) subRecEl.textContent = `${cm.true_positives} of ${totalFrauds} frauds caught`;

    if (precEl) precEl.textContent = `${(currentItem.precision * 100).toFixed(1)}%`;
    if (subPrecEl) subPrecEl.textContent = `${cm.false_positives} false alarms on legit txns`;

    if (f1El) f1El.textContent = currentItem.f1_score.toFixed(3);
    if (subF1El) {
      subF1El.textContent = isRf ? 'Validation tuned peak: 0.818 at t=0.40' : 'Validation tuned peak: 0.514 at t=0.95';
    }

    if (fprEl) fprEl.textContent = `${(currentItem.false_positive_rate * 100).toFixed(4)}%`;
    if (subFprEl) subFprEl.textContent = `${cm.false_positives} / ${totalLegit.toLocaleString()} legit flagged`;

    // 5. 2x2 Confusion Matrix Grid
    const cmThresh = document.getElementById('cm-threshold-label');
    const cmTp = document.getElementById('cm-tp-val');
    const cmTpPct = document.getElementById('cm-tp-pct');
    const cmFn = document.getElementById('cm-fn-val');
    const cmFnPct = document.getElementById('cm-fn-pct');
    const cmFp = document.getElementById('cm-fp-val');
    const cmFpPct = document.getElementById('cm-fp-pct');
    const cmTn = document.getElementById('cm-tn-val');

    if (cmThresh) cmThresh.textContent = currentItem.threshold.toFixed(2);
    if (cmTp) cmTp.textContent = cm.true_positives.toLocaleString();
    if (cmTpPct) cmTpPct.textContent = `${(currentItem.recall * 100).toFixed(1)}% attacks caught`;

    if (cmFn) cmFn.textContent = cm.false_negatives.toLocaleString();
    if (cmFnPct) cmFnPct.textContent = `${((1 - currentItem.recall) * 100).toFixed(1)}% attacks missed`;

    if (cmFp) cmFp.textContent = cm.false_positives.toLocaleString();
    if (cmFpPct) cmFpPct.textContent = `${cm.false_positives} false alarms (${(currentItem.false_positive_rate * 100).toFixed(3)}% FPR)`;

    if (cmTn) cmTn.textContent = cm.true_negatives.toLocaleString();

    // 6. Operational Economics text
    const ecoText = document.getElementById('tradeoff-economics-text');
    if (ecoText) {
      ecoText.innerHTML = `Operating at threshold <strong>${currentItem.threshold.toFixed(2)}</strong> intercepts <strong>${cm.true_positives}</strong> fraudulent transactions while declining <strong>${cm.false_positives}</strong> legitimate cardholders and letting <strong>${cm.false_negatives}</strong> attacks slip through.`;
    }

    // 7. Scenario Presets: Badges, Rules, and Validation vs Held-Out Test Stats
    const catchBadge = document.getElementById('scenario-badge-catch');
    const balBadge = document.getElementById('scenario-badge-balanced');
    const redBadge = document.getElementById('scenario-badge-reduce');

    const catchRule = document.getElementById('scenario-rule-catch');
    const balRule = document.getElementById('scenario-rule-balanced');
    const redRule = document.getElementById('scenario-rule-reduce');

    const catchStats = document.getElementById('scenario-stats-catch');
    const balStats = document.getElementById('scenario-stats-balanced');
    const redStats = document.getElementById('scenario-stats-reduce');

    if (isRf) {
      if (catchBadge) catchBadge.textContent = 'Recommended Operating Point';
      if (balBadge) balBadge.textContent = 'Validation Optimum';
      if (redBadge) {
        redBadge.textContent = 'High Precision (0 FP)';
        redBadge.className = 'scenario-badge';
      }

      if (catchRule) catchRule.textContent = 'Validation Rule: Recall_val ≥ 75%, maximizing precision';
      if (balRule) balRule.textContent = 'Validation Rule: Maximize F1-score (harmonic mean)';
      if (redRule) redRule.textContent = 'Validation Rule: FP_val == 0 while Recall_val > 0';

      if (catchStats) catchStats.textContent = 'Validation: 75.4% Recall, 16 FP • Held-out Test: 76.0% Recall, 16 FP';
      if (balStats) balStats.textContent = 'Validation: 83.2% F1, 2 FP • Held-out Test: 81.8% F1, 6 FP';
      if (redStats) redStats.textContent = 'Validation: 0 FP, 7.0% Recall • Held-out Test: 0 FP, 12.0% Recall';
    } else {
      if (catchBadge) catchBadge.textContent = 'Recommended Operating Point';
      if (balBadge) balBadge.textContent = 'Validation Optimum';
      if (redBadge) {
        redBadge.textContent = '⚠️ Illustrative Example';
        redBadge.className = 'scenario-badge illustrative';
      }

      if (catchRule) catchRule.textContent = 'Validation Rule: Recall_val ≥ 75%, maximizing precision';
      if (balRule) balRule.textContent = 'Validation Rule: Maximize F1-score (harmonic mean)';
      if (redRule) redRule.textContent = 'Validation: 0 FP is unattainable on validation data (min 136 FP)';

      if (catchStats) catchStats.textContent = 'Validation: 80.7% Recall, 136 FP • Held-out Test: 84.0% Recall, 107 FP';
      if (balStats) balStats.textContent = 'Validation: 38.5% F1, 136 FP • Held-out Test: 51.4% F1, 107 FP';
      if (redStats) redStats.textContent = 'Illustrative test-set example (zero false alarms unsupported on validation data)';
    }

    // 8. Raw Evidence Accordion Slice
    const rawEl = document.getElementById('raw-evidence-json');
    if (rawEl) {
      const rawSlice = {
        scope: "Real-world Kaggle/ULB Benchmark (reports/real_data_evaluation.json)",
        model_evaluated: modelDisplayName,
        decision_threshold: currentItem.threshold,
        active_scenario: benchmarkState.activeScenario,
        operating_scenarios: benchmarkState.report.operating_scenarios
          ? benchmarkState.report.operating_scenarios[benchmarkState.activeModel]
          : null,
        metrics: {
          recall: currentItem.recall,
          precision: currentItem.precision,
          f1_score: currentItem.f1_score,
          false_positive_rate: currentItem.false_positive_rate,
          roc_auc: currentItem.roc_auc,
          pr_auc: currentItem.pr_auc,
        },
        confusion_matrix: currentItem.confusion_matrix,
        reference_tuned_operating_threshold: isRf
          ? benchmarkState.report.results.random_forest.validation_tuned_threshold
          : benchmarkState.report.results.baseline_logistic_regression.validation_tuned_threshold,
      };
      rawEl.textContent = JSON.stringify(rawSlice, null, 2);
    }

    // Announce to screen reader
    const announcer = document.getElementById('result-announcer');
    if (announcer) {
      announcer.textContent = `Benchmark updated: ${modelDisplayName} at threshold ${currentItem.threshold.toFixed(2)}. Recall ${(currentItem.recall * 100).toFixed(1)} percent, precision ${(currentItem.precision * 100).toFixed(1)} percent, ${cm.false_positives} false alarms.`;
    }
  }

  function renderTemporalTable() {
    const tbody = document.getElementById('temporal-table-body');
    if (!tbody || !benchmarkState.report || !benchmarkState.report.temporal_window_evaluation) return;

    const temporal = benchmarkState.report.temporal_window_evaluation;
    const windows = temporal.windows || [];
    let rowsHtml = '';

    windows.forEach((w) => {
      const idx = w.window_index;
      const counts = w.window_counts;
      const rf = w.random_forest.test_at_tuned_threshold;
      const lr = w.baseline_logistic_regression.test_at_tuned_threshold;

      // Random Forest Row
      rowsHtml += `
        <tr data-window="${idx}" class="temporal-rf-row">
          <td><strong>🌲 Random Forest (W${idx})</strong></td>
          <td>${counts.test_time_range_seconds[0].toFixed(0)}s &ndash; ${counts.test_time_range_seconds[1].toFixed(0)}s</td>
          <td>${counts.train_samples.toLocaleString()} (${counts.train_frauds})</td>
          <td>${counts.test_samples.toLocaleString()} (${counts.test_frauds})</td>
          <td>${rf.threshold.toFixed(4)}</td>
          <td><strong>${rf.pr_auc.toFixed(4)}</strong></td>
          <td>${rf.roc_auc.toFixed(4)}</td>
          <td>${(rf.precision * 100).toFixed(1)}%</td>
          <td>${(rf.recall * 100).toFixed(1)}%</td>
          <td>${(rf.f1_score * 100).toFixed(1)}%</td>
          <td>${rf.confusion_matrix.false_positives} (${(rf.false_positive_rate * 100).toFixed(4)}%)</td>
        </tr>
      `;

      // Logistic Regression Row
      rowsHtml += `
        <tr data-window="${idx}" class="temporal-lr-row" style="color:var(--text-secondary);">
          <td>📈 Baseline LR (W${idx})</td>
          <td>${counts.test_time_range_seconds[0].toFixed(0)}s &ndash; ${counts.test_time_range_seconds[1].toFixed(0)}s</td>
          <td>${counts.train_samples.toLocaleString()} (${counts.train_frauds})</td>
          <td>${counts.test_samples.toLocaleString()} (${counts.test_frauds})</td>
          <td>${lr.threshold.toFixed(4)}</td>
          <td>${lr.pr_auc.toFixed(4)}</td>
          <td>${lr.roc_auc.toFixed(4)}</td>
          <td>${(lr.precision * 100).toFixed(1)}%</td>
          <td>${(lr.recall * 100).toFixed(1)}%</td>
          <td>${(lr.f1_score * 100).toFixed(1)}%</td>
          <td>${lr.confusion_matrix.false_positives.toLocaleString()} (${(lr.false_positive_rate * 100).toFixed(4)}%)</td>
        </tr>
      `;
    });

    // Add aggregate multi-window row
    if (temporal.aggregate_metrics) {
      const aggRf = temporal.aggregate_metrics.random_forest.tuned_threshold_aggregate.metrics_summary;
      const aggLr = temporal.aggregate_metrics.baseline_logistic_regression.tuned_threshold_aggregate.metrics_summary;

      rowsHtml += `
        <tr class="agg-row" style="border-top: 2px solid var(--primary);">
          <td><strong>🌲 RF Aggregate (Mean &plusmn; Std)</strong></td>
          <td>3 Windows Total</td>
          <td>Expanding</td>
          <td>170,885 total</td>
          <td>Tuned/Val</td>
          <td><strong>${aggRf.pr_auc.mean} &plusmn; ${aggRf.pr_auc.std}</strong></td>
          <td>${aggRf.roc_auc.mean}</td>
          <td>${(aggRf.precision.mean * 100).toFixed(1)}% &plusmn; ${(aggRf.precision.std * 100).toFixed(1)}%</td>
          <td>${(aggRf.recall.mean * 100).toFixed(1)}% &plusmn; ${(aggRf.recall.std * 100).toFixed(1)}%</td>
          <td>${(aggRf.f1_score.mean * 100).toFixed(1)}%</td>
          <td>Spread: ${aggRf.pr_auc.spread}</td>
        </tr>
        <tr class="agg-row" style="color:var(--text-secondary);">
          <td><strong>📈 LR Aggregate (Mean &plusmn; Std)</strong></td>
          <td>3 Windows Total</td>
          <td>Expanding</td>
          <td>170,885 total</td>
          <td>0.9500</td>
          <td>${aggLr.pr_auc.mean} &plusmn; ${aggLr.pr_auc.std}</td>
          <td>${aggLr.roc_auc.mean}</td>
          <td>${(aggLr.precision.mean * 100).toFixed(1)}% &plusmn; ${(aggLr.precision.std * 100).toFixed(1)}%</td>
          <td>${(aggLr.recall.mean * 100).toFixed(1)}% &plusmn; ${(aggLr.recall.std * 100).toFixed(1)}%</td>
          <td>${(aggLr.f1_score.mean * 100).toFixed(1)}%</td>
          <td>Spread: ${aggLr.precision.spread} (Drift)</td>
        </tr>
      `;
    }

    tbody.innerHTML = rowsHtml;
  }


  // ───────────────────────────────────────────────────────────────────
  // NAVIGATION (Landing vs Detection View vs Benchmark Learning Lab)
  // ───────────────────────────────────────────────────────────────────
  function showDetection() {
    const landing = document.getElementById('landing');
    const detection = document.getElementById('detection');
    const lab = document.getElementById('learning-lab');
    if (landing) landing.style.display = 'none';
    if (lab) lab.style.display = 'none';
    if (detection) {
      detection.style.display = 'block';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    updateNavTabs('demo');
  }

  function showLearningLab() {
    const landing = document.getElementById('landing');
    const detection = document.getElementById('detection');
    const lab = document.getElementById('learning-lab');
    if (landing) landing.style.display = 'none';
    if (detection) detection.style.display = 'none';
    if (lab) {
      lab.style.display = 'block';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    updateNavTabs('lab');
    if (benchmarkState.isLoaded) {
      updateBenchmarkDisplay();
    } else {
      loadBenchmarkData();
    }
  }

  function showLanding() {
    const detection = document.getElementById('detection');
    const lab = document.getElementById('learning-lab');
    const landing = document.getElementById('landing');
    if (detection) detection.style.display = 'none';
    if (lab) lab.style.display = 'none';
    if (landing) {
      landing.style.display = 'flex';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  function updateNavTabs(activeTab) {
    const isDemo = activeTab === 'demo';
    const pairs = [
      ['nav-btn-demo', 'nav-btn-lab'],
      ['detect-tab-demo', 'detect-tab-lab'],
      ['lab-tab-demo', 'lab-tab-lab'],
    ];

    pairs.forEach(([demoId, labId]) => {
      const demoBtn = document.getElementById(demoId);
      const labBtn = document.getElementById(labId);
      if (demoBtn) {
        demoBtn.classList.toggle('active', isDemo);
        demoBtn.setAttribute('aria-selected', isDemo ? 'true' : 'false');
      }
      if (labBtn) {
        labBtn.classList.toggle('active', !isDemo);
        labBtn.setAttribute('aria-selected', !isDemo ? 'true' : 'false');
      }
    });
  }


  // ───────────────────────────────────────────────────────────────────
  // UTILITIES
  // ───────────────────────────────────────────────────────────────────
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ───────────────────────────────────────────────────────────────────
  // EXPOSE GLOBAL API & INITIALIZE ON DOM READY
  // ───────────────────────────────────────────────────────────────────
  window.FraudApp = {
    state,
    benchmarkState,
    loadModel,
    runDetection,
    loadPreset,
    setToggle,
    showDetection,
    showLearningLab,
    showLanding,
    toggleTheme,
    validateAllInputs,
    loadBenchmarkData,
    setBenchmarkModel,
    onThresholdSliderInput,
    selectScenario,
    setTemporalWindow,
  };

  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    attachFieldListeners();
    loadModel();
    loadBenchmarkData();

    // Event listener bindings
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

    const retryBtn = document.getElementById('retry-load-btn');
    if (retryBtn) retryBtn.addEventListener('click', loadModel);

    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) analyzeBtn.addEventListener('click', runDetection);

    const startBtn = document.getElementById('hero-detect-btn');
    if (startBtn) startBtn.addEventListener('click', showDetection);

    const backBtn = document.getElementById('nav-back-btn');
    if (backBtn) backBtn.addEventListener('click', showLanding);
  });
})();
