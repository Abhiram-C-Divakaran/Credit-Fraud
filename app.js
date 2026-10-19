/**
 * Credit Card Fraud Detection — Client-Side Inference Engine & UI Controller
 * =========================================================================
 * Evaluates transaction risk using a client-side Random Forest model exported
 * from scikit-learn. No server-side runtime, no API keys, full user privacy.
 */

(function () {
  'use strict';

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

      // Validate schema
      if (!data.scaler || !data.trees || !Array.isArray(data.trees) || data.trees.length === 0) {
        throw new Error('Model payload missing required scaler or trees data.');
      }
      if (!data.metrics || typeof data.metrics.accuracy !== 'number') {
        throw new Error('Model payload missing evaluation metrics.');
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
  // INFERENCE ENGINE: TREE TRAVERSAL & STANDARDIZATION
  // ───────────────────────────────────────────────────────────────────
  function predictTree(node, scaledFeatures) {
    if (node.leaf) {
      return node.prob;
    }
    const val = scaledFeatures[node.feature];
    return val <= node.threshold
      ? predictTree(node.left, scaledFeatures)
      : predictTree(node.right, scaledFeatures);
  }

  function standardize(rawFeatures, scaler) {
    const { mean, scale } = scaler;
    return rawFeatures.map((v, i) => (v - mean[i]) / scale[i]);
  }

  function runInference(rawFeatures) {
    if (!state.model) {
      throw new Error('Model is not initialized.');
    }
    const scaled = standardize(rawFeatures, state.model.scaler);
    const treeProbs = state.model.trees.map((tree) => predictTree(tree, scaled));
    const avgProb = treeProbs.reduce((acc, p) => acc + p, 0) / treeProbs.length;
    const threshold = (state.model.metadata && state.model.metadata.decision_threshold) || 0.5;
    return {
      fraudProbability: avgProb,
      isFraud: avgProb >= threshold,
      threshold,
      treeProbs,
    };
  }

  // ───────────────────────────────────────────────────────────────────
  // FORM INPUT VALIDATION
  // ───────────────────────────────────────────────────────────────────
  function validateInput(id, value, rule) {
    const input = document.getElementById(id);
    const errorEl = document.getElementById(`${id}-error`);
    const errorText = rule(value);

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

    // Amount validation: required, numeric, >= 0, finite
    const amountValid = validateInput('amount', amountVal, (val) => {
      if (val === '') return 'Transaction amount is required.';
      const num = Number(val);
      if (Number.isNaN(num) || !Number.isFinite(num)) return 'Enter a valid numeric amount.';
      if (num < 0) return 'Amount cannot be negative.';
      if (num > 1000000) return 'Amount exceeds the demo maximum ($1,000,000).';
      return null;
    });
    if (!amountValid && !firstInvalidInput) firstInvalidInput = document.getElementById('amount');
    isValid = isValid && amountValid;

    // Hour validation: required, integer, 0 <= hour <= 23
    const hourValid = validateInput('hour', hourVal, (val) => {
      if (val === '') return 'Hour is required (0–23).';
      const num = Number(val);
      if (!Number.isInteger(num)) return 'Hour must be a whole integer.';
      if (num < 0 || num > 23) return 'Hour must be between 0 and 23.';
      return null;
    });
    if (!hourValid && !firstInvalidInput) firstInvalidInput = document.getElementById('hour');
    isValid = isValid && hourValid;

    // Distance validation: required, numeric, >= 0
    const distanceValid = validateInput('distance', distanceVal, (val) => {
      if (val === '') return 'Distance is required.';
      const num = Number(val);
      if (Number.isNaN(num) || !Number.isFinite(num)) return 'Enter a valid distance in km.';
      if (num < 0) return 'Distance cannot be negative.';
      if (num > 40000) return 'Distance exceeds Earth circumference.';
      return null;
    });
    if (!distanceValid && !firstInvalidInput) firstInvalidInput = document.getElementById('distance');
    isValid = isValid && distanceValid;

    // Frequency validation: required, integer, >= 0
    const freqValid = validateInput('frequency', frequencyVal, (val) => {
      if (val === '') return 'Frequency is required.';
      const num = Number(val);
      if (!Number.isInteger(num)) return 'Frequency must be an integer count.';
      if (num < 0) return 'Frequency cannot be negative.';
      if (num > 1000) return 'Frequency count is unrealistically high.';
      return null;
    });
    if (!freqValid && !firstInvalidInput) firstInvalidInput = document.getElementById('frequency');
    isValid = isValid && freqValid;

    if (!isValid && firstInvalidInput) {
      firstInvalidInput.focus();
    }

    return isValid;
  }

  // Clear validation error on field input
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
  // RUN DETECTION & UPDATE UI
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
    const { fraudProbability, isFraud } = runInference(rawFeatures);
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

    // Clear any previous validation errors
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
  // NAVIGATION (Landing vs Detection View)
  // ───────────────────────────────────────────────────────────────────
  function showDetection() {
    const landing = document.getElementById('landing');
    const detection = document.getElementById('detection');
    if (landing) landing.style.display = 'none';
    if (detection) {
      detection.style.display = 'block';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  function showLanding() {
    const detection = document.getElementById('detection');
    const landing = document.getElementById('landing');
    if (detection) detection.style.display = 'none';
    if (landing) {
      landing.style.display = 'flex';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
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
    loadModel,
    runDetection,
    loadPreset,
    setToggle,
    showDetection,
    showLanding,
    toggleTheme,
    runInference,
    standardize,
    validateAllInputs,
  };

  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    attachFieldListeners();
    loadModel();

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
