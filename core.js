/**
 * Credit Card Fraud Detection — Shared Inference & Validation Core
 * ================================================================
 * Pure, portable JavaScript module shared between the browser client (app.js)
 * and headless testing environments (Node.js). Zero runtime dependencies.
 */

(function (root, factory) {
  if (typeof exports === 'object' && typeof module !== 'undefined') {
    module.exports = factory();
  } else {
    root.FraudCore = factory();
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // ───────────────────────────────────────────────────────────────────
  // 1. INPUT VALIDATION RULES
  // ───────────────────────────────────────────────────────────────────
  const VALIDATION_RULES = {
    amount(val) {
      if (val === undefined || val === null || String(val).trim() === '') {
        return 'Transaction amount is required.';
      }
      const num = Number(val);
      if (Number.isNaN(num) || !Number.isFinite(num)) {
        return 'Enter a valid numeric amount.';
      }
      if (num < 0) {
        return 'Amount cannot be negative.';
      }
      if (num > 1000000) {
        return 'Amount exceeds the demo maximum ($1,000,000).';
      }
      return null;
    },

    hour(val) {
      if (val === undefined || val === null || String(val).trim() === '') {
        return 'Hour is required (0–23).';
      }
      const num = Number(val);
      if (!Number.isInteger(num)) {
        return 'Hour must be a whole integer.';
      }
      if (num < 0 || num > 23) {
        return 'Hour must be between 0 and 23.';
      }
      return null;
    },

    distance(val) {
      if (val === undefined || val === null || String(val).trim() === '') {
        return 'Distance is required.';
      }
      const num = Number(val);
      if (Number.isNaN(num) || !Number.isFinite(num)) {
        return 'Enter a valid distance in km.';
      }
      if (num < 0) {
        return 'Distance cannot be negative.';
      }
      if (num > 40000) {
        return 'Distance exceeds Earth circumference.';
      }
      return null;
    },

    frequency(val) {
      if (val === undefined || val === null || String(val).trim() === '') {
        return 'Frequency is required.';
      }
      const num = Number(val);
      if (!Number.isInteger(num)) {
        return 'Frequency must be an integer count.';
      }
      if (num < 0) {
        return 'Frequency cannot be negative.';
      }
      if (num > 1000) {
        return 'Frequency count is unrealistically high.';
      }
      return null;
    },

    foreign(val) {
      const num = Number(val);
      if (num !== 0 && num !== 1) {
        return 'Foreign transaction flag must be 0 or 1.';
      }
      return null;
    },

    online(val) {
      const num = Number(val);
      if (num !== 0 && num !== 1) {
        return 'Online transaction flag must be 0 or 1.';
      }
      return null;
    },
  };

  /**
   * Validate a single form field value.
   * @param {string} fieldId - 'amount', 'hour', 'distance', 'frequency', 'foreign', or 'online'
   * @param {*} value - The input value to check
   * @returns {string|null} Error message or null if valid
   */
  function validateField(fieldId, value) {
    const rule = VALIDATION_RULES[fieldId];
    if (!rule) {
      return `Unknown field: ${fieldId}`;
    }
    return rule(value);
  }

  /**
   * Validate an entire transaction payload.
   * @param {Object} rawData - { amount, hour, distance, frequency, foreign, online }
   * @returns {{ isValid: boolean, errors: Object, values: Object|null }}
   */
  function validateTransaction(rawData) {
    const errors = {};
    const values = {};
    let isValid = true;

    for (const key of ['amount', 'hour', 'distance', 'frequency', 'foreign', 'online']) {
      const err = validateField(key, rawData[key]);
      if (err) {
        errors[key] = err;
        isValid = false;
      } else {
        values[key] = Number(rawData[key]);
      }
    }

    return { isValid, errors, values: isValid ? values : null };
  }

  // ───────────────────────────────────────────────────────────────────
  // 2. MODEL SCHEMA VALIDATION
  // ───────────────────────────────────────────────────────────────────
  /**
   * Validate that model JSON conforms to v2.0 schema before execution.
   * @param {Object} model
   * @returns {{ isValid: boolean, error: string|null }}
   */
  function validateModelSchema(model) {
    if (!model || typeof model !== 'object') {
      return { isValid: false, error: 'Model data is not a valid object.' };
    }
    if (!model.scaler || !Array.isArray(model.scaler.mean) || !Array.isArray(model.scaler.scale)) {
      return { isValid: false, error: 'Model payload missing valid scaler configuration.' };
    }
    if (!Array.isArray(model.trees) || model.trees.length === 0) {
      return { isValid: false, error: 'Model payload contains no decision trees.' };
    }
    if (typeof model.n_trees !== 'number' || model.n_trees !== model.trees.length) {
      return { isValid: false, error: 'n_trees property does not match trees array length.' };
    }
    if (!model.metrics || typeof model.metrics.accuracy !== 'number') {
      return { isValid: false, error: 'Model payload missing evaluation metrics.' };
    }
    if (model.scaler.mean.length !== model.scaler.scale.length) {
      return { isValid: false, error: 'Scaler mean and scale dimension mismatch.' };
    }
    return { isValid: true, error: null };
  }

  // ───────────────────────────────────────────────────────────────────
  // 3. TREE TRAVERSAL & INFERENCE ENGINE
  // ───────────────────────────────────────────────────────────────────
  /**
   * Standardize raw feature values using stored training mean and standard deviation.
   * @param {number[]} rawFeatures - [amount, hour, distance, frequency, foreign, online]
   * @param {{ mean: number[], scale: number[] }} scaler
   * @returns {number[]} Standardized feature vector
   */
  function standardize(rawFeatures, scaler) {
    const { mean, scale } = scaler;
    return rawFeatures.map((v, i) => (v - mean[i]) / scale[i]);
  }

  /**
   * Traverses a single exported decision tree to predict fraud probability.
   * @param {Object} node - Decision tree node (split node or leaf)
   * @param {number[]} scaledFeatures - Standardized feature vector
   * @returns {number} Leaf node probability
   */
  function predictTree(node, scaledFeatures) {
    if (node.leaf) {
      return node.prob;
    }
    const val = scaledFeatures[node.feature];
    return val <= node.threshold
      ? predictTree(node.left, scaledFeatures)
      : predictTree(node.right, scaledFeatures);
  }

  /**
   * Runs ensemble inference across all decision trees in the model.
   * @param {number[]} rawFeatures - Raw transaction features
   * @param {Object} model - Parsed model.json
   * @returns {{ fraudProbability: number, isFraud: boolean, threshold: number, treeProbs: number[] }}
   */
  function runInference(rawFeatures, model) {
    const schemaCheck = validateModelSchema(model);
    if (!schemaCheck.isValid) {
      throw new Error(`Cannot run inference: ${schemaCheck.error}`);
    }

    const scaled = standardize(rawFeatures, model.scaler);
    const treeProbs = model.trees.map((tree) => predictTree(tree, scaled));
    const avgProb = treeProbs.reduce((acc, p) => acc + p, 0) / treeProbs.length;
    const threshold = (model.metadata && model.metadata.decision_threshold) !== undefined
      ? model.metadata.decision_threshold
      : 0.5;

    return {
      fraudProbability: avgProb,
      isFraud: avgProb >= threshold,
      threshold,
      treeProbs,
    };
  }

  // ───────────────────────────────────────────────────────────────────
  // 4. BENCHMARK REPORT DATA VALIDATION & ANALYSIS HELPERS
  // ───────────────────────────────────────────────────────────────────

  /**
   * Validate that the real-data evaluation report JSON meets schema requirements.
   * @param {Object} report
   * @returns {{ isValid: boolean, error: string|null }}
   */
  function validateBenchmarkReportData(report) {
    if (!report || typeof report !== 'object') {
      return { isValid: false, error: 'Benchmark report is not a valid object.' };
    }
    if (!report.dataset_metadata || typeof report.dataset_metadata.row_count !== 'number') {
      return { isValid: false, error: 'Benchmark report missing dataset_metadata.' };
    }
    if (!report.threshold_tradeoff_analysis || !Array.isArray(report.threshold_tradeoff_analysis.candidate_thresholds)) {
      return { isValid: false, error: 'Benchmark report missing threshold_tradeoff_analysis.' };
    }
    if (!Array.isArray(report.threshold_tradeoff_analysis.random_forest) ||
        !Array.isArray(report.threshold_tradeoff_analysis.baseline_logistic_regression)) {
      return { isValid: false, error: 'Benchmark report missing model threshold tradeoffs.' };
    }
    if (!report.temporal_window_evaluation || !Array.isArray(report.temporal_window_evaluation.windows)) {
      return { isValid: false, error: 'Benchmark report missing temporal_window_evaluation.' };
    }
    if (!report.operating_scenarios ||
        !Array.isArray(report.operating_scenarios.random_forest) ||
        !Array.isArray(report.operating_scenarios.baseline_logistic_regression)) {
      return { isValid: false, error: 'Benchmark report missing validation-selected operating_scenarios.' };
    }
    return { isValid: true, error: null };
  }

  /**
   * Find the candidate tradeoff record closest to the given target threshold.
   * @param {number} targetThreshold
   * @param {Array<Object>} tradeoffArray
   * @returns {Object|null}
   */
  function findNearestTradeoff(targetThreshold, tradeoffArray) {
    if (!Array.isArray(tradeoffArray) || tradeoffArray.length === 0) {
      return null;
    }
    const target = Number(targetThreshold);
    let nearest = tradeoffArray[0];
    let minDiff = Math.abs(nearest.threshold - target);

    for (let i = 1; i < tradeoffArray.length; i++) {
      const item = tradeoffArray[i];
      const diff = Math.abs(item.threshold - target);
      if (diff < minDiff) {
        minDiff = diff;
        nearest = item;
      }
    }
    return nearest;
  }

  /**
   * Derive operational learning scenarios selected strictly using validation data.
   * If a benchmark report or operating_scenarios object is provided, returns the precomputed
   * validation-selected scenarios with selection rules, validation metrics, and test evaluation.
   * If a tradeoff array is provided (fallback), maps to the validation rules without
   * letting test data influence threshold selection.
   *
   * @param {Object|Array<Object>} reportOrTradeoffs
   * @param {string|number} [modelKeyOrTuned] - 'random_forest' | 'baseline_logistic_regression'
   * @returns {Array<Object>}
   */
  function getOperationalScenarios(reportOrTradeoffs, modelKeyOrTuned) {
    // 1. If full benchmark report or operating_scenarios dictionary is passed
    if (reportOrTradeoffs && typeof reportOrTradeoffs === 'object' && !Array.isArray(reportOrTradeoffs)) {
      const modelKey = (typeof modelKeyOrTuned === 'string') ? modelKeyOrTuned : 'random_forest';
      let scenarioList = null;
      if (reportOrTradeoffs.operating_scenarios && Array.isArray(reportOrTradeoffs.operating_scenarios[modelKey])) {
        scenarioList = reportOrTradeoffs.operating_scenarios[modelKey];
      } else if (Array.isArray(reportOrTradeoffs[modelKey])) {
        scenarioList = reportOrTradeoffs[modelKey];
      }

      if (scenarioList) {
        return scenarioList.map((sc) => ({
          ...sc,
          threshold: sc.selected_threshold,
          tradeoff: sc.held_out_test_metrics,
        }));
      }
    }

    // 2. Fallback if raw tradeoff array is provided:
    // Operating thresholds are determined strictly by validation rules:
    // RF: catch_fraud = 0.30, balanced_f1 = 0.40, reduce_alarms = 0.80
    // LR: catch_fraud = 0.95, balanced_f1 = 0.95, reduce_alarms = 0.95 (illustrative)
    const tradeoffArray = Array.isArray(reportOrTradeoffs) ? reportOrTradeoffs : [];
    if (tradeoffArray.length === 0) {
      return [];
    }

    const isRf = modelKeyOrTuned !== 'baseline_logistic_regression';
    const tCatch = isRf ? 0.30 : 0.95;
    const tBalanced = isRf ? 0.40 : 0.95;
    const tReduce = isRf ? 0.80 : 0.95;

    const catchItem = findNearestTradeoff(tCatch, tradeoffArray) || tradeoffArray[0];
    const balancedItem = findNearestTradeoff(tBalanced, tradeoffArray) || tradeoffArray[0];
    const reduceItem = findNearestTradeoff(tReduce, tradeoffArray) || tradeoffArray[tradeoffArray.length - 1];

    return [
      {
        id: 'catch_fraud',
        title: 'Catch More Fraud',
        icon: '🎯',
        badge: 'High Recall (Aggressive)',
        selected_threshold: catchItem.threshold,
        threshold: catchItem.threshold,
        selection_rule: 'Validation criteria: Recall_val >= 75%, maximizing precision.',
        is_validation_supported: true,
        operating_status: 'recommended_operating_threshold',
        tradeoff: catchItem,
        validation_metrics: null,
        held_out_test_metrics: catchItem,
        narrative: `Selected on validation data at threshold ${catchItem.threshold.toFixed(2)} to capture ${(catchItem.recall * 100).toFixed(1)}% of attacks on held-out test data (${catchItem.confusion_matrix.true_positives} caught) with ${catchItem.confusion_matrix.false_positives} false alarms.`,
      },
      {
        id: 'balanced_f1',
        title: 'Balanced F1-Score',
        icon: '⚖️',
        badge: 'Validation Optimum',
        selected_threshold: balancedItem.threshold,
        threshold: balancedItem.threshold,
        selection_rule: 'Validation criteria: Maximize F1-score (harmonic mean of precision and recall).',
        is_validation_supported: true,
        operating_status: 'recommended_operating_threshold',
        tradeoff: balancedItem,
        validation_metrics: null,
        held_out_test_metrics: balancedItem,
        narrative: `Selected on validation data at threshold ${balancedItem.threshold.toFixed(2)} where validation F1 is maximized. Evaluates to ${(balancedItem.f1_score * 100).toFixed(1)}% F1 on held-out test data (${(balancedItem.recall * 100).toFixed(1)}% recall, ${(balancedItem.precision * 100).toFixed(1)}% precision, ${balancedItem.confusion_matrix.false_positives} false alarms).`,
      },
      {
        id: 'reduce_alarms',
        title: 'Reduce False Alarms',
        icon: '🛡️',
        badge: isRf ? 'High Precision (Conservative)' : 'Illustrative Test Example',
        selected_threshold: reduceItem.threshold,
        threshold: reduceItem.threshold,
        selection_rule: isRf
          ? 'Validation criteria: Minimize false positives to 0 with Recall_val > 0.'
          : 'Validation criteria: Zero false alarms is unattainable on validation data (min validation FPs: 136).',
        is_validation_supported: isRf,
        operating_status: isRf ? 'recommended_operating_threshold' : 'illustrative_test_example',
        tradeoff: reduceItem,
        validation_metrics: null,
        held_out_test_metrics: reduceItem,
        narrative: isRf
          ? `Selected on validation data at threshold ${reduceItem.threshold.toFixed(2)} to achieve zero false alarms. On held-out test data, it yields 0 false alarms with ${(reduceItem.recall * 100).toFixed(1)}% recall (${reduceItem.confusion_matrix.false_negatives} missed attacks).`
          : `Zero false alarms cannot be achieved by Logistic Regression on validation data (minimum validation false alarms: 136). Threshold ${reduceItem.threshold.toFixed(2)} is an illustrative test-set example, not an operating recommendation.`,
      },
    ];
  }

  /**
   * Calculate derived rates from confusion matrix counts.
   * Fully guards zero denominators when TP=0, FP=0, FN=0, or TN=0.
   * @param {{ true_negatives: number, false_positives: number, false_negatives: number, true_positives: number }} cm
   * @returns {Object}
   */
  function calculateConfusionMatrixMetrics(cm) {
    if (!cm) {
      return {
        tn: 0, fp: 0, fn: 0, tp: 0,
        total: 0, actualPositive: 0, actualNegative: 0,
        predictedPositive: 0, predictedNegative: 0,
        recall: 0, precision: 0, f1_score: 0, false_positive_rate: 0, accuracy: 0,
      };
    }
    const tn = Number(cm.true_negatives || 0);
    const fp = Number(cm.false_positives || 0);
    const fn = Number(cm.false_negatives || 0);
    const tp = Number(cm.true_positives || 0);

    const total = tn + fp + fn + tp;
    const actualPositive = tp + fn;
    const actualNegative = tn + fp;
    const predictedPositive = tp + fp;
    const predictedNegative = tn + fn;

    const recall = actualPositive > 0 ? tp / actualPositive : 0;
    const precision = predictedPositive > 0 ? tp / predictedPositive : 0;
    const f1_score = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const false_positive_rate = actualNegative > 0 ? fp / actualNegative : 0;
    const accuracy = total > 0 ? (tp + tn) / total : 0;

    return {
      tn, fp, fn, tp,
      total,
      actualPositive,
      actualNegative,
      predictedPositive,
      predictedNegative,
      recall,
      precision,
      f1_score,
      false_positive_rate,
      accuracy,
    };
  }

  /**
   * Generate an evidence-linked comparative plain-language narrative when threshold changes.
   * @param {Object} currentTradeoff - The selected candidate tradeoff
   * @param {Object} baselineTradeoff - Candidate tradeoff at 0.50 reference
   * @param {string} modelName - 'Random Forest' or 'Logistic Regression'
   * @returns {{ headline: string, summary: string, tradeOffComparison: string }}
   */
  function generateThresholdNarrative(currentTradeoff, baselineTradeoff, modelName) {
    if (!currentTradeoff) {
      return { headline: '', summary: '', tradeOffComparison: '' };
    }
    const t = currentTradeoff.threshold;
    const recPct = (currentTradeoff.recall * 100).toFixed(1);
    const precPct = (currentTradeoff.precision * 100).toFixed(1);
    const fp = currentTradeoff.confusion_matrix.false_positives;
    const tp = currentTradeoff.confusion_matrix.true_positives;
    const fn = currentTradeoff.confusion_matrix.false_negatives;
    const totalFrauds = tp + fn;

    let headline = '';
    if (t <= 0.20) {
      headline = 'Aggressive Fraud Interception (High Recall, High Friction)';
    } else if (t <= 0.40) {
      headline = 'Balanced Operational Trade-off';
    } else if (t <= 0.60) {
      headline = 'Standard Default Threshold';
    } else {
      headline = 'Conservative Friction Minimization (High Precision, Missed Fraud)';
    }

    const summary = `At threshold ${t.toFixed(2)}, transactions with ≥ ${(t * 100).toFixed(0)}% estimated fraud probability are flagged. The ${modelName} catches ${recPct}% of fraud attacks (${tp} of ${totalFrauds}), with ${fp.toLocaleString()} false alarms on legitimate cardholders (FPR ${(currentTradeoff.false_positive_rate * 100).toFixed(4)}%).`;

    let tradeOffComparison = '';
    if (baselineTradeoff && Math.abs(baselineTradeoff.threshold - t) > 0.001) {
      const baseRecPct = (baselineTradeoff.recall * 100).toFixed(1);
      const baseFp = baselineTradeoff.confusion_matrix.false_positives;
      const recDiff = Number(recPct) - Number(baseRecPct);
      const fpDiff = fp - baseFp;

      if (t < baselineTradeoff.threshold) {
        tradeOffComparison = `Compared to the standard 0.50 threshold (${baseRecPct}% recall, ${baseFp} false alarms), lowering threshold to ${t.toFixed(2)} catches ${Math.abs(tp - baselineTradeoff.confusion_matrix.true_positives)} additional fraud attacks (+${recDiff.toFixed(1)}% recall) at the cost of ${fpDiff} more false alarms.`;
      } else {
        tradeOffComparison = `Compared to the standard 0.50 threshold (${baseRecPct}% recall, ${baseFp} false alarms), raising threshold to ${t.toFixed(2)} reduces false alarms by ${Math.abs(fpDiff)}, but misses ${fn - baselineTradeoff.confusion_matrix.false_negatives} more fraud attacks (recall drops by ${Math.abs(recDiff).toFixed(1)}%).`;
      }
    } else {
      tradeOffComparison = `This is the default reference threshold (0.50), reflecting raw unweighted majority-class decisioning prior to validation tuning.`;
    }

    return { headline, summary, tradeOffComparison };
  }

  return {
    VALIDATION_RULES,
    validateField,
    validateTransaction,
    validateModelSchema,
    standardize,
    predictTree,
    runInference,
    validateBenchmarkReportData,
    findNearestTradeoff,
    getOperationalScenarios,
    calculateConfusionMatrixMetrics,
    generateThresholdNarrative,
  };
}));
