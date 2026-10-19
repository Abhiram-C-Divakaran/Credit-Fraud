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

  return {
    VALIDATION_RULES,
    validateField,
    validateTransaction,
    validateModelSchema,
    standardize,
    predictTree,
    runInference,
  };
}));
