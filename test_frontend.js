/**
 * Node.js Test Suite for Client-Side Inference and Validation Logic
 * =================================================================
 * Runs automated tests on model.json loading, tree traversal, standardization,
 * and form input validation edge cases.
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');

// 1. Verify model.json exists and parses
const modelPath = path.join(__dirname, 'model.json');
assert.ok(fs.existsSync(modelPath), 'model.json must exist');
const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));

console.log('✓ [1/5] model.json loaded successfully.');
assert.strictEqual(model.schema_version, '2.0.0', 'Schema version must be 2.0.0');
assert.strictEqual(model.n_trees, 20, 'Should have exactly 20 trees');
assert.strictEqual(model.trees.length, 20, 'Trees array length must be 20');
assert.strictEqual(model.metadata.is_synthetic, true, 'is_synthetic flag must be true');

// 2. Client-side inference algorithm (matching app.js)
function predictTree(node, scaledFeatures) {
  if (node.leaf) return node.prob;
  return scaledFeatures[node.feature] <= node.threshold
    ? predictTree(node.left, scaledFeatures)
    : predictTree(node.right, scaledFeatures);
}

function standardize(rawFeatures, scaler) {
  const { mean, scale } = scaler;
  return rawFeatures.map((v, i) => (v - mean[i]) / scale[i]);
}

function predict(rawFeatures) {
  const scaled = standardize(rawFeatures, model.scaler);
  const probs = model.trees.map((t) => predictTree(t, scaled));
  const avg = probs.reduce((a, b) => a + b, 0) / probs.length;
  return {
    fraudProbability: avg,
    isFraud: avg >= (model.metadata.decision_threshold || 0.5),
  };
}

console.log('✓ [2/5] Tree traversal and standardization functions defined.');

// 3. Test presets
const normalTxn = [65.0, 14, 12.0, 2, 0, 0];
const resNormal = predict(normalTxn);
console.log(`      • Normal Preset: prob=${(resNormal.fraudProbability * 100).toFixed(1)}%, isFraud=${resNormal.isFraud}`);
assert.ok(resNormal.fraudProbability >= 0.0 && resNormal.fraudProbability <= 1.0);
assert.strictEqual(resNormal.isFraud, false, 'Normal transaction must be classified as safe');

const suspiciousTxn = [3200.0, 2, 350.0, 9, 1, 1];
const resSuspicious = predict(suspiciousTxn);
console.log(`      • Suspicious Preset: prob=${(resSuspicious.fraudProbability * 100).toFixed(1)}%, isFraud=${resSuspicious.isFraud}`);
assert.ok(resSuspicious.fraudProbability >= 0.0 && resSuspicious.fraudProbability <= 1.0);
assert.strictEqual(resSuspicious.isFraud, true, 'Suspicious transaction must be classified as fraud');

console.log('✓ [3/5] Inference test on normal and suspicious presets passed.');

// 4. Test input validation logic
function validateAmount(val) {
  if (val === '') return 'Transaction amount is required.';
  const num = Number(val);
  if (Number.isNaN(num) || !Number.isFinite(num)) return 'Enter a valid numeric amount.';
  if (num < 0) return 'Amount cannot be negative.';
  if (num > 1000000) return 'Amount exceeds the demo maximum ($1,000,000).';
  return null;
}

function validateHour(val) {
  if (val === '') return 'Hour is required (0–23).';
  const num = Number(val);
  if (!Number.isInteger(num)) return 'Hour must be a whole integer.';
  if (num < 0 || num > 23) return 'Hour must be between 0 and 23.';
  return null;
}

function validateDistance(val) {
  if (val === '') return 'Distance is required.';
  const num = Number(val);
  if (Number.isNaN(num) || !Number.isFinite(num)) return 'Enter a valid distance in km.';
  if (num < 0) return 'Distance cannot be negative.';
  return null;
}

function validateFrequency(val) {
  if (val === '') return 'Frequency is required.';
  const num = Number(val);
  if (!Number.isInteger(num)) return 'Frequency must be an integer count.';
  if (num < 0) return 'Frequency cannot be negative.';
  return null;
}

// Amount tests
assert.ok(validateAmount(''), 'Empty amount should fail');
assert.ok(validateAmount('-10'), 'Negative amount should fail');
assert.ok(validateAmount('abc'), 'Non-numeric amount should fail');
assert.strictEqual(validateAmount('150.50'), null, 'Valid amount should pass');
assert.strictEqual(validateAmount('0'), null, 'Zero amount should pass');

// Hour tests
assert.ok(validateHour(''), 'Empty hour should fail');
assert.ok(validateHour('-1'), 'Negative hour should fail');
assert.ok(validateHour('24'), 'Hour 24 should fail');
assert.ok(validateHour('12.5'), 'Fractional hour should fail');
assert.strictEqual(validateHour('0'), null, 'Hour 0 should pass');
assert.strictEqual(validateHour('23'), null, 'Hour 23 should pass');

// Distance tests
assert.ok(validateDistance(''), 'Empty distance should fail');
assert.ok(validateDistance('-5'), 'Negative distance should fail');
assert.strictEqual(validateDistance('0'), null, 'Zero distance should pass');
assert.strictEqual(validateDistance('25.4'), null, 'Decimal distance should pass');

// Frequency tests
assert.ok(validateFrequency(''), 'Empty frequency should fail');
assert.ok(validateFrequency('-1'), 'Negative frequency should fail');
assert.ok(validateFrequency('2.5'), 'Fractional frequency should fail');
assert.strictEqual(validateFrequency('0'), null, 'Zero frequency should pass');
assert.strictEqual(validateFrequency('5'), null, 'Integer frequency should pass');

console.log('✓ [4/5] All input validation edge-case tests passed.');

// 5. Verify feature specifications and schema integrity
assert.strictEqual(model.feature_names.length, 6, 'Must have 6 feature names');
assert.strictEqual(model.feature_importances.length, 6, 'Must have 6 feature importances');
const totalPct = model.feature_importances.reduce((acc, f) => acc + f.percentage, 0);
assert.ok(totalPct >= 99 && totalPct <= 101, `Feature importance total should be ~100% (got ${totalPct}%)`);

console.log('✓ [5/5] Schema and feature specification tests passed.');
console.log('\n======================================================');
console.log('  All Frontend Inference & Validation Tests Passed!');
console.log('======================================================');
