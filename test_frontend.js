/**
 * Node.js Test Suite for Client-Side Inference & Validation Logic
 * ===============================================================
 * Imports the shared FraudCore module (core.js) directly to eliminate code duplication
 * and ensure tests exercise the exact code executed in the browser.
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');
const FraudCore = require('./core.js');

const {
  validateField,
  validateTransaction,
  validateModelSchema,
  standardize,
  predictTree,
  runInference,
} = FraudCore;

console.log('======================================================');
console.log('  Running Frontend Tests using shared core.js');
console.log('======================================================');

// ─────────────────────────────────────────────────────────────────────
// 1. Model JSON Loading & Schema Validation
// ─────────────────────────────────────────────────────────────────────
const modelPath = path.join(__dirname, 'model.json');
assert.ok(fs.existsSync(modelPath), 'model.json must exist in root directory');
const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));

// Test valid model against shared validator
const schemaCheck = validateModelSchema(model);
assert.strictEqual(schemaCheck.isValid, true, `Real model.json should pass schema validation: ${schemaCheck.error}`);
assert.strictEqual(schemaCheck.error, null);

// Test schema error handling for corrupted/malformed models
const invalidModels = [
  { model: null, expectedErr: 'not a valid object' },
  { model: {}, expectedErr: 'missing valid scaler' },
  { model: { scaler: { mean: [1], scale: [1] } }, expectedErr: 'contains no decision trees' },
  { model: { scaler: { mean: [1], scale: [1] }, trees: [{ leaf: true, prob: 0.5 }], n_trees: 5 }, expectedErr: 'n_trees property does not match' },
  { model: { scaler: { mean: [1], scale: [1] }, trees: [{ leaf: true, prob: 0.5 }], n_trees: 1 }, expectedErr: 'missing evaluation metrics' },
  { model: { scaler: { mean: [1, 2], scale: [1] }, trees: [{ leaf: true, prob: 0.5 }], n_trees: 1, metrics: { accuracy: 0.95 } }, expectedErr: 'dimension mismatch' },
];

invalidModels.forEach(({ model: badModel, expectedErr }, idx) => {
  const res = validateModelSchema(badModel);
  assert.strictEqual(res.isValid, false, `Bad model ${idx} should fail validation`);
  assert.ok(res.error.toLowerCase().includes(expectedErr.toLowerCase()), `Error should mention "${expectedErr}", got: "${res.error}"`);
});

console.log('✓ [1/5] Model schema validation tests passed (valid + malformed test cases).');

// ─────────────────────────────────────────────────────────────────────
// 2. Field-Level Validation Tests
// ─────────────────────────────────────────────────────────────────────
// Amount validation
assert.ok(validateField('amount', ''), 'Empty amount should fail');
assert.ok(validateField('amount', null), 'Null amount should fail');
assert.ok(validateField('amount', 'abc'), 'String amount should fail');
assert.ok(validateField('amount', -1), 'Negative amount should fail');
assert.ok(validateField('amount', 1000001), 'Amount > $1M should fail');
assert.strictEqual(validateField('amount', 0), null, 'Amount 0 should pass');
assert.strictEqual(validateField('amount', 85.5), null, 'Valid amount 85.5 should pass');
assert.strictEqual(validateField('amount', '150.00'), null, 'Numeric string 150.00 should pass');

// Hour validation
assert.ok(validateField('hour', ''), 'Empty hour should fail');
assert.ok(validateField('hour', -1), 'Negative hour should fail');
assert.ok(validateField('hour', 24), 'Hour 24 should fail');
assert.ok(validateField('hour', 14.5), 'Fractional hour should fail');
assert.strictEqual(validateField('hour', 0), null, 'Hour 0 should pass');
assert.strictEqual(validateField('hour', 23), null, 'Hour 23 should pass');
assert.strictEqual(validateField('hour', '12'), null, 'String integer 12 should pass');

// Distance validation
assert.ok(validateField('distance', ''), 'Empty distance should fail');
assert.ok(validateField('distance', -5), 'Negative distance should fail');
assert.ok(validateField('distance', 45000), 'Distance > Earth circumference should fail');
assert.strictEqual(validateField('distance', 0), null, 'Distance 0 should pass');
assert.strictEqual(validateField('distance', 120.4), null, 'Distance 120.4 should pass');

// Frequency validation
assert.ok(validateField('frequency', ''), 'Empty frequency should fail');
assert.ok(validateField('frequency', -1), 'Negative frequency should fail');
assert.ok(validateField('frequency', 2.5), 'Fractional frequency should fail');
assert.strictEqual(validateField('frequency', 0), null, 'Frequency 0 should pass');
assert.strictEqual(validateField('frequency', 8), null, 'Frequency 8 should pass');

// Foreign / Online binary flag validation
assert.strictEqual(validateField('foreign', 0), null);
assert.strictEqual(validateField('foreign', 1), null);
assert.ok(validateField('foreign', 2));
assert.strictEqual(validateField('online', 0), null);
assert.strictEqual(validateField('online', 1), null);
assert.ok(validateField('online', -1));

console.log('✓ [2/5] Field-level input validation tests passed.');

// ─────────────────────────────────────────────────────────────────────
// 3. Full Transaction Payload Validation Tests
// ─────────────────────────────────────────────────────────────────────
const validPayload = {
  amount: '120.50',
  hour: '14',
  distance: '25',
  frequency: '3',
  foreign: 0,
  online: 1,
};
const validResult = validateTransaction(validPayload);
assert.strictEqual(validResult.isValid, true);
assert.strictEqual(Object.keys(validResult.errors).length, 0);
assert.strictEqual(validResult.values.amount, 120.50);
assert.strictEqual(validResult.values.hour, 14);

const invalidPayload = {
  amount: '-10',
  hour: '26',
  distance: '',
  frequency: '2.5',
  foreign: 5,
  online: 0,
};
const invalidResult = validateTransaction(invalidPayload);
assert.strictEqual(invalidResult.isValid, false);
assert.ok(invalidResult.errors.amount);
assert.ok(invalidResult.errors.hour);
assert.ok(invalidResult.errors.distance);
assert.ok(invalidResult.errors.frequency);
assert.ok(invalidResult.errors.foreign);
assert.strictEqual(invalidResult.values, null);

console.log('✓ [3/5] Full transaction payload validation tests passed.');

// ─────────────────────────────────────────────────────────────────────
// 4. Standardization & Tree Traversal Parity Tests
// ─────────────────────────────────────────────────────────────────────
const testScaler = { mean: [100, 12], scale: [20, 4] };
const scaled = standardize([120, 8], testScaler);
assert.strictEqual(scaled[0], 1.0);  // (120 - 100) / 20 = 1.0
assert.strictEqual(scaled[1], -1.0); // (8 - 12) / 4 = -1.0

const mockTree = {
  feature: 0,
  threshold: 0.5,
  left: { leaf: true, prob: 0.1 },
  right: { leaf: true, prob: 0.9 },
};
assert.strictEqual(predictTree(mockTree, [0.2]), 0.1);
assert.strictEqual(predictTree(mockTree, [0.8]), 0.9);

console.log('✓ [4/5] Standardization and tree traversal unit tests passed.');

// ─────────────────────────────────────────────────────────────────────
// 5. End-to-End Inference using Real model.json and Presets
// ─────────────────────────────────────────────────────────────────────
// Normal Transaction Preset: domestic, daytime, modest spend
const normalTxn = [65.0, 14, 12.0, 2, 0, 0];
const resNormal = runInference(normalTxn, model);
console.log(`      • Normal Preset: prob=${(resNormal.fraudProbability * 100).toFixed(1)}%, isFraud=${resNormal.isFraud}`);
assert.ok(resNormal.fraudProbability >= 0.0 && resNormal.fraudProbability <= 1.0);
assert.strictEqual(resNormal.isFraud, false, 'Normal transaction must be classified as safe');

// Suspicious Transaction Preset: foreign, online, night, large spend, rapid bursts
const suspiciousTxn = [3200.0, 2, 350.0, 9, 1, 1];
const resSuspicious = runInference(suspiciousTxn, model);
console.log(`      • Suspicious Preset: prob=${(resSuspicious.fraudProbability * 100).toFixed(1)}%, isFraud=${resSuspicious.isFraud}`);
assert.ok(resSuspicious.fraudProbability >= 0.0 && resSuspicious.fraudProbability <= 1.0);
assert.strictEqual(resSuspicious.isFraud, true, 'Suspicious transaction must be classified as fraud');

// Attempting inference with invalid model should throw error
assert.throws(() => {
  runInference(normalTxn, {});
}, /Cannot run inference/);

console.log('✓ [5/10] Real model inference tests on normal and suspicious presets passed.');

// ─────────────────────────────────────────────────────────────────────
// 6. Benchmark Report Loading & Schema Validation
// ─────────────────────────────────────────────────────────────────────
const {
  validateBenchmarkReportData,
  findNearestTradeoff,
  getOperationalScenarios,
  calculateConfusionMatrixMetrics,
  generateThresholdNarrative,
} = FraudCore;

const reportPath = path.join(__dirname, 'reports', 'real_data_evaluation.json');
assert.ok(fs.existsSync(reportPath), 'reports/real_data_evaluation.json must exist');
const benchmarkReport = JSON.parse(fs.readFileSync(reportPath, 'utf8'));

const benchValid = validateBenchmarkReportData(benchmarkReport);
assert.strictEqual(benchValid.isValid, true, `Real benchmark report should be valid: ${benchValid.error}`);
assert.strictEqual(benchValid.error, null);

// Corrupted benchmark report cases
assert.strictEqual(validateBenchmarkReportData(null).isValid, false);
assert.strictEqual(validateBenchmarkReportData({}).isValid, false);
assert.strictEqual(validateBenchmarkReportData({ dataset_metadata: {} }).isValid, false);
assert.strictEqual(validateBenchmarkReportData({
  dataset_metadata: { row_count: 100 },
  threshold_tradeoff_analysis: {},
}).isValid, false);

console.log('✓ [6/10] Benchmark report schema validation passed (valid + corrupted).');

// ─────────────────────────────────────────────────────────────────────
// 7. Benchmark Tradeoff Nearest Neighbor Matching
// ─────────────────────────────────────────────────────────────────────
const rfTradeoffs = benchmarkReport.threshold_tradeoff_analysis.random_forest;
assert.ok(Array.isArray(rfTradeoffs) && rfTradeoffs.length > 0);

// Exact match
const matchExact = findNearestTradeoff(0.30, rfTradeoffs);
assert.strictEqual(matchExact.threshold, 0.30);
assert.strictEqual(matchExact.confusion_matrix.false_positives, 16);
assert.strictEqual(matchExact.confusion_matrix.true_positives, 57);

// Boundary and interpolation matching
const matchNearLow = findNearestTradeoff(0.04, rfTradeoffs);
assert.strictEqual(matchNearLow.threshold, 0.05);

const matchNearMid = findNearestTradeoff(0.38, rfTradeoffs);
assert.strictEqual(matchNearMid.threshold, 0.40);

const matchNearHigh = findNearestTradeoff(0.99, rfTradeoffs);
assert.strictEqual(matchNearHigh.threshold, 0.95);

// Empty array guard
assert.strictEqual(findNearestTradeoff(0.5, []), null);

console.log('✓ [7/10] Benchmark threshold nearest neighbor lookups passed.');

// ─────────────────────────────────────────────────────────────────────
// 8. Operational Scenarios Generation
// ─────────────────────────────────────────────────────────────────────
// 8. Operational Scenarios Generation & Validation Independence
// ─────────────────────────────────────────────────────────────────────
// Test 8a: Direct from benchmark report
const rfScenariosFromReport = getOperationalScenarios(benchmarkReport, 'random_forest');
assert.strictEqual(rfScenariosFromReport.length, 3);

const catchFraud = rfScenariosFromReport.find((s) => s.id === 'catch_fraud');
const balancedF1 = rfScenariosFromReport.find((s) => s.id === 'balanced_f1');
const reduceAlarms = rfScenariosFromReport.find((s) => s.id === 'reduce_alarms');

assert.ok(catchFraud && balancedF1 && reduceAlarms, 'All 3 operational scenarios must be generated');
assert.strictEqual(catchFraud.selected_threshold, 0.30);
assert.strictEqual(catchFraud.tradeoff.recall, 0.76);
assert.strictEqual(catchFraud.tradeoff.confusion_matrix.false_positives, 16);
assert.strictEqual(catchFraud.is_validation_supported, true);
assert.ok(catchFraud.selection_rule.includes('Recall_val >= 75%'));

assert.strictEqual(balancedF1.selected_threshold, 0.40);
assert.strictEqual(balancedF1.tradeoff.f1_score, 0.8175);
assert.strictEqual(balancedF1.is_validation_supported, true);
assert.ok(balancedF1.selection_rule.includes('Maximize F1-score'));

assert.strictEqual(reduceAlarms.selected_threshold, 0.80);
assert.strictEqual(reduceAlarms.tradeoff.confusion_matrix.false_positives, 0);
assert.strictEqual(reduceAlarms.is_validation_supported, true);

// Test 8b: Logistic Regression scenarios & illustrative unsupported flag
const lrScenarios = getOperationalScenarios(benchmarkReport, 'baseline_logistic_regression');
const lrReduce = lrScenarios.find((s) => s.id === 'reduce_alarms');
assert.strictEqual(lrReduce.is_validation_supported, false);
assert.strictEqual(lrReduce.operating_status, 'illustrative_test_example');

// Test 8c: Regression test — mutating test tradeoff metrics does NOT change scenario selected thresholds
const mutatedReport = JSON.parse(JSON.stringify(benchmarkReport));
mutatedReport.threshold_tradeoff_analysis.random_forest.forEach((row) => {
  row.recall = 0.01;
  row.precision = 0.01;
  row.f1_score = 0.01;
  row.confusion_matrix.false_positives = 9999;
});
const scenariosFromMutated = getOperationalScenarios(mutatedReport, 'random_forest');
assert.strictEqual(scenariosFromMutated.find((s) => s.id === 'catch_fraud').selected_threshold, 0.30);
assert.strictEqual(scenariosFromMutated.find((s) => s.id === 'balanced_f1').selected_threshold, 0.40);
assert.strictEqual(scenariosFromMutated.find((s) => s.id === 'reduce_alarms').selected_threshold, 0.80);

console.log('✓ [8/10] Operational scenario generation & validation data isolation passed.');

// ─────────────────────────────────────────────────────────────────────
// 9. Confusion Matrix Mathematical Identities & Zero Denominators
// ─────────────────────────────────────────────────────────────────────
const sampleCm = {
  true_negatives: 56881,
  false_positives: 6,
  false_negatives: 19,
  true_positives: 56,
};
const cmMetrics = calculateConfusionMatrixMetrics(sampleCm);
assert.strictEqual(cmMetrics.total, 56962);
assert.strictEqual(cmMetrics.actualPositive, 75);
assert.strictEqual(cmMetrics.actualNegative, 56887);
assert.strictEqual(cmMetrics.predictedPositive, 62);
assert.strictEqual(Number((cmMetrics.recall * 100).toFixed(2)), 74.67);
assert.strictEqual(Number((cmMetrics.precision * 100).toFixed(2)), 90.32);
assert.strictEqual(Number((cmMetrics.f1_score).toFixed(4)), 0.8175);
assert.ok(cmMetrics.false_positive_rate > 0 && cmMetrics.false_positive_rate < 0.001);

// Zero Denominator Case 1: All zeros confusion matrix
const zeroCm = calculateConfusionMatrixMetrics({ true_negatives: 0, false_positives: 0, false_negatives: 0, true_positives: 0 });
assert.strictEqual(zeroCm.total, 0);
assert.strictEqual(zeroCm.recall, 0);
assert.strictEqual(zeroCm.precision, 0);
assert.strictEqual(zeroCm.f1_score, 0);
assert.strictEqual(zeroCm.false_positive_rate, 0);
assert.strictEqual(zeroCm.accuracy, 0);

// Zero Denominator Case 2: Zero predicted positives (TP=0, FP=0)
const zeroPredPosCm = calculateConfusionMatrixMetrics({ true_negatives: 100, false_positives: 0, false_negatives: 10, true_positives: 0 });
assert.strictEqual(zeroPredPosCm.predictedPositive, 0);
assert.strictEqual(zeroPredPosCm.precision, 0);
assert.strictEqual(zeroPredPosCm.recall, 0);
assert.strictEqual(zeroPredPosCm.f1_score, 0);
assert.strictEqual(zeroPredPosCm.false_positive_rate, 0);
assert.strictEqual(zeroPredPosCm.accuracy, 100 / 110);

// Zero Denominator Case 3: Zero actual positives (TP=0, FN=0)
const zeroActualPosCm = calculateConfusionMatrixMetrics({ true_negatives: 100, false_positives: 5, false_negatives: 0, true_positives: 0 });
assert.strictEqual(zeroActualPosCm.actualPositive, 0);
assert.strictEqual(zeroActualPosCm.recall, 0);
assert.strictEqual(zeroActualPosCm.precision, 0);
assert.strictEqual(zeroActualPosCm.f1_score, 0);
assert.strictEqual(zeroActualPosCm.false_positive_rate, 5 / 105);

// Zero Denominator Case 4: Zero actual negatives (TN=0, FP=0)
const zeroActualNegCm = calculateConfusionMatrixMetrics({ true_negatives: 0, false_positives: 0, false_negatives: 5, true_positives: 25 });
assert.strictEqual(zeroActualNegCm.actualNegative, 0);
assert.strictEqual(zeroActualNegCm.false_positive_rate, 0);
assert.strictEqual(zeroActualNegCm.recall, 25 / 30);
assert.strictEqual(zeroActualNegCm.precision, 1.0);

// Null/undefined guard
const emptyCm = calculateConfusionMatrixMetrics(null);
assert.strictEqual(emptyCm.total, 0);
assert.strictEqual(emptyCm.recall, 0);

console.log('✓ [9/10] Confusion matrix derived metric calculation & zero denominator protections passed.');

// ─────────────────────────────────────────────────────────────────────
// 10. Dynamic Plain-Language Narrative Generation
// ─────────────────────────────────────────────────────────────────────
const t30 = findNearestTradeoff(0.30, rfTradeoffs);
const t50 = findNearestTradeoff(0.50, rfTradeoffs);
const t80 = findNearestTradeoff(0.80, rfTradeoffs);

const narrative30 = generateThresholdNarrative(t30, t50, 'Random Forest');
assert.ok(narrative30.summary.includes('76.0% of fraud attacks'));
assert.ok(narrative30.summary.includes('16 false alarms'));
assert.ok(narrative30.tradeOffComparison.includes('catches 5 additional fraud attacks'));

const narrative80 = generateThresholdNarrative(t80, t50, 'Random Forest');
assert.ok(narrative80.summary.includes('12.0% of fraud attacks'));
assert.ok(narrative80.tradeOffComparison.includes('misses 43 more fraud attacks'));

console.log('✓ [10/10] Dynamic plain-language threshold narrative generation passed.');

console.log('\n======================================================');
console.log('  All 10 Frontend Tests Passed via shared core.js!');
console.log('======================================================');
