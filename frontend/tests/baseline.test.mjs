import test from 'node:test';
import assert from 'node:assert/strict';

// The frontend state module only needs browser storage during initialisation.
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
const { state } = await import('../js/state.js');
const { renderSummaryStrip } = await import('../js/ui.js');

test('completed classifier and HMM accuracy appear in the summary', () => {
  state.trips = [{ trip_id: 'sample' }];
  state.tripRows = new Map([['sample', { status: 'done', result: {
    total_runtime_s: 1, hmm_used: true, metrics: {
      classifier: { top1_all: .8, route_continuity: .7, sequence_arc_f1: .6 },
      hmm: { top1_all: .9, route_continuity: .8, sequence_arc_f1: .7 },
    },
  }}]]);
  const element = { style: {}, innerHTML: '' };
  globalThis.document = { getElementById: id => id === 'summary-strip' ? element : null };
  renderSummaryStrip(state.getSummaryStats());
  assert.match(element.innerHTML, /90\.0%/);
  assert.match(element.innerHTML, /Classifier: 80\.0%/);
});

test('mixed fallback results compare accuracy on the same trips', () => {
  state.trips = [{ trip_id: 'a' }, { trip_id: 'b' }];
  state.tripRows = new Map([
    ['a', {status:'done', result:{hmm_used:true, metrics:{classifier:{top1_all:.8}, hmm:{top1_all:.9}}}}],
    ['b', {status:'done', result:{hmm_used:false, metrics:{classifier:{top1_all:.4}, hmm:null}}}],
  ]);
  const summary = state.getSummaryStats();
  assert.ok(Math.abs(summary.averages.top1_all.delta - .1) < 1e-12);
  assert.equal(summary.hmmCount, 1);
});

test('classifier-only results do not claim an HMM result', () => {
  state.tripRows = new Map([['sample', { status: 'done', result: {
    total_runtime_s: 1, metrics: { classifier: { top1_all: .8 }, hmm: null },
  }}]]);
  const summary = state.getSummaryStats();
  assert.equal(summary.hasHmm, false);
  assert.equal(summary.averages.top1_all.hmm, null);
});
