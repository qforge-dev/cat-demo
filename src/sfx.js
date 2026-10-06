// Tiny synthesized sound effects: no audio files, nothing to download.
let ctx;
let master;
const MASTER_VOLUME = 0.3; // overall loudness of every effect
let muted = false;
try { muted = localStorage.getItem('cat-muted') === '1'; } catch {}

function audio() {
  if (muted) return null;
  if (!ctx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    ctx = new Ctx();
    master = ctx.createGain();
    master.gain.value = MASTER_VOLUME;
    master.connect(ctx.destination);
  }
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  return ctx;
}

// One note: oscillator with a pitch path and a quick attack/decay envelope.
function tone(c, { type = 'sine', from, to = from, at = 0, length = 0.15, volume = 0.2, filter }) {
  const t = c.currentTime + at;
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(from, t);
  osc.frequency.exponentialRampToValueAtTime(to, t + length);
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(volume, t + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + length);
  let node = osc;
  if (filter) {
    const f = c.createBiquadFilter();
    f.type = 'lowpass';
    f.frequency.value = filter;
    osc.connect(f);
    node = f;
  }
  node.connect(gain).connect(master);
  osc.start(t);
  osc.stop(t + length + 0.05);
}

// A meow: a buzzy source pushed through two vowel-like bands, pitch rising then falling.
function meow(c, at = 0, pitch = 1) {
  const t = c.currentTime + at;
  const length = 0.55;
  const osc = c.createOscillator();
  osc.type = 'sawtooth';
  osc.frequency.setValueAtTime(380 * pitch, t);
  osc.frequency.linearRampToValueAtTime(720 * pitch, t + 0.16);
  osc.frequency.linearRampToValueAtTime(430 * pitch, t + length);

  const gain = c.createGain();
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.exponentialRampToValueAtTime(0.22, t + 0.05);
  gain.gain.setValueAtTime(0.22, t + 0.25);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + length);

  // "mee" to "ow": the lower band opens up as the mouth does.
  const low = c.createBiquadFilter();
  low.type = 'bandpass';
  low.Q.value = 4;
  low.frequency.setValueAtTime(600, t);
  low.frequency.linearRampToValueAtTime(950, t + length);
  const high = c.createBiquadFilter();
  high.type = 'bandpass';
  high.Q.value = 5;
  high.frequency.setValueAtTime(2600, t);
  high.frequency.linearRampToValueAtTime(1400, t + length);

  const mix = c.createGain();
  mix.gain.value = 1.4;
  osc.connect(low).connect(mix);
  osc.connect(high).connect(mix);
  mix.connect(gain).connect(master);
  osc.start(t);
  osc.stop(t + length + 0.05);
}

export const sfx = {
  get muted() { return muted; },
  setMuted(value) {
    muted = value;
    try { localStorage.setItem('cat-muted', value ? '1' : '0'); } catch {}
  },
  pop() {
    const c = audio();
    if (!c) return;
    tone(c, { type: 'triangle', from: 520, to: 980, length: 0.07, volume: 0.18 });
  },
  thump(c = audio()) {
    if (!c) return;
    tone(c, { type: 'sine', from: 170, to: 45, length: 0.16, volume: 0.3 });
  },
  cat(sure = 1) {
    const c = audio();
    if (!c) return;
    this.thump(c);
    meow(c, 0.1, 0.92 + (sure - 0.5) * 0.3);
  },
  notCat() {
    const c = audio();
    if (!c) return;
    this.thump(c);
    tone(c, { type: 'sawtooth', from: 310, to: 250, at: 0.12, length: 0.3, volume: 0.11, filter: 900 });
    tone(c, { type: 'sawtooth', from: 250, to: 150, at: 0.44, length: 0.5, volume: 0.11, filter: 700 });
  },
  unsure() {
    const c = audio();
    if (!c) return;
    tone(c, { type: 'triangle', from: 330, to: 400, at: 0, length: 0.12, volume: 0.16 });
    tone(c, { type: 'triangle', from: 400, to: 310, at: 0.14, length: 0.18, volume: 0.16 });
  },
  error() {
    const c = audio();
    if (!c) return;
    tone(c, { type: 'square', from: 180, to: 140, length: 0.18, volume: 0.1, filter: 600 });
  },
};
