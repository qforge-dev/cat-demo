import './styles.css';

const $ = selector => document.querySelector(selector);
const fileInput = $('#file');
const choose = $('#choose');
const random = $('#random');
const stage = $('#stage');
const dropzone = $('#dropzone');
const empty = $('#empty');
const photo = $('#photo');
const verdict = $('#verdict');
const comment = $('#comment');
const meta = $('#meta');
const error = $('#error');

const worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
const modelUrl = new URL(`${import.meta.env.BASE_URL}cat.onnx?v=ec662b6ca20a633b`, location.href).href;

const lines = {
  cat: ['Yep, that’s a cat.', 'Cat detected. Very cat.', 'Confirmed: cat.', 'Big cat energy.', 'Checks out. Cat.'],
  notCat: ['Not a cat. Sorry.', 'No cat here. Suspicious.', 'Cat-free. Bold choice.', 'Nope. Not even close.', 'Nothing cat-like in sight.'],
  unsure: ['Hmm. Could go either way.', 'I’m squinting and still not sure.'],
};
const pick = list => list[Math.floor(Math.random() * list.length)];

let current = 0;
let photoUrl;
let busy = false;
let samplesPromise;
let remainingSamples = [];

function setBusy(value) {
  busy = value;
  choose.disabled = value;
  random.disabled = value;
  stage.dataset.loading = String(value);
}

function clearResult() {
  delete stage.dataset.result;
  verdict.hidden = true;
  meta.textContent = '';
  error.hidden = true;
}

function showError(message) {
  clearResult();
  comment.textContent = 'That one didn’t work.';
  error.textContent = message;
  error.hidden = false;
  setBusy(false);
}

function check(file) {
  if (busy) return;
  if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return showError('Pick a JPG, PNG or WebP photo.');
  if (file.size > 20 * 1024 * 1024) return showError('That file is over 20 MB. Pick a smaller photo.');
  current++;
  clearResult();
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(file);
  photo.src = photoUrl;
  photo.hidden = false;
  empty.hidden = true;
  comment.textContent = 'Squinting…';
  setBusy(true);
  worker.postMessage({ id: current, file, modelUrl });
}

worker.onmessage = ({ data }) => {
  if (data.id !== current) return;
  if (data.state === 'loading') comment.textContent = 'Waking up the model…';
  if (data.state === 'scoring') comment.textContent = 'Squinting…';
  if (data.state === 'error') showError(data.message);
  if (data.state === 'done') {
    setBusy(false);
    const isCat = data.prediction === 1;
    const catChance = 1 / (1 + Math.exp(-data.logit));
    const sure = isCat ? catChance : 1 - catChance;
    stage.dataset.result = String(data.prediction);
    verdict.textContent = isCat ? 'It’s a cat' : 'Not a cat';
    verdict.hidden = false;
    // Restart the stamp animation on repeat results.
    verdict.style.animation = 'none';
    void verdict.offsetWidth;
    verdict.style.animation = '';
    comment.textContent = sure < 0.65 ? pick(lines.unsure) : pick(isCat ? lines.cat : lines.notCat);
    meta.textContent = `${Math.round(sure * 100)}% sure · ${Math.max(1, Math.round(data.milliseconds))} ms, on your device`;
  }
};
worker.onerror = () => showError('This browser could not start the model. Try a recent Chrome, Edge, Firefox or Safari.');

choose.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { check(fileInput.files[0]); fileInput.value = ''; });
dropzone.addEventListener('dragover', event => { event.preventDefault(); if (!busy) dropzone.classList.add('dragging'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
dropzone.addEventListener('drop', event => { event.preventDefault(); dropzone.classList.remove('dragging'); check(event.dataTransfer.files[0]); });
document.addEventListener('paste', event => {
  const file = [...(event.clipboardData?.files ?? [])].find(f => f.type.startsWith('image/'));
  if (file) check(file);
});

async function nextSample() {
  if (!samplesPromise) {
    samplesPromise = fetch(`${import.meta.env.BASE_URL}samples.json`).then(async response => {
      if (!response.ok) throw new Error('samples');
      const samples = await response.json();
      if (samples.length !== 100) throw new Error('samples');
      return samples;
    }).catch(err => {
      samplesPromise = undefined;
      throw err;
    });
  }
  if (!remainingSamples.length) {
    remainingSamples = [...await samplesPromise];
    for (let i = remainingSamples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [remainingSamples[i], remainingSamples[j]] = [remainingSamples[j], remainingSamples[i]];
    }
  }
  return remainingSamples.pop();
}

random.addEventListener('click', async () => {
  if (busy) return;
  setBusy(true);
  try {
    const sample = await nextSample();
    const response = await fetch(`${import.meta.env.BASE_URL}${sample.file}`);
    if (!response.ok) throw new Error('sample');
    const blob = await response.blob();
    setBusy(false);
    check(new File([blob], `Random photo ${sample.number}`, { type: 'image/jpeg' }));
  } catch {
    showError('Could not load a random photo. Upload your own instead.');
  }
});

// Eyes in the empty frame follow the pointer.
const pupils = [...document.querySelectorAll('.pupil')];
window.addEventListener('pointermove', event => {
  if (empty.hidden) return;
  for (const pupil of pupils) {
    const rect = pupil.parentElement.getBoundingClientRect();
    const dx = event.clientX - (rect.left + rect.width / 2);
    const dy = event.clientY - (rect.top + rect.height / 2);
    const distance = Math.hypot(dx, dy) || 1;
    const reach = Math.min(14, distance / 12);
    pupil.style.setProperty('--px', `${(dx / distance) * reach}px`);
    pupil.style.setProperty('--py', `${(dy / distance) * reach * 0.6}px`);
  }
});
