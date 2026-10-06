import './styles.css';

const fileInput = document.querySelector('#file');
const choose = document.querySelector('#choose');
const dropzone = document.querySelector('#dropzone');
const readout = document.querySelector('.readout');
const answer = document.querySelector('#answer');
const status = document.querySelector('#status');
const digit = document.querySelector('#digit');
const error = document.querySelector('#error');
const photo = document.querySelector('#photo');
const filename = document.querySelector('#filename');
const worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module' });
const modelUrl = new URL(`${import.meta.env.BASE_URL}cat.onnx`, location.href).href;
let current = 0;
let photoUrl;
let busy = false;

function setBusy(value) {
  busy = value;
  choose.disabled = value;
  document.querySelectorAll('.sample').forEach(button => button.disabled = value);
  readout.dataset.loading = String(value);
}

function showError(message) {
  error.textContent = message;
  error.hidden = false;
  answer.textContent = 'Try another picture';
  status.textContent = '1 = cat · 0 = not cat';
  digit.textContent = '—';
  digit.setAttribute('aria-label', 'No prediction');
  delete readout.dataset.result;
  setBusy(false);
}

function check(file) {
  if (busy) return;
  if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return showError('Choose a JPG, PNG or WebP picture.');
  if (file.size > 20 * 1024 * 1024) return showError('That file is too large. Choose a picture under 20 MB.');
  current++;
  error.hidden = true;
  delete readout.dataset.result;
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(file);
  photo.src = photoUrl;
  photo.hidden = false;
  document.querySelector('#empty').hidden = true;
  filename.textContent = file.name || 'Sample picture';
  filename.hidden = false;
  answer.textContent = 'Checking your picture';
  status.textContent = 'Loading the model on your device…';
  digit.textContent = '…';
  digit.setAttribute('aria-label', 'Prediction in progress');
  setBusy(true);
  worker.postMessage({ id: current, file, modelUrl });
}

worker.onmessage = ({ data }) => {
  if (data.id !== current) return;
  if (data.state === 'loading') status.textContent = 'Loading the model on your device…';
  if (data.state === 'scoring') status.textContent = 'Looking for a cat…';
  if (data.state === 'error') showError(data.message);
  if (data.state === 'done') {
    setBusy(false);
    readout.dataset.result = String(data.prediction);
    digit.textContent = String(data.prediction);
    digit.setAttribute('aria-label', `${data.prediction}: ${data.prediction ? 'cat' : 'not cat'}`);
    answer.textContent = data.prediction ? 'Looks like a cat' : 'Doesn’t look like a cat';
    status.textContent = `Checked on your device in ${Math.round(data.milliseconds)} ms`;
  }
};
worker.onerror = () => showError('This browser could not start the model. Try a current Chrome, Edge, Firefox or Safari.');
choose.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { check(fileInput.files[0]); fileInput.value = ''; });
dropzone.addEventListener('dragover', event => { event.preventDefault(); if (!busy) dropzone.classList.add('dragging'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
dropzone.addEventListener('drop', event => { event.preventDefault(); dropzone.classList.remove('dragging'); check(event.dataTransfer.files[0]); });
document.querySelectorAll('.sample').forEach(button => button.addEventListener('click', async () => {
  if (busy) return;
  setBusy(true);
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}${button.dataset.sample}.jpg`);
    if (!response.ok) throw new Error('sample');
    const blob = await response.blob();
    setBusy(false);
    check(new File([blob], `${button.dataset.sample}.jpg`, { type: 'image/jpeg' }));
  } catch {
    showError('Could not load the sample. Choose your own picture instead.');
  }
}));
