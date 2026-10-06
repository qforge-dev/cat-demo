import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import mjsUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import { resizeRGB } from './resize.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
let sessionPromise;

async function trainingPixels(file) {
  if (file.type !== 'image/png') return file;
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes.length < 8 || bytes[0] !== 137 || bytes[1] !== 80) return file;
  const view = new DataView(bytes.buffer);
  const chunks = [bytes.subarray(0, 8)];
  for (let offset = 8; offset + 12 <= bytes.length;) {
    const end = offset + view.getUint32(offset) + 12;
    if (end > bytes.length) return file;
    // Pillow RGB conversion retains keyed transparent pixels; canvas would discard them.
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    if (type !== 'tRNS') chunks.push(bytes.subarray(offset, end));
    offset = end;
  }
  return new Blob(chunks, { type: 'image/png' });
}

async function getSession(modelUrl) {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }).catch(error => {
      sessionPromise = undefined;
      throw error;
    });
  }
  return sessionPromise;
}

self.onmessage = async ({ data }) => {
  const { id, file, modelUrl } = data;
  let bitmap;
  try {
    self.postMessage({ id, state: 'loading' });
    const session = await getSession(modelUrl);
    // Match training's RGB values without applying embedded color profiles.
    bitmap = await createImageBitmap(await trainingPixels(file), { colorSpaceConversion: 'none' });
    if (bitmap.width * bitmap.height > 32000000) throw new Error('That picture is too large. Try an image under 32 megapixels.');
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    self.postMessage({ id, state: 'scoring' });
    const started = performance.now();
    const input = new ort.Tensor('float32', resizeRGB(pixels, bitmap.width, bitmap.height), [1, 3, 128, 128]);
    const output = await session.run({ image: input });
    const logit = output.logit.data[0];
    if (!Number.isFinite(logit)) throw new Error('The model could not read that picture. Please try another.');
    self.postMessage({ id, state: 'done', prediction: Number(logit >= 0), logit, milliseconds: performance.now() - started });
  } catch (error) {
    const message = error.message?.includes('megapixels') ? error.message : 'Could not check this picture. Try a JPG, PNG or WebP, or reload the page.';
    self.postMessage({ id, state: 'error', message });
  } finally {
    bitmap?.close();
  }
};
