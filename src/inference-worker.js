import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import mjsUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import { resizeRGB } from './resize.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
let sessionPromise;

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
    bitmap = await createImageBitmap(file);
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
