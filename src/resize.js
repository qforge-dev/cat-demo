// Pillow's bilinear downsampling: separable filters, pixel-center coordinates and 22-bit coefficients.
function coefficients(inputSize, outputSize) {
  const scale = inputSize / outputSize;
  const filterScale = Math.max(1, scale);
  return Array.from({ length: outputSize }, (_, i) => {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - filterScale + 0.5));
    const end = Math.min(inputSize, Math.floor(center + filterScale + 0.5));
    const weights = Array.from({ length: end - start }, (_, j) => Math.max(0, 1 - Math.abs((j + start - center + 0.5) / filterScale)));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    return { start, weights: weights.map(weight => Math.floor(weight / total * 4194304 + 0.5)) };
  });
}

export function resizeRGB(rgba, width, height, size = 128) {
  const horizontal = coefficients(width, size);
  const vertical = coefficients(height, size);
  const intermediate = new Uint8Array(size * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < size; x++) {
      const { start, weights } = horizontal[x];
      for (let c = 0; c < 3; c++) {
        let sum = 2097152;
        for (let j = 0; j < weights.length; j++) sum += rgba[(y * width + start + j) * 4 + c] * weights[j];
        intermediate[(y * size + x) * 3 + c] = Math.min(255, Math.floor(sum / 4194304));
      }
    }
  }
  const result = new Float32Array(3 * size * size);
  for (let y = 0; y < size; y++) {
    const { start, weights } = vertical[y];
    for (let x = 0; x < size; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 2097152;
        for (let j = 0; j < weights.length; j++) sum += intermediate[((start + j) * size + x) * 3 + c] * weights[j];
        const pixel = Math.min(255, Math.floor(sum / 4194304));
        result[c * size * size + y * size + x] = Math.fround(Math.fround(pixel / 127.5) - 1);
      }
    }
  }
  return result;
}
