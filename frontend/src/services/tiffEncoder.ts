/**
 * In-browser Baseline RGB TIFF Encoder for SatQuery AI.
 * Enables camera snapshots and standard image formats (.png, .jpg, .webp)
 * to be converted into 100% standard baseline 3-band RGB GeoTIFF-compatible .tif files
 * directly in the browser so that Rasterio/GDAL backend receives valid TIFF data.
 */

export function encodeRgbToTiff(rgbaData: Uint8ClampedArray, width: number, height: number): Blob {
  const pixelCount = width * height;
  const rgbBytes = new Uint8Array(pixelCount * 3);

  // Extract RGB bands from RGBA
  for (let i = 0, j = 0; i < rgbaData.length; i += 4, j += 3) {
    rgbBytes[j] = rgbaData[i];       // R
    rgbBytes[j + 1] = rgbaData[i + 1]; // G
    rgbBytes[j + 2] = rgbaData[i + 2]; // B
  }

  const numTags = 12;
  const headerSize = 8;
  const ifdSize = 2 + numTags * 12 + 4;
  const extraValuesOffset = headerSize + ifdSize;
  const extraValuesSize = 6 + 8 + 8; // BitsPerSample (6 bytes) + XRes (8 bytes) + YRes (8 bytes)
  const imageOffset = extraValuesOffset + extraValuesSize;
  const totalFileSize = imageOffset + rgbBytes.length;

  const buffer = new ArrayBuffer(totalFileSize);
  const view = new DataView(buffer);

  // 1. TIFF Header (Little Endian "II")
  view.setUint8(0, 0x49); // 'I'
  view.setUint8(1, 0x49); // 'I'
  view.setUint16(2, 42, true); // Magic 42
  view.setUint32(4, headerSize, true); // Offset to IFD0

  // 2. IFD Entry count
  let offset = headerSize;
  view.setUint16(offset, numTags, true);
  offset += 2;

  // Offsets for extra values
  const bitsPerSampleOffset = extraValuesOffset;
  const xResOffset = extraValuesOffset + 6;
  const yResOffset = extraValuesOffset + 14;

  const addTag = (tag: number, type: number, count: number, valueOrOffset: number) => {
    view.setUint16(offset, tag, true);
    view.setUint16(offset + 2, type, true);
    view.setUint32(offset + 4, count, true);
    view.setUint32(offset + 8, valueOrOffset, true);
    offset += 12;
  };

  // Types: 3 = SHORT (2 bytes), 4 = LONG (4 bytes), 5 = RATIONAL (8 bytes)
  addTag(0x0100, 4, 1, width);                       // ImageWidth
  addTag(0x0101, 4, 1, height);                      // ImageLength
  addTag(0x0102, 3, 3, bitsPerSampleOffset);          // BitsPerSample [8, 8, 8]
  addTag(0x0103, 3, 1, 1);                           // Compression: None
  addTag(0x0106, 3, 1, 2);                           // Photometric: RGB
  addTag(0x0111, 4, 1, imageOffset);                 // StripOffsets
  addTag(0x0115, 3, 1, 3);                           // SamplesPerPixel: 3
  addTag(0x0116, 4, 1, height);                      // RowsPerStrip
  addTag(0x0117, 4, 1, rgbBytes.length);             // StripByteCounts
  addTag(0x011a, 5, 1, xResOffset);                  // XResolution
  addTag(0x011b, 5, 1, yResOffset);                  // YResolution
  addTag(0x0128, 3, 1, 2);                           // ResolutionUnit: Inch

  // Offset to next IFD: 0
  view.setUint32(offset, 0, true);

  // 3. Write Extra Values
  // BitsPerSample: 8, 8, 8 (SHORT)
  view.setUint16(bitsPerSampleOffset, 8, true);
  view.setUint16(bitsPerSampleOffset + 2, 8, true);
  view.setUint16(bitsPerSampleOffset + 4, 8, true);

  // XResolution: 72/1 (RATIONAL)
  view.setUint32(xResOffset, 72, true);
  view.setUint32(xResOffset + 4, 1, true);

  // YResolution: 72/1 (RATIONAL)
  view.setUint32(yResOffset, 72, true);
  view.setUint32(yResOffset + 4, 1, true);

  // 4. Write Pixel Data
  const outBytes = new Uint8Array(buffer);
  outBytes.set(rgbBytes, imageOffset);

  return new Blob([buffer], { type: 'image/tiff' });
}

export async function canvasToTiffBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get 2D context from canvas');
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return encodeRgbToTiff(imgData.data, canvas.width, canvas.height);
}

export async function imageFileToTiff(file: File): Promise<File> {
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  if (['.tif', '.tiff', '.geotiff'].includes(ext)) {
    return file; // Already a GeoTIFF — pass through untouched (preserves georeferencing)
  }

  // Maximum dimension cap: keeps uncompressed TIFF output under ~12MB on mobile/low-memory devices.
  // A 2048×2048 RGB TIFF = 2048*2048*3 ≈ 12.6MB, well within upload limits.
  const MAX_DIM = 2048;

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Scale down preserving aspect ratio if either dimension exceeds MAX_DIM
        let { width, height } = img;
        if (width <= 0 || height <= 0) {
          reject(new Error('Image has invalid dimensions (0 width or height)'));
          return;
        }

        if (width > MAX_DIM || height > MAX_DIM) {
          const scale = MAX_DIM / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Canvas 2D context error'));
          return;
        }
        ctx.drawImage(img, 0, 0, width, height);
        const imgData = ctx.getImageData(0, 0, width, height);
        const blob = encodeRgbToTiff(imgData.data, width, height);
        const baseName = file.name.replace(/\.[^/.]+$/, '');
        const convertedFile = new File([blob], `${baseName}.tif`, { type: 'image/tiff' });
        img.src = '';
        resolve(convertedFile);
      };
      img.onerror = () => {
        img.src = '';
        reject(new Error('Failed to load image file for conversion'));
      };
      img.src = e.target?.result as string;
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}
