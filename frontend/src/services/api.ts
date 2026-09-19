import {
  ImageUploadResponse,
  ReportRequest,
  ReportResponse,
  SatQueryResult,
  SystemHealth,
} from '../types/api';
import { imageFileToTiff } from './tiffEncoder';

const API_BASE = ''; // Uses Vite proxy in development, or relative paths in production

export class SatQueryAPI {
  /**
   * Health check for system and inference status
   */
  static async getHealth(): Promise<SystemHealth> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      throw new Error(`Health check failed with status ${res.status}`);
    }
    return res.json();
  }

  /**
   * Upload a satellite raster or standard image
   * Automatically converts non-TIFF images (PNG, JPG, WebP) to baseline RGB TIFF
   */
  static async uploadImage(
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<ImageUploadResponse> {
    // Convert to TIFF if needed (camera snapshots, PNG, JPG)
    const tiffFile = await imageFileToTiff(file);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_BASE}/api/v1/upload`);

      if (xhr.upload && onProgress) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            onProgress(percent);
          }
        };
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const json: ImageUploadResponse = JSON.parse(xhr.responseText);
            resolve(json);
          } catch {
            reject(new Error('Invalid JSON response from upload endpoint'));
          }
        } else {
          let errorDetail = `Upload failed with status ${xhr.status}`;
          try {
            const errJson = JSON.parse(xhr.responseText);
            if (errJson.detail) errorDetail = errJson.detail;
          } catch {
            // fallback to status
          }
          reject(new Error(errorDetail));
        }
      };

      xhr.onerror = () => reject(new Error('Network error during file upload'));

      const formData = new FormData();
      formData.append('file', tiffFile, tiffFile.name);
      xhr.send(formData);
    });
  }

  /**
   * Execute query via standard REST POST (used directly or as WebSocket fallback)
   */
  static async executeQuery(
    query: string,
    imageIds: string[],
    history?: { role: string; content: string }[]
  ): Promise<SatQueryResult> {
    const res = await fetch(`${API_BASE}/api/v1/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        image_ids: imageIds,
        history: history || [],
      }),
    });

    if (!res.ok) {
      let errorDetail = `Query failed with status ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errorDetail = errJson.detail;
      } catch {
        // fallback
      }
      throw new Error(errorDetail);
    }

    return res.json();
  }

  /**
   * Generate an executive mission briefing PDF report
   */
  static async generateReport(request: ReportRequest): Promise<ReportResponse> {
    const res = await fetch(`${API_BASE}/api/v1/report/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!res.ok) {
      let errorDetail = `Report generation failed (${res.status})`;
      try {
        const err = await res.json();
        if (err.detail) errorDetail = err.detail;
      } catch {
        // fallback
      }
      throw new Error(errorDetail);
    }

    return res.json();
  }

  /**
   * Get direct download link for a generated briefing report (PDF)
   */
  static getReportDownloadUrl(reportId: string): string {
    return `${API_BASE}/api/v1/report/${reportId}`;
  }

  /**
   * Get direct download link for Word document (.docx) format
   */
  static getReportDocxDownloadUrl(reportId: string): string {
    return `${API_BASE}/api/v1/report/${reportId}/docx`;
  }

  /**
   * Get direct link to view briefing PDF inline in the browser
   */
  static getReportViewUrl(reportId: string): string {
    return `${API_BASE}/api/v1/report/${reportId}?view=true`;
  }

  /**
   * Get direct preview URL for a GeoTIFF raster
   * Safely transforms full paths, static URLs, or bare filenames into dynamic preview endpoints.
   */
  static getRasterPreviewUrl(rasterIdOrName?: string): string {
    if (!rasterIdOrName) return '';
    if (rasterIdOrName.startsWith('/api/v1/preview/')) return `${API_BASE}${rasterIdOrName}`;
    const clean = rasterIdOrName.split('?')[0].split('/').pop() || '';
    if (!clean) return rasterIdOrName;
    return `${API_BASE}/api/v1/preview/${clean}`;
  }

  /**
   * Get map tile URL template for Leaflet
   */
  static getTileUrlTemplate(fileId: string): string {
    return `${API_BASE}/api/v1/tiles/${fileId}/{z}/{x}/{y}.png`;
  }

  /**
   * Fetch specialist model registry and empirical benchmarks
   */
  static async getModelsCatalog(): Promise<Record<string, any>> {
    const res = await fetch(`${API_BASE}/api/v1/models`);
    if (!res.ok) {
      throw new Error(`Failed to load model registry (${res.status})`);
    }
    return res.json();
  }

  /**
   * Execute Multimodal Cloud-Penetrating Flood & Safe Zone Assessment
   */
  static async analyzeMultimodalFlood(
    query?: string,
    opticalImageId?: string,
    sarImageId?: string
  ): Promise<SatQueryResult> {
    const res = await fetch(`${API_BASE}/api/v1/multimodal-flood/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query || undefined,
        optical_image_id: opticalImageId || undefined,
        sar_image_id: sarImageId || undefined,
      }),
    });

    if (!res.ok) {
      let errorDetail = `Multimodal flood assessment failed (${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errorDetail = errJson.detail;
      } catch {
        // fallback
      }
      throw new Error(errorDetail);
    }

    return res.json();
  }
}
