import React, { useState } from 'react';
import { Satellite, Radio, Check, Loader2, Layers, Waves, Building2, ArrowRight } from 'lucide-react';
import { Modal } from '../ui/Modal';
import { Badge } from '../ui/Badge';
import { SatQueryAPI } from '../../services/api';
import { useChat } from '../../context/ChatContext';

interface SamplePickerModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SamplePresetInfo {
  id: string;
  filename: string;
  url: string;
  thumbnailUrl: string;
  name: string;
  sensor: string;
  modality: 'OPTICAL' | 'SAR';
  category: 'pairs' | 'optical' | 'sar' | 'disaster';
  description: string;
  details: string;
}

interface MissionPairInfo {
  id: string;
  title: string;
  category: string;
  icon: React.ElementType;
  description: string;
  query: string;
  files: Array<{ url: string; name: string; thumb: string; modality: 'OPTICAL' | 'SAR' }>;
}

const MISSION_PAIRS: MissionPairInfo[] = [
  {
    id: 'flood_pair',
    title: 'Disaster Flood Inundation Assessment',
    category: 'Bi-temporal Change',
    icon: Waves,
    description: 'Pre-flood baseline (t₁) vs post-monsoon inundated surge (t₂). Evaluates submerged hectares and parcel displacement with ChangeFormer.',
    query: 'Detect flood inundation, map submerged parcel boundaries, and quantify displacement between pre- and post-disaster dates.',
    files: [
      { url: '/samples/flood_t1.tif', name: 'pre_flood_t1.tif', thumb: '/api/v1/preview/flood_t1.tif', modality: 'OPTICAL' },
      { url: '/samples/flood_t2.tif', name: 'post_flood_t2.tif', thumb: '/api/v1/preview/flood_t2.tif', modality: 'OPTICAL' },
    ],
  },
  {
    id: 'fusion_pair',
    title: 'Cloud-Penetrating Optical + SAR Fusion',
    category: 'Cross-Modal Fusion',
    icon: Satellite,
    description: 'Combines dense cloud-obscured Sentinel-2 optical scene with all-weather Sentinel-1 C-Band microwave radar backscatter.',
    query: 'Execute cross-modal optical and microwave SAR fusion to penetrate dense cloud cover and reconstruct ground terrain.',
    files: [
      { url: '/samples/fusion_optical.tif', name: 'sentinel2_optical_cloudy.tif', thumb: '/api/v1/preview/fusion_optical.tif', modality: 'OPTICAL' },
      { url: '/samples/fusion_sar.tif', name: 'sentinel1_sar_backscatter.tif', thumb: '/api/v1/preview/fusion_sar.tif', modality: 'SAR' },
    ],
  },
  {
    id: 'urban_pair',
    title: 'Urban Sprawl & Infrastructure Development',
    category: 'Bi-temporal Expansion',
    icon: Building2,
    description: 'Bi-temporal survey comparing historical baseline with newly built industrial and residential corridors.',
    query: 'Analyze bi-temporal urban expansion, new residential footprints, and roadway corridors between these acquisition dates.',
    files: [
      { url: '/samples/urban_t1.tif', name: 'urban_baseline_t1.tif', thumb: '/api/v1/preview/urban_t1.tif', modality: 'OPTICAL' },
      { url: '/samples/urban_t2.tif', name: 'urban_expansion_t2.tif', thumb: '/api/v1/preview/urban_t2.tif', modality: 'OPTICAL' },
    ],
  },
  {
    id: 'cloud_free_flood_pair',
    title: 'All-Weather Cloud-Free Flood & Safe Zones',
    category: 'Multimodal Disaster',
    icon: Waves,
    description: 'Penetrate storm clouds using Sentinel-1 SAR radar to clear cloud cover, map inundated areas, and delineate elevated dry safe zones.',
    query: 'Penetrate storm clouds using Sentinel-1 SAR and Sentinel-2 optical imagery, reconstruct a cloud-free ground view, calculate total flooded area, and pinpoint elevated safe evacuation zones.',
    files: [
      { url: '/samples/public_flood_cloudy_optical.tif', name: 'cloudy_optical.tif', thumb: '/api/v1/preview/public_flood_cloudy_optical.tif', modality: 'OPTICAL' },
      { url: '/samples/public_flood_sentinel1_sar.tif', name: 'sentinel1_sar.tif', thumb: '/api/v1/preview/public_flood_sentinel1_sar.tif', modality: 'SAR' },
    ],
  },
];

const SERVER_SAMPLES: SamplePresetInfo[] = [
  // ── Optical Multispectral ──────────────────────────────────
  {
    id: 'forest_vqa',
    filename: 'forest_vqa.tif',
    url: '/samples/forest_vqa.tif',
    thumbnailUrl: '/api/v1/preview/forest_vqa.tif',
    name: 'Forest Canopy & Vegetation VQA',
    sensor: 'Sentinel-2 MSI Multispectral',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Multispectral scene featuring dense forest canopies, spectral vegetation index responses, and river channels for VQA analysis.',
    details: '3 Bands • EPSG:4326 • 10m GSD',
  },
  {
    id: 'port_grounding',
    filename: 'port_grounding.tif',
    url: '/samples/port_grounding.tif',
    thumbnailUrl: '/api/v1/preview/port_grounding.tif',
    name: 'Maritime Port & Harbor Grounding',
    sensor: 'High-Resolution Optical Satellite',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Complex harbor facility with commercial vessels, dry docks, storage tanks, and industrial cranes for text-guided visual grounding.',
    details: '3 Bands (RGB) • EPSG:4326 • Object Grounding',
  },
  {
    id: 'urban_t1',
    filename: 'urban_t1.tif',
    url: '/samples/urban_t1.tif',
    thumbnailUrl: '/api/v1/preview/urban_t1.tif',
    name: 'Urban Core Baseline (t₁)',
    sensor: 'High-Resolution Optical Satellite',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Optical satellite scene showing dense built-up residential districts, arterial highways, and industrial blocks.',
    details: '3 Bands • EPSG:4326 • Sub-meter GSD',
  },
  {
    id: 'fusion_optical',
    filename: 'fusion_optical.tif',
    url: '/samples/fusion_optical.tif',
    thumbnailUrl: '/api/v1/preview/fusion_optical.tif',
    name: 'Sentinel-2 Optical (Cloudy Pass)',
    sensor: 'Copernicus Sentinel-2 MSI',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Multispectral optical capture severely obscured by thick atmospheric cumulus clouds.',
    details: '3 Bands (RGB) • EPSG:4326 • 10m GSD',
  },

  // ── SAR Microwave Radar ──────────────────────────────────
  {
    id: 'fusion_sar',
    filename: 'fusion_sar.tif',
    url: '/samples/fusion_sar.tif',
    thumbnailUrl: '/api/v1/preview/fusion_sar.tif',
    name: 'Sentinel-1 C-Band SAR Radar',
    sensor: 'Copernicus Sentinel-1 RTC Radar',
    modality: 'SAR',
    category: 'sar',
    description: 'All-weather, day-and-night C-band microwave synthetic aperture radar with dual polarizations (VV/VH), penetrating cloud cover.',
    details: '2 Bands (VV/VH) • EPSG:4326 • Calibrated Backscatter',
  },
  {
    id: 'public_flood_sar',
    filename: 'public_flood_sentinel1_sar.tif',
    url: '/samples/public_flood_sentinel1_sar.tif',
    thumbnailUrl: '/api/v1/preview/public_flood_sentinel1_sar.tif',
    name: 'Sentinel-1 SAR Inundation Radar',
    sensor: 'Sentinel-1 C-Band SAR Radar',
    modality: 'SAR',
    category: 'sar',
    description: 'Low-backscatter specular radar signature delineating standing water bodies, floodplains, and submerged surfaces.',
    details: '2 Bands (VV/VH) • EPSG:4326 • Water Penetration',
  },

  // ── Disasters & Floods ───────────────────────────────────
  {
    id: 'flood_t1',
    filename: 'flood_t1.tif',
    url: '/samples/flood_t1.tif',
    thumbnailUrl: '/api/v1/preview/flood_t1.tif',
    name: 'Pre-Disaster River Basin (t₁ Baseline)',
    sensor: 'Multispectral Satellite Sensor',
    modality: 'OPTICAL',
    category: 'disaster',
    description: 'Agricultural floodplains captured under dry baseline conditions prior to severe riverine embankment breach.',
    details: '3 Bands • EPSG:4326 • Baseline Pass',
  },
  {
    id: 'flood_t2',
    filename: 'flood_t2.tif',
    url: '/samples/flood_t2.tif',
    thumbnailUrl: '/api/v1/preview/flood_t2.tif',
    name: 'Post-Disaster Submerged Inundation (t₂ Surge)',
    sensor: 'Multispectral Satellite Sensor',
    modality: 'OPTICAL',
    category: 'disaster',
    description: 'Post-disaster inundation pass showing massive submerged acreage, damaged infrastructure, and waterlogged parcels.',
    details: '3 Bands • EPSG:4326 • Surge Pass',
  },
];

export const SamplePickerModal: React.FC<SamplePickerModalProps> = ({ isOpen, onClose }) => {
  const { attachImage, activeImages, clearAttachedImages, submitQuery, setPendingQueryText } = useChat();
  const [activeTab, setActiveTab] = useState<'pairs' | 'optical' | 'sar' | 'disaster'>('pairs');
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelectSample = async (sample: SamplePresetInfo) => {
    setLoadingId(sample.id);
    setError(null);

    try {
      const resp = await fetch(`${sample.url}?t=${Date.now()}`, { cache: 'no-store' });
      if (!resp.ok) throw new Error(`Failed to load sample: ${sample.filename}`);
      const blob = await resp.blob();
      const file = new File([blob], sample.filename, { type: 'image/tiff' });
      const uploaded = await SatQueryAPI.uploadImage(file);
      attachImage(uploaded);
      onClose();
    } catch (err: any) {
      console.error('Error attaching sample:', err);
      setError(err.message || 'Failed to attach sample');
    } finally {
      setLoadingId(null);
    }
  };

  const handleLoadPair = async (pair: MissionPairInfo) => {
    setLoadingId(pair.id);
    setError(null);

    try {
      clearAttachedImages();
      const uploadedImages: any[] = [];
      for (const item of pair.files) {
        const resp = await fetch(`${item.url}?t=${Date.now()}`, { cache: 'no-store' });
        if (!resp.ok) throw new Error(`Failed to load: ${item.name}`);
        const blob = await resp.blob();
        const file = new File([blob], item.name, { type: 'image/tiff' });
        const uploaded = await SatQueryAPI.uploadImage(file);
        uploadedImages.push(uploaded);
        attachImage(uploaded);
      }

      onClose();
      // Focus textarea so user can type their own query for the paired rasters
      setTimeout(() => {
        const textarea = document.querySelector('textarea');
        if (textarea) {
          textarea.focus();
        }
      }, 50);
    } catch (err: any) {
      console.error('Error loading mission pair:', err);
      setError(err.message || 'Failed to ingest mission pair');
    } finally {
      setLoadingId(null);
    }
  };

  const filteredSamples = SERVER_SAMPLES.filter((s) => {
    if (activeTab === 'optical') return s.category === 'optical';
    if (activeTab === 'sar') return s.category === 'sar';
    if (activeTab === 'disaster') return s.category === 'disaster';
    return true;
  });

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Satellite Samples Catalog"
      maxWidth="max-w-3xl"
    >
      <div className="space-y-3.5 select-none text-xs">
        {/* Category Navigation Pills */}
        <div className="flex items-center gap-1.5 p-1 bg-[var(--bg-surface)] rounded-xl border border-[var(--border-subtle)] text-[11px] font-medium overflow-x-auto">
          <button
            type="button"
            onClick={() => setActiveTab('pairs')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
              activeTab === 'pairs'
                ? 'bg-[var(--bg-card)] text-[#0EA5E9] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-[#0EA5E9]" />
            <span>Mission Pairs (Attach 2 Rasters)</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('optical')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
              activeTab === 'optical'
                ? 'bg-[var(--bg-card)] text-[#0EA5E9] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Satellite className="w-3.5 h-3.5 text-[#0EA5E9]" />
            <span>Optical Multispectral</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('sar')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
              activeTab === 'sar'
                ? 'bg-[var(--bg-card)] text-[#0EA5E9] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Radio className="w-3.5 h-3.5 text-[#0EA5E9]" />
            <span>SAR Microwave Radar</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('disaster')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
              activeTab === 'disaster'
                ? 'bg-[var(--bg-card)] text-[#0EA5E9] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Waves className="w-3.5 h-3.5 text-[#0EA5E9]" />
            <span>Disasters & Floods</span>
          </button>
        </div>

        {error && (
          <div className="p-3 bg-[#0EA5E9]/10 border border-[#0EA5E9]/30 rounded-xl text-[#0EA5E9] text-xs">
            {error}
          </div>
        )}

        {/* ── TAB 1: 1-CLICK MISSION PAIRS ── */}
        {activeTab === 'pairs' && (
          <div className="space-y-3 max-h-[460px] overflow-y-auto pr-1">
            {MISSION_PAIRS.map((pair) => {
              const Icon = pair.icon;
              const isLoading = loadingId === pair.id;

              return (
                <div
                  key={pair.id}
                  className="p-3.5 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] hover:border-[#0EA5E9]/40 hover:bg-[var(--bg-card)] transition-all shadow-subtle"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-2.5">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] flex items-center justify-center text-[#0EA5E9]">
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-xs text-[var(--text-main)]">
                          {pair.title}
                        </h4>
                        <span className="text-[10px] font-mono text-[#0EA5E9]">
                          {pair.category}
                        </span>
                      </div>
                    </div>

                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() => handleLoadPair(pair)}
                      className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0EA5E9] hover:bg-[#0284C7] text-white font-medium text-xs shadow-sm transition-colors cursor-pointer disabled:opacity-50"
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          <span>Ingesting Pair...</span>
                        </>
                      ) : (
                        <>
                          <span>Attach Pair</span>
                          <ArrowRight className="w-3 h-3" />
                        </>
                      )}
                    </button>
                  </div>

                  <p className="text-[11.5px] text-[var(--text-muted)] leading-relaxed mb-3">
                    {pair.description}
                  </p>

                  {/* Thumbnail Pair Strip */}
                  <div className="grid grid-cols-2 gap-2">
                    {pair.files.map((f, fIdx) => (
                      <div
                        key={fIdx}
                        className="flex items-center gap-2 p-1.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-[11px]"
                      >
                        <img
                          src={f.thumb}
                          alt={f.name}
                          className="w-8 h-8 rounded object-cover border border-[var(--border-subtle)] bg-black"
                          onError={(e) => {
                            (e.currentTarget as HTMLImageElement).src = '/api/v1/preview/cartosat_t1.tif';
                          }}
                        />
                        <div className="truncate">
                          <span className="font-mono text-[10.5px] text-[var(--text-main)] block truncate">
                            {f.name}
                          </span>
                          <span className="font-mono text-[9px] text-[#0EA5E9]">
                            {f.modality}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ── TAB 2, 3, 4: INDIVIDUAL SATELLITE RASTERS ── */}
        {activeTab !== 'pairs' && (
          <div className="grid gap-2.5 max-h-[460px] overflow-y-auto pr-1">
            {filteredSamples.map((sample) => {
              const isAlreadyAttached = activeImages.some((img) => img.filename === sample.filename);
              const isLoading = loadingId === sample.id;

              return (
                <div
                  key={sample.id}
                  onClick={() => !isLoading && !isAlreadyAttached && handleSelectSample(sample)}
                  className={`p-3 rounded-xl border transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                    isAlreadyAttached
                      ? 'border-[#0EA5E9]/40 bg-[#0EA5E9]/5 cursor-default'
                      : 'border-[var(--border-subtle)] hover:border-[#0EA5E9]/40 bg-[var(--bg-panel)] hover:bg-[var(--bg-card)] cursor-pointer shadow-subtle'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <img
                      src={sample.thumbnailUrl}
                      alt={sample.name}
                      className="w-12 h-12 rounded-lg object-cover border border-[var(--border-subtle)] bg-black shrink-0"
                      onError={(e) => {
                        (e.currentTarget as HTMLImageElement).src = '/api/v1/preview/cartosat_t1.tif';
                      }}
                    />

                    <div>
                      <div className="flex items-center gap-2 mb-0.5">
                        <h4 className="text-xs font-semibold text-[var(--text-main)]">
                          {sample.name}
                        </h4>
                        <Badge variant="neutral">{sample.modality}</Badge>
                      </div>
                      <p className="text-[11px] text-[var(--text-muted)] line-clamp-2 leading-relaxed">
                        {sample.description}
                      </p>
                      <span className="text-[10px] font-mono text-[#0EA5E9] block mt-1">
                        {sample.sensor} • {sample.details}
                      </span>
                    </div>
                  </div>

                  <div className="shrink-0 flex items-center justify-end">
                    {isLoading ? (
                      <div className="flex items-center gap-1.5 text-xs text-[#0EA5E9] font-mono">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Attaching...</span>
                      </div>
                    ) : isAlreadyAttached ? (
                      <div className="flex items-center gap-1 text-[11px] font-mono text-[#0EA5E9] bg-[#0EA5E9]/10 px-2.5 py-1 rounded-lg border border-[#0EA5E9]/25">
                        <Check className="w-3.5 h-3.5" />
                        <span>Attached</span>
                      </div>
                    ) : (
                      <span className="text-[11px] font-medium text-[var(--text-muted)] group-hover:text-[var(--text-main)] px-3 py-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] hover:bg-[var(--bg-card)]">
                        Attach
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Modal>
  );
};
