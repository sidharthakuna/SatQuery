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
    description: 'Cartosat bi-temporal survey comparing historical baseline with newly built industrial and residential corridors.',
    query: 'Analyze bi-temporal urban expansion, new residential footprints, and roadway corridors between these acquisition dates.',
    files: [
      { url: '/samples/urban_t1.tif', name: 'cartosat_urban_t1.tif', thumb: '/api/v1/preview/urban_t1.tif', modality: 'OPTICAL' },
      { url: '/samples/urban_t2.tif', name: 'cartosat_urban_t2.tif', thumb: '/api/v1/preview/urban_t2.tif', modality: 'OPTICAL' },
    ],
  },
];

const SERVER_SAMPLES: SamplePresetInfo[] = [
  {
    id: 'cartosat_t1',
    filename: 'cartosat_t1.tif',
    url: '/samples/cartosat_t1.tif',
    thumbnailUrl: '/api/v1/preview/cartosat_t1.tif',
    name: 'Cartosat-2S Optical (t₁ Baseline)',
    sensor: 'ISRO Cartosat-2S High-Res Imager',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Multispectral 4-band optical raster (Blue, Green, Red, NIR) at sub-meter ground sampling distance.',
    details: '4 Bands • EPSG:4326 • 0.65m GSD',
  },
  {
    id: 'cartosat_t2',
    filename: 'cartosat_t2.tif',
    url: '/samples/cartosat_t2.tif',
    thumbnailUrl: '/api/v1/preview/cartosat_t2.tif',
    name: 'Cartosat-2S Optical (t₂ Surveillance)',
    sensor: 'ISRO Cartosat-2S Repeat Pass',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Bi-temporal repeat optical scene acquired post-development for urban change and infrastructure analysis.',
    details: '4 Bands • EPSG:4326 • 0.65m GSD',
  },
  {
    id: 'risat_sar',
    filename: 'risat_sar.tif',
    url: '/samples/risat_sar.tif',
    thumbnailUrl: '/api/v1/preview/risat_sar.tif',
    name: 'RISAT-1 C-Band Microwave SAR',
    sensor: 'ISRO Radar Imaging Satellite (RISAT)',
    modality: 'SAR',
    category: 'sar',
    description: 'All-weather, day-and-night C-band microwave synthetic aperture radar with dual polarizations (VV/VH).',
    details: '2 Bands (VV/VH) • EPSG:4326 • Radar Backscatter',
  },
  {
    id: 'forest_vqa',
    filename: 'forest_vqa.tif',
    url: '/samples/forest_vqa.tif',
    thumbnailUrl: '/api/v1/preview/forest_vqa.tif',
    name: 'Forest Canopy & Vegetation VQA',
    sensor: 'Sentinel-2 Multispectral MSI',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Multispectral scene featuring dense forest canopies, spectral vegetation index responses, and water bodies.',
    details: '4 Bands • EPSG:4326 • 10m GSD',
  },
  {
    id: 'port_grounding',
    filename: 'port_grounding.tif',
    url: '/samples/port_grounding.tif',
    thumbnailUrl: '/api/v1/preview/port_grounding.tif',
    name: 'Maritime Port & Defense Infrastructure',
    sensor: 'DIOR High-Res Aerial Sensor',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Complex harbor facility with commercial vessels, dry docks, storage tanks, and industrial cranes for visual grounding.',
    details: '3 Bands (RGB) • EPSG:4326 • Object Grounding',
  },
  {
    id: 'sentinel2_coastal',
    filename: 'sentinel2_coastal.tif',
    url: '/samples/sentinel2_coastal.tif',
    thumbnailUrl: '/api/v1/preview/sentinel2_coastal.tif',
    name: 'Sentinel-2 Coastal Metropolis & Port',
    sensor: 'Sentinel-2 MSI Optical Imager',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Ultra-sharp 768x768 multispectral optical scene covering commercial deepwater berths, breakwaters, and urban core.',
    details: '3 Bands (RGB) • EPSG:4326 • 768x768 High-Res GeoTIFF',
  },
  {
    id: 'flood_t1',
    filename: 'flood_t1.tif',
    url: '/samples/flood_t1.tif',
    thumbnailUrl: '/api/v1/preview/flood_t1.tif',
    name: 'Pre-Disaster Flood Basin (t₁)',
    sensor: 'RapidEye Multispectral Sensor',
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
    name: 'Post-Disaster Submerged Inundation (t₂)',
    sensor: 'RapidEye Multispectral Sensor',
    modality: 'OPTICAL',
    category: 'disaster',
    description: 'Post-disaster inundation pass showing massive submerged acreage, damaged infrastructure, and waterlogged parcels.',
    details: '3 Bands • EPSG:4326 • Surge Pass',
  },
  {
    id: 'fusion_optical',
    filename: 'fusion_optical.tif',
    url: '/samples/fusion_optical.tif',
    thumbnailUrl: '/api/v1/preview/fusion_optical.tif',
    name: 'Sentinel-2 Optical (Cloudy Pass)',
    sensor: 'Copernicus Sentinel-2 MSI (Real Satellite GeoTIFF)',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Authentic 768x768 Sentinel-2 multispectral scene over Visakhapatnam Port with atmospheric cumulus cloud cover.',
    details: '3 Bands (RGB) • EPSG:4326 • 10m GSD',
  },
  {
    id: 'fusion_optical_clean',
    filename: 'fusion_optical_clean.tif',
    url: '/samples/fusion_optical_clean.tif',
    thumbnailUrl: '/api/v1/preview/fusion_optical_clean.tif',
    name: 'Sentinel-2 Optical (Clear Sky Ground Truth)',
    sensor: 'Copernicus Sentinel-2 MSI (Real Satellite GeoTIFF)',
    modality: 'OPTICAL',
    category: 'optical',
    description: 'Pristine, cloud-free true-color satellite capture of Visakhapatnam Port under clear skies.',
    details: '3 Bands (RGB) • EPSG:4326 • 10m GSD',
  },
  {
    id: 'fusion_sar',
    filename: 'fusion_sar.tif',
    url: '/samples/fusion_sar.tif',
    thumbnailUrl: '/api/v1/preview/fusion_sar.tif',
    name: 'Sentinel-1 C-Band SAR Radar',
    sensor: 'Copernicus Sentinel-1 RTC (Real Satellite GeoTIFF)',
    modality: 'SAR',
    category: 'sar',
    description: 'Authentic C-band radar backscatter penetrating clouds to reveal maritime shipping and port structures.',
    details: '2 Bands (VV/VH) • EPSG:4326 • Calibrated Backscatter',
  },
];

export const SamplePickerModal: React.FC<SamplePickerModalProps> = ({ isOpen, onClose }) => {
  const { attachImage, activeImages, setPendingQueryText } = useChat();
  const [activeTab, setActiveTab] = useState<'pairs' | 'optical' | 'sar' | 'disaster'>('pairs');
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelectSample = async (sample: SamplePresetInfo) => {
    if (activeImages.length >= 2) {
      setError('Maximum 2 images can be attached simultaneously. Remove one before adding another.');
      return;
    }

    setLoadingId(sample.id);
    setError(null);

    try {
      const resp = await fetch(`${sample.url}?t=${Date.now()}`, { cache: 'no-store' });
      if (!resp.ok) throw new Error(`Failed to load server sample: ${sample.filename}`);
      const blob = await resp.blob();
      const file = new File([blob], sample.filename, { type: 'image/tiff' });

      const uploaded = await SatQueryAPI.uploadImage(file);
      attachImage(uploaded);
      onClose();
    } catch (err: any) {
      console.error('Error loading sample:', err);
      setError(err.message || 'Failed to ingest sample GeoTIFF');
    } finally {
      setLoadingId(null);
    }
  };

  const handleLoadPair = async (pair: MissionPairInfo) => {
    setLoadingId(pair.id);
    setError(null);

    try {
      for (const item of pair.files) {
        const resp = await fetch(`${item.url}?t=${Date.now()}`, { cache: 'no-store' });
        if (!resp.ok) throw new Error(`Failed to load: ${item.name}`);
        const blob = await resp.blob();
        const file = new File([blob], item.name, { type: 'image/tiff' });
        const uploaded = await SatQueryAPI.uploadImage(file);
        attachImage(uploaded);
      }

      // Pre-fill the query text so the user can review and send manually
      setPendingQueryText(pair.query);
      onClose();
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
      title="ISRO Remote Sensing Dataset Catalog"
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
                ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-[#cc785c]" />
            <span>Mission Pairs (1-Click Launch)</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('optical')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
              activeTab === 'optical'
                ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Satellite className="w-3.5 h-3.5 text-[#cc785c]" />
            <span>Optical Multispectral</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('sar')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
              activeTab === 'sar'
                ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Radio className="w-3.5 h-3.5 text-[#cc785c]" />
            <span>SAR Microwave Radar</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('disaster')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
              activeTab === 'disaster'
                ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            <Waves className="w-3.5 h-3.5 text-[#cc785c]" />
            <span>Disasters & Floods</span>
          </button>
        </div>

        {error && (
          <div className="p-3 bg-[#cc785c]/10 border border-[#cc785c]/30 rounded-xl text-[#cc785c] text-xs">
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
                  className="p-3.5 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] hover:border-[#cc785c]/40 hover:bg-[var(--bg-card)] transition-all shadow-subtle"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-2.5">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] flex items-center justify-center text-[#cc785c]">
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-xs text-[var(--text-main)]">
                          {pair.title}
                        </h4>
                        <span className="text-[10px] font-mono text-[#cc785c]">
                          {pair.category}
                        </span>
                      </div>
                    </div>

                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() => handleLoadPair(pair)}
                      className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#cc785c] hover:bg-[#b8674d] text-white font-medium text-xs shadow-sm transition-colors cursor-pointer disabled:opacity-50"
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
                          <span className="font-mono text-[9px] text-[#cc785c]">
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
                      ? 'border-[#cc785c]/40 bg-[#cc785c]/5 cursor-default'
                      : 'border-[var(--border-subtle)] hover:border-[#cc785c]/40 bg-[var(--bg-panel)] hover:bg-[var(--bg-card)] cursor-pointer shadow-subtle'
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
                      <span className="text-[10px] font-mono text-[#cc785c] block mt-1">
                        {sample.sensor} • {sample.details}
                      </span>
                    </div>
                  </div>

                  <div className="shrink-0 flex items-center justify-end">
                    {isLoading ? (
                      <div className="flex items-center gap-1.5 text-xs text-[#cc785c] font-mono">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Attaching...</span>
                      </div>
                    ) : isAlreadyAttached ? (
                      <div className="flex items-center gap-1 text-[11px] font-mono text-[#cc785c] bg-[#cc785c]/10 px-2.5 py-1 rounded-lg border border-[#cc785c]/25">
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
