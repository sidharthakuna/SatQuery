import React, { useEffect, useRef, useState } from 'react';
import { Navigation } from 'lucide-react';
import type { GeoBoundsLatLon } from '../../types/api';
import L from 'leaflet';

interface MapViewProps {
  bounds?: GeoBoundsLatLon | null;
  fileId?: string | null;
  filename?: string;
  height?: string;
}

export const MapView: React.FC<MapViewProps> = ({
  bounds,
  fileId,
  filename = 'Satellite Scene',
  height = '340px',
}) => {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const [mouseCoords, setMouseCoords] = useState<{ lat: number; lon: number } | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    let centerLat = 23.0225;
    let centerLon = 72.5714;
    let initialZoom = 12;

    const isValidLat = (lat?: number) => typeof lat === 'number' && !isNaN(lat) && lat >= -90 && lat <= 90;
    const isValidLon = (lon?: number) => typeof lon === 'number' && !isNaN(lon) && lon >= -180 && lon <= 180;
    const hasValidBounds = Boolean(
      bounds &&
      isValidLat(bounds.min_lat) &&
      isValidLat(bounds.max_lat) &&
      isValidLon(bounds.min_lon) &&
      isValidLon(bounds.max_lon) &&
      bounds.max_lat >= bounds.min_lat &&
      bounds.max_lon >= bounds.min_lon
    );

    if (hasValidBounds && bounds) {
      centerLat = (bounds.min_lat + bounds.max_lat) / 2;
      centerLon = (bounds.min_lon + bounds.max_lon) / 2;
    }

    if (mapInstanceRef.current) {
      mapInstanceRef.current.remove();
      mapInstanceRef.current = null;
    }

    const map = L.map(mapContainerRef.current, {
      center: [centerLat, centerLon],
      zoom: initialZoom,
      zoomControl: true,
      attributionControl: false,
    });

    const isLightMode = document.documentElement.classList.contains('light');
    const tileServerUrl = isLightMode
      ? 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

    L.tileLayer(tileServerUrl, {
      maxZoom: 19,
      subdomains: 'abcd',
    }).addTo(map);

    if (fileId) {
      const tileUrl = `/api/v1/tiles/${fileId}/{z}/{x}/{y}.png`;
      L.tileLayer(tileUrl, {
        maxZoom: 18,
        tileSize: 256,
        opacity: 0.85,
      }).addTo(map);
    }

    if (hasValidBounds && bounds) {
      const southWest = L.latLng(bounds.min_lat, bounds.min_lon);
      const northEast = L.latLng(bounds.max_lat, bounds.max_lon);
      const latLngBounds = L.latLngBounds(southWest, northEast);

      L.rectangle(latLngBounds, {
        color: '#cc785c',
        weight: 1.5,
        fillColor: '#cc785c',
        fillOpacity: 0.08,
        dashArray: '4, 4',
      }).addTo(map);

      map.fitBounds(latLngBounds, { padding: [25, 25] });

      L.circleMarker([centerLat, centerLon], {
        radius: 5,
        color: '#ffffff',
        weight: 1.5,
        fillColor: '#cc785c',
        fillOpacity: 1,
      })
        .addTo(map)
        .bindTooltip(filename, { permanent: false, direction: 'top' });
    }

    map.on('mousemove', (e: L.LeafletMouseEvent) => {
      setMouseCoords({
        lat: Number(e.latlng.lat.toFixed(5)),
        lon: Number(e.latlng.lng.toFixed(5)),
      });
    });

    map.on('mouseout', () => {
      setMouseCoords(null);
    });

    mapInstanceRef.current = map;

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [bounds, fileId, filename]);

  return (
    <div className="w-full rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-card)] my-3 select-none shadow-subtle">
      <div className="flex items-center justify-between px-3.5 py-2 bg-[var(--bg-panel)] border-b border-[var(--border-subtle)] text-xs">
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <Navigation className="w-3.5 h-3.5 text-[#cc785c]" />
          <span className="font-medium text-[var(--text-main)]">Map View</span>
        </div>

        {mouseCoords && (
          <span className="text-[10px] font-mono text-[var(--text-muted)]">
            LAT: {mouseCoords.lat}° • LON: {mouseCoords.lon}°
          </span>
        )}
      </div>

      <div ref={mapContainerRef} style={{ height }} className="w-full relative z-0" />

      {bounds &&
        typeof bounds.min_lat === 'number' &&
        !isNaN(bounds.min_lat) &&
        bounds.min_lat >= -90 &&
        bounds.max_lat <= 90 && (
        <div className="px-3.5 py-2 bg-[var(--bg-panel)] border-t border-[var(--border-subtle)] grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono text-[var(--text-muted)]">
          <div className="bg-[var(--bg-card)] p-2 rounded-lg border border-[var(--border-subtle)]">
            <span className="text-[var(--text-dim)] block text-[9px] uppercase tracking-wider">MIN LAT</span>
            <span className="text-[var(--text-main)] font-semibold">{bounds.min_lat.toFixed(4)}°</span>
          </div>
          <div className="bg-[var(--bg-card)] p-2 rounded-lg border border-[var(--border-subtle)]">
            <span className="text-[var(--text-dim)] block text-[9px] uppercase tracking-wider">MAX LAT</span>
            <span className="text-[var(--text-main)] font-semibold">{bounds.max_lat.toFixed(4)}°</span>
          </div>
          <div className="bg-[var(--bg-card)] p-2 rounded-lg border border-[var(--border-subtle)]">
            <span className="text-[var(--text-dim)] block text-[9px] uppercase tracking-wider">MIN LON</span>
            <span className="text-[var(--text-main)] font-semibold">{bounds.min_lon.toFixed(4)}°</span>
          </div>
          <div className="bg-[var(--bg-card)] p-2 rounded-lg border border-[var(--border-subtle)]">
            <span className="text-[var(--text-dim)] block text-[9px] uppercase tracking-wider">MAX LON</span>
            <span className="text-[var(--text-main)] font-semibold">{bounds.max_lon.toFixed(4)}°</span>
          </div>
        </div>
      )}
    </div>
  );
};
