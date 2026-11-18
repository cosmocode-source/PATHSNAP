/**
 * Leaflet Map Controller for PathSnap Evaluator.
 * Renders GPS fixes, classifier route, HMM route, and diagnostic point markers.
 * NEVER renders reference routes or per-point correctness flags.
 */

export class TripMap {
  constructor(containerId) {
    this.containerId = containerId;
    this.map = null;
    this.layerControl = null;
    this.layers = {
      gps: null,
      classifierRoute: null,
      hmmRoute: null,
      hmmChanged: null,
      gaps: null,
      stationary: null,
      endpoints: null,
    };
  }

  destroy() {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }
  }

  init(resultData) {
    this.destroy();

    const container = document.getElementById(this.containerId);
    if (!container) return;
    if (typeof L === "undefined") {
      container.textContent = "Map unavailable. Check your connection to the map library; accuracy results remain available below.";
      container.classList.add("map-unavailable");
      return;
    }

    // Create Leaflet map instance
    this.map = L.map(this.containerId, {
      zoomControl: true,
      attributionControl: false,
    });

    // OpenStreetMap is used directly so the map never depends on a CARTO API key.
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      className: "dark-map-tiles",
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(this.map);

    // Attribution
    L.control.attribution({ position: "bottomright", prefix: false })
      .addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors')
      .addTo(this.map);

    this.renderLayers(resultData);
  }

  renderLayers(res) {
    const gpsPoints = res.gps || [];
    const comparison = res.per_point_comparison || {};
    const step = res.display_downsample_step || 1;

    const clfArcIds = comparison.classifier_arc_ids || [];
    const hmmArcIds = comparison.hmm_arc_ids || [];
    const hmmChangedSet = new Set(comparison.hmm_changed_indices || []);
    const gapSet = new Set(comparison.after_gap_indices || []);
    const stationarySet = new Set(comparison.stationary_indices || []);
    const diagnosticPoints = res.diagnostic_points || {};
    const pointsFor = (key, selected) => diagnosticPoints[key] || gpsPoints.flatMap((p, i) => {
      const index = i * step;
      return selected.has(index) ? [[index, ...p]] : [];
    });

    // 1. Layer: GPS Observations
    const gpsMarkers = [];
    for (let i = 0; i < gpsPoints.length; i++) {
      const origIdx = i * step;
      const [lat, lon] = gpsPoints[i];
      const clfArc = clfArcIds[origIdx] != null ? clfArcIds[origIdx] : "N/A";
      const hmmArc = hmmArcIds[origIdx] != null ? hmmArcIds[origIdx] : "N/A";

      const circle = L.circleMarker([lat, lon], {
        radius: 3.5,
        fillColor: "#60A5FA",
        color: "#1E3A8A",
        weight: 1,
        fillOpacity: 0.85,
      });

      circle.bindTooltip(`GPS #${origIdx}`, { sticky: true, className: "map-tooltip" });
      circle.bindPopup(`
        <div class="map-popup">
          <strong>GPS Observation #${origIdx}</strong>
          <div>Classifier Arc: <code>${clfArc}</code></div>
          ${res.hmm_used ? `<div>HMM Arc: <code>${hmmArc}</code></div>` : ""}
          <div class="popup-subtext">Arc IDs are local to this trip's network</div>
        </div>
      `);
      gpsMarkers.push(circle);
    }
    this.layers.gps = L.layerGroup(gpsMarkers);

    // 2. Layer: Classifier-only Route (Amber, dashed)
    const clfSegments = (res.routes && res.routes.classifier && res.routes.classifier.segments) || [];
    const clfPolylines = clfSegments.map((seg) =>
      L.polyline(seg, {
        color: "#F59E0B",
        weight: 3.5,
        dashArray: "6, 6",
        opacity: 0.9,
      })
    );
    this.layers.classifierRoute = L.layerGroup(clfPolylines);

    // 3. Layer: HMM Route (Teal, solid, thicker)
    const hmmSegments = (res.routes && res.routes.hmm && res.routes.hmm.segments) || [];
    const hmmPolylines = hmmSegments.map((seg) =>
      L.polyline(seg, {
        color: "#2DD4BF",
        weight: 4.5,
        opacity: 0.95,
      })
    );
    this.layers.hmmRoute = L.layerGroup(hmmPolylines);

    // 4. Layer: "HMM changed this point" markers (violet rings)
    const changedMarkers = [];
    if (res.hmm_used) {
      for (const [origIdx, lat, lon] of pointsFor("hmm_changed_indices", hmmChangedSet)) {
          const ring = L.circleMarker([lat, lon], {
            radius: 7,
            color: "#A78BFA",
            weight: 2.5,
            fill: false,
            opacity: 1,
          });
          ring.bindTooltip(`HMM modified arc at #${origIdx}`, { className: "map-tooltip" });
          changedMarkers.push(ring);
      }
    }
    this.layers.hmmChanged = L.layerGroup(changedMarkers);

    // 5. Layer: Points after long gap (Purple)
    const gapMarkers = [];
    for (const [origIdx, lat, lon] of pointsFor("after_gap_indices", gapSet)) {
        const m = L.circleMarker([lat, lon], {
          radius: 5,
          fillColor: "#C084FC",
          color: "#7E22CE",
          weight: 1.5,
          fillOpacity: 0.9,
        });
        m.bindTooltip(`After GPS Gap (dt > 5s) at #${origIdx}`, { className: "map-tooltip" });
        gapMarkers.push(m);
    }
    this.layers.gaps = L.layerGroup(gapMarkers);

    // 6. Layer: Stationary points (Grey)
    const stationaryMarkers = [];
    for (const [origIdx, lat, lon] of pointsFor("stationary_indices", stationarySet)) {
        const m = L.circleMarker([lat, lon], {
          radius: 4,
          fillColor: "#94A3B8",
          color: "#334155",
          weight: 1,
          fillOpacity: 0.8,
        });
        m.bindTooltip(`Stationary cluster at #${origIdx}`, { className: "map-tooltip" });
        stationaryMarkers.push(m);
    }
    this.layers.stationary = L.layerGroup(stationaryMarkers);

    // 7. Layer: Start & End endpoints
    const endpointMarkers = [];
    if (gpsPoints.length > 0) {
      const startPt = res.gps_endpoints?.[0] || gpsPoints[0];
      const endPt = res.gps_endpoints?.[1] || gpsPoints[gpsPoints.length - 1];

      const startMarker = L.circleMarker(startPt, {
        radius: 8,
        fillColor: "#34D399",
        color: "#065F46",
        weight: 2.5,
        fillOpacity: 1,
      }).bindTooltip("Trip Start", { permanent: true, direction: "top", className: "map-badge-start" });

      const endMarker = L.circleMarker(endPt, {
        radius: 8,
        fillColor: "#F87171",
        color: "#991B1B",
        weight: 2.5,
        fillOpacity: 1,
      }).bindTooltip("Trip End", { permanent: true, direction: "top", className: "map-badge-end" });

      endpointMarkers.push(startMarker, endMarker);
    }
    this.layers.endpoints = L.layerGroup(endpointMarkers);

    // Add default layers to map
    this.layers.classifierRoute.addTo(this.map);
    if (res.hmm_used) {
      this.layers.hmmRoute.addTo(this.map);
      this.layers.hmmChanged.addTo(this.map);
    }
    this.layers.gps.addTo(this.map);
    this.layers.gaps.addTo(this.map);
    this.layers.stationary.addTo(this.map);
    this.layers.endpoints.addTo(this.map);

    // Layer toggle control
    const overlayMaps = {
      '<span class="legend-chip clf-line"></span> Classifier Route': this.layers.classifierRoute,
      '<span class="legend-chip gps-dot"></span> GPS Observations': this.layers.gps,
      '<span class="legend-chip gap-dot"></span> Long Gaps (dt &gt; 5s)': this.layers.gaps,
      '<span class="legend-chip stat-dot"></span> Stationary Points': this.layers.stationary,
      '<span class="legend-chip pin-dot"></span> Start / End': this.layers.endpoints,
    };

    if (res.hmm_used) {
      overlayMaps['<span class="legend-chip hmm-line"></span> HMM Route'] = this.layers.hmmRoute;
      overlayMaps['<span class="legend-chip changed-ring"></span> HMM Changed Point'] = this.layers.hmmChanged;
    }

    if (this.layerControl) {
      this.layerControl.remove();
    }
    this.layerControl = L.control.layers(null, overlayMaps, { collapsed: false, position: "topright" }).addTo(this.map);

    // Fit map bounds to GPS and route segments
    const allCoords = [];
    for (const p of gpsPoints) allCoords.push(p);
    for (const seg of clfSegments) for (const pt of seg) allCoords.push(pt);
    for (const seg of hmmSegments) for (const pt of seg) allCoords.push(pt);

    if (allCoords.length > 0) {
      this.map.fitBounds(L.latLngBounds(allCoords), { padding: [30, 30] });
    }
  }

  fitBounds() {
    if (this.map && this.layers.gps) {
      const bounds = L.latLngBounds([]);
      this.layers.gps.eachLayer((layer) => bounds.extend(layer.getLatLng()));
      if (bounds.isValid()) {
        this.map.fitBounds(bounds, { padding: [30, 30] });
      }
    }
  }
}
