"""Build offline Folium maps for selected training trips, with reference routes."""
import json
import os
import sys
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import folium
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'trained_models'))
sys.path.insert(0, str(ROOT / 'data_preprocessing/data_preprocessing'))
from common.metrics import point_ids, top_rows
from pathsnap_prep import load_trip, load_route
from trained_models.hmm.hmm_decoder import decode_trip

metrics = pd.read_csv(HERE / 'results/per_trip_metrics.csv', dtype={'trip_id': str})
m = metrics[(metrics.partition == 'train') & (metrics.method == 'hmm')].set_index('trip_id')
b = metrics[(metrics.partition == 'train') & (metrics.method == 'classifier')].set_index('trip_id')
choices = {'clean': m.sort_values(['far_share', 'gap_share']).index[0],
           'gap_heavy': m.gap_share.idxmax(), 'no_help': (m.top1_all-b.top1_all).idxmin()}
raw = pd.read_parquet(ROOT / 'data_preprocessing/cleaned_data/model_data/train_candidates.parquet')
raw = raw.sort_values(['trip_id', 'gps_row'], kind='stable').reset_index(drop=True)
oof = pd.read_parquet(HERE / 'results/oof_predictions.parquet')
keys = ['trip_id', 'point_id', 'gps_row', 'arc_id']
if not raw[keys].equals(oof[keys]):
    raise ValueError('OOF keys do not match training candidate rows')
raw['probability'] = oof.probability
cfg = json.loads((HERE / 'config.json').read_text())
(HERE / 'maps').mkdir(exist_ok=True)
for kind, trip_id in choices.items():
    part = raw[raw.trip_id == trip_id].reset_index(drop=True)
    trip = load_trip(ROOT / 'Dataset', 'train', trip_id)
    ref = load_route(ROOT / 'Dataset', 'train', trip_id, trip.n_arcs)
    pred = part.arc_id.to_numpy()[top_rows(part.probability.to_numpy(), point_ids(part))]
    decoded = decode_trip(part[['gps_row', 'arc_id']], part.probability.to_numpy(), trip, trip.gps_xy, trip.gps_t, cfg)['arc_per_point']
    xy = trip.gps_lonlat[:, ::-1]
    mp = folium.Map(location=xy.mean(axis=0).tolist(), zoom_start=15, tiles='OpenStreetMap')
    tf = Transformer.from_crs(trip.epsg, 4326, always_xy=True)
    # Restrict the grey context to a GPS bounding box plus a 100-m margin.
    lo, hi = trip.gps_xy.min(axis=0)-100, trip.gps_xy.max(axis=0)+100
    keep = np.all(np.maximum(trip.arc_a, trip.arc_b) >= lo, axis=1) & np.all(np.minimum(trip.arc_a, trip.arc_b) <= hi, axis=1)
    def draw_arcs(ids, layer, color, weight, dash=None):
        layer_obj = folium.FeatureGroup(name=layer)
        for arc in ids:
            xx, yy = tf.transform([trip.arc_a[arc, 0], trip.arc_b[arc, 0]], [trip.arc_a[arc, 1], trip.arc_b[arc, 1]])
            folium.PolyLine(list(zip(yy, xx)), color=color, weight=weight, opacity=.8, dash_array=dash).add_to(layer_obj)
        layer_obj.add_to(mp)
    nearby = np.flatnonzero(keep)
    # Bound HTML size on unusually large networks; the displayed context is a
    # deterministic sample, while predicted and reference arcs remain complete.
    if len(nearby) > 10000:
        nearby = nearby[np.linspace(0, len(nearby)-1, 10000, dtype=int)]
    draw_arcs(nearby, 'Nearby network (up to 10000 arcs)', '#999999', 1)
    draw_arcs(ref, 'Reference route', 'green', 4)
    draw_arcs(pred[np.r_[True, pred[1:] != pred[:-1]]], 'Classifier route', '#d99000', 4, '6 5')
    draw_arcs(decoded[np.r_[True, decoded[1:] != decoded[:-1]]], 'HMM route', 'teal', 3)
    gps = folium.FeatureGroup(name='Raw GPS'); folium.PolyLine(xy.tolist(), color='blue', weight=2).add_to(gps); gps.add_to(mp)
    changed = folium.FeatureGroup(name='Changed points')
    for p in xy[pred != decoded]:
        folium.CircleMarker(p.tolist(), radius=4, color='violet', fill=False).add_to(changed)
    changed.add_to(mp)
    gaps = folium.FeatureGroup(name='Gap points')
    for p in xy[np.r_[False, np.diff(trip.gps_t) > cfg['gap_seconds']]]:
        folium.CircleMarker(p.tolist(), radius=3, color='purple', fill=True).add_to(gaps)
    gaps.add_to(mp)
    mp.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px">Blue: GPS · grey: roads · green: reference · amber: classifier · teal: HMM · violet: changed · purple: gap</div>'))
    folium.LayerControl().add_to(mp)
    mp.fit_bounds([[float(xy[:,0].min()), float(xy[:,1].min())], [float(xy[:,0].max()), float(xy[:,1].max())]])
    mp.save(str(HERE / f'maps/train_{kind}_{trip_id}.html'))
    print(f'Map {kind}: {trip_id}', flush=True)
