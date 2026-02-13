#!/usr/bin/env python3
"""
Fetch OSM roads for a bbox, create a buffered road mask, export shapefiles.
No GeoPandas. Tiled Overpass queries + tqdm progress.

Deps:
  pip install requests shapely pyproj fiona tqdm

Notes:
- bbox is WGS84: south, west, north, east
- Overpass query pulls ways + referenced nodes via (._;>;); out body;
- Geometries built with Shapely; export with Fiona; reprojection with pyproj.
"""

import os
import random
import time

import fiona
import requests
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import unary_union
from tqdm import tqdm

# =========================
# GLOBALS (EDIT THESE)
# =========================

# BBOX in WGS84 degrees (south, west, north, east)
SOUTH = 53.46
WEST = 8.69
NORTH = 54.9
EAST = 10.85

OUT_DIR = "roads_db"

# Overpass instances to try (fallback if one is overloaded)
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# HTTP timeout for requests (seconds)
HTTP_TIMEOUT_S = 240

# Overpass query header knobs (server may still cap these)
QUERY_TIMEOUT_S = 180
QUERY_MAXSIZE = 512 * 1024 * 1024  # 512 MiB

# Tiling (degrees). Reduce (e.g., 0.10) if you still get timeouts.
TILE_DEG = 0.20

# Retries per tile per endpoint on overload/timeouts
RETRIES = 4

# Keep only these highway categories (edit to your definition of "road")
# (If you want "all highway=*" set KEEP_HIGHWAY = None)
KEEP_HIGHWAY = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "living_street",
}
# Typical non-road paths you may want excluded if you use KEEP_HIGHWAY=None
EXCLUDE_HIGHWAY = {"footway", "path", "cycleway", "steps", "corridor", "pedestrian"}

# Mask buffer in meters (half-width around the road centerline)
BUFFER_M = 4.0

# Output CRS: if empty, pick local UTM from bbox center; else set e.g. "EPSG:25833"
TARGET_CRS = ""

# Output filenames (in OUT_DIR)
ROADS_LINES_SHP = "roads_lines.shp"
ROADS_MASK_SHP = "roads_mask.shp"

# =========================
# IMPLEMENTATION
# =========================


def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """Pick a UTM EPSG code for the lon/lat point."""
    zone = int((lon + 180) / 6) + 1
    epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
    return f"EPSG:{epsg}"


def _tile_bboxes(south: float, west: float, north: float, east: float, tile_deg: float):
    """Yield (s, w, n, e) tiles covering the bbox."""
    lat = south
    while lat < north:
        lat2 = min(lat + tile_deg, north)
        lon = west
        while lon < east:
            lon2 = min(lon + tile_deg, east)
            yield (lat, lon, lat2, lon2)
            lon = lon2
        lat = lat2


def _make_overpass_query(s: float, w: float, n: float, e: float) -> str:
    """
    Build an Overpass QL query that returns only the highway categories you want,
    and includes referenced nodes for way geometry via (._;>;); out body;.
    """
    if KEEP_HIGHWAY is not None:
        # Server-side filter: only the road classes you keep
        keep_re = "|".join(sorted(KEEP_HIGHWAY))
        way_sel = f'way["highway"~"^({keep_re})$"]({s},{w},{n},{e});'
    else:
        # Keep all highway ways but optionally exclude some types
        if EXCLUDE_HIGHWAY:
            excl_re = "|".join(sorted(EXCLUDE_HIGHWAY))
            way_sel = f'way["highway"]["highway"!~"^({excl_re})$"]({s},{w},{n},{e});'
        else:
            way_sel = f'way["highway"]({s},{w},{n},{e});'

    return f"""
[out:json][timeout:{QUERY_TIMEOUT_S}][maxsize:{QUERY_MAXSIZE}];
(
  {way_sel}
);
(._;>;);
out body;
""".strip()


def _post_overpass(url: str, query: str) -> dict:
    r = requests.post(url, data={"data": query}, timeout=HTTP_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def overpass_query_roads_tiled(
    south: float, west: float, north: float, east: float
) -> dict:
    """
    Query roads by tiling the bbox. Dedupes elements by (type,id).
    Returns a dict like {"elements": [...]} compatible with build_ways_as_lines().
    """
    tiles = list(_tile_bboxes(south, west, north, east, TILE_DEG))
    elements = []
    seen = set()  # (type, id)

    for s, w, n, e in tqdm(tiles, desc="Overpass tiles", unit="tile"):
        query = _make_overpass_query(s, w, n, e)

        last_err = None
        for base_url in OVERPASS_URLS:
            for attempt in range(RETRIES):
                try:
                    data = _post_overpass(base_url, query)
                    for el in data.get("elements", []):
                        key = (el.get("type"), el.get("id"))
                        if key not in seen:
                            seen.add(key)
                            elements.append(el)
                    last_err = None
                    break
                except requests.HTTPError as err:
                    last_err = err
                    status = getattr(err.response, "status_code", None)
                    # Retry on typical overload/timeouts
                    if status in (429, 502, 503, 504):
                        time.sleep((2**attempt) + random.random())
                        continue
                    raise
            if last_err is None:
                break

        if last_err is not None:
            raise last_err

    return {"elements": elements}


def build_ways_as_lines(overpass_json: dict):
    """
    Convert Overpass JSON elements into Shapely LineStrings.
    Returns: list of dicts: {"osmid": int, "highway": str, "name": str|None, "geometry": LineString}
    """
    elements = overpass_json.get("elements", [])

    # Index nodes by id
    nodes = {}
    ways = []
    for el in elements:
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el.get("type") == "way":
            ways.append(el)

    features = []
    for w in ways:
        tags = w.get("tags", {})
        highway = tags.get("highway")
        if not highway:
            continue

        # If server-side filter is off, filter here as well
        if KEEP_HIGHWAY is not None:
            if highway not in KEEP_HIGHWAY:
                continue
        else:
            if highway in EXCLUDE_HIGHWAY:
                continue

        node_ids = w.get("nodes", [])
        coords = [nodes.get(nid) for nid in node_ids]
        coords = [c for c in coords if c is not None]

        if len(coords) < 2:
            continue

        geom = LineString(coords)
        features.append(
            {
                "osmid": int(w["id"]),
                "highway": str(highway),
                "name": tags.get("name"),
                "geometry": geom,
            }
        )

    return features


def reproject_geom(geom, transformer: Transformer):
    """Reproject a shapely geometry using a pyproj Transformer (always_xy=True recommended)."""
    if geom.geom_type == "LineString":
        xs, ys = zip(*list(geom.coords))
        X, Y = transformer.transform(xs, ys)
        return LineString(list(zip(X, Y)))
    elif geom.geom_type == "MultiLineString":
        parts = []
        for ls in geom.geoms:
            xs, ys = zip(*list(ls.coords))
            X, Y = transformer.transform(xs, ys)
            parts.append(LineString(list(zip(X, Y))))
        return MultiLineString(parts)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def write_lines_shp(path: str, crs_epsg: str, line_features):
    schema = {
        "geometry": "LineString",
        "properties": {"osmid": "int", "highway": "str:40", "name": "str:80"},
    }
    with fiona.open(
        path,
        "w",
        driver="ESRI Shapefile",
        crs=CRS.from_user_input(crs_epsg).to_wkt(),
        schema=schema,
    ) as sink:
        for f in line_features:
            g = f["geometry"]
            if g.geom_type == "MultiLineString":
                for part in g.geoms:
                    sink.write(
                        {
                            "geometry": mapping(part),
                            "properties": {
                                "osmid": f["osmid"],
                                "highway": f["highway"],
                                "name": (f["name"] or "")[:80],
                            },
                        }
                    )
            else:
                sink.write(
                    {
                        "geometry": mapping(g),
                        "properties": {
                            "osmid": f["osmid"],
                            "highway": f["highway"],
                            "name": (f["name"] or "")[:80],
                        },
                    }
                )


def write_mask_shp(path: str, crs_epsg: str, mask_geom):
    schema = {
        "geometry": "Polygon",
        "properties": {"id": "int"},
    }
    with fiona.open(
        path,
        "w",
        driver="ESRI Shapefile",
        crs=CRS.from_user_input(crs_epsg).to_wkt(),
        schema=schema,
    ) as sink:
        if mask_geom.is_empty:
            return
        if mask_geom.geom_type == "Polygon":
            sink.write({"geometry": mapping(mask_geom), "properties": {"id": 1}})
        elif mask_geom.geom_type == "MultiPolygon":
            i = 1
            for poly in mask_geom.geoms:
                sink.write({"geometry": mapping(poly), "properties": {"id": i}})
                i += 1
        else:
            # unary_union can return GeometryCollection; write polygons only
            geoms = getattr(mask_geom, "geoms", [])
            i = 1
            for g in geoms:
                if g.geom_type == "Polygon":
                    sink.write({"geometry": mapping(g), "properties": {"id": i}})
                    i += 1
                elif g.geom_type == "MultiPolygon":
                    for poly in g.geoms:
                        sink.write({"geometry": mapping(poly), "properties": {"id": i}})
                        i += 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Fetch from Overpass (tiled)
    data = overpass_query_roads_tiled(SOUTH, WEST, NORTH, EAST)

    # 2) Build shapely lines in WGS84
    feats = build_ways_as_lines(data)
    if not feats:
        raise RuntimeError(
            "No road features found after filtering. Adjust bbox or KEEP_HIGHWAY / TILE_DEG."
        )

    # 3) Choose output CRS (meters) for buffering
    center_lat = (SOUTH + NORTH) / 2.0
    center_lon = (WEST + EAST) / 2.0
    out_crs = TARGET_CRS.strip() or utm_epsg_from_lonlat(center_lon, center_lat)

    # 4) Reproject WGS84 -> out_crs using always_xy=True (lon,lat order)
    tf = Transformer.from_crs("EPSG:4326", out_crs, always_xy=True)

    proj_feats = []
    for f in feats:
        g_proj = reproject_geom(f["geometry"], tf)
        proj_feats.append({**f, "geometry": g_proj})

    # 5) Write road lines shapefile
    roads_path = os.path.join(OUT_DIR, ROADS_LINES_SHP)
    write_lines_shp(roads_path, out_crs, proj_feats)

    # 6) Build road mask: buffer each line by BUFFER_M and dissolve
    buffered = [f["geometry"].buffer(BUFFER_M) for f in proj_feats]
    mask = unary_union(buffered)

    # 7) Write road mask shapefile
    mask_path = os.path.join(OUT_DIR, ROADS_MASK_SHP)
    write_mask_shp(mask_path, out_crs, mask)

    print("Done.")
    print("Road lines:", roads_path)
    print("Road mask :", mask_path)
    print("CRS used  :", out_crs)


if __name__ == "__main__":
    main()
