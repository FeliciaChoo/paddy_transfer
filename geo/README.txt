Administrative boundaries used by the dashboard map (dashboard/adminmap.py).

malaysia_districts.geojson, malaysia_states.geojson
    Source: https://github.com/nullifye/malaysia.geojson (district and state boundaries of Malaysia).
    Check that repository's licence / terms before publishing the app outside your organisation.

granary_districts.csv   (EDIT THIS)
    Which administrative district(s) belong to each granary. Taken from the granary-to-district table of this project.
    Granary names must match the 'District' column of the granary-level data (DailyGranary_by_granary.csv);
    District names must match the 'name' property in malaysia_districts.geojson.
    A granary missing from this file is drawn from its sampled-pixel outline instead.

paddy_planmalaysia.geojson
    Paddy polygons from PLANMalaysia (ArcGIS REST: scharms.planmalaysia.gov.my ... /DPFDN/AlamSekitar/MapServer/7, queried as GeoJSON in WGS84).
    Fields used: negeri (state), jelapang (Dalam = inside a granary, Luar = outside), nmskim (scheme name, empty outside granaries), luhek (area, ha).
    Peninsular Malaysia only, no date or season field. Check the data owner's terms before publishing the app outside your organisation.
    Used by pipeline/paddy_mask.py. The dashboard map does not read it.

granary_scheme_map.csv   (EDIT THIS)
    Which scheme name(s) in paddy_planmalaysia.geojson belong to each granary of this project. Use 'unnamed' for polygons that are inside a granary
    but carry no scheme name. Several rows per granary are allowed. A granary missing from this file gets In_Own_Scheme = <NA> (unknown, never False).
    Two rows are assumptions to confirm with the data owner: Seberang Perak = IADP Seberang Perak + Sungai Manik, and Pekan/Rompin sharing the unnamed Pahang polygons.

