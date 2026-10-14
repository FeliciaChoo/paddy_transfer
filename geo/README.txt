Administrative boundaries used by the dashboard map (dashboard/adminmap.py).

malaysia_districts.geojson, malaysia_states.geojson
    Source: https://github.com/nullifye/malaysia.geojson (district and state boundaries of Malaysia).
    Check that repository's licence / terms before publishing the app outside your organisation.

granary_districts.csv   (EDIT THIS)
    Which administrative district(s) belong to each granary. Taken from the granary-to-district table of this project.
    Granary names must match the 'District' column of the granary-level data (DailyGranary_by_granary.csv);
    District names must match the 'name' property in malaysia_districts.geojson.
    A granary missing from this file is drawn from its sampled-pixel outline instead.
