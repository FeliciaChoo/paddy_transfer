"""Evidence base for the advisory: reference sources, and what every collected feature means for the paddy crop.

Every intervention rule in data/intervention_rules.csv has a `sources` column holding ids from REFERENCES. A source is cited only
for what it says; where no source supports an action, the rule says so. Sources were found by search in the course of building
this app; check the links (and the licence/terms of each publisher) before quoting them in an official document.
"""
import pandas as pd

# id -> (kind, citation, url)
REFERENCES = {
    # --- Malaysian agencies and extension material
    "M1": ("MARDI (via news report)", "MARDI, Manual Teknologi Penanaman Padi Lestari (Pakej Lestari Padi): land preparation, optimum seed rate, "
           "pest and disease control, supplementary fertiliser. Described in Sinar Harian, 'MARDI perkenal tujuh teknologi tingkat pengeluaran padi'.",
           "https://www.sinarharian.com.my/ampArticle/281589"),
    "M2": ("MARDI (library record)", "MARDI, Manual Teknologi Pengurusan Jerami dan Air Tanaman Padi: Strategi Adaptasi dan Mitigasi Perubahan Iklim "
           "(Serdang: Penerbit MARDI). Consult the manual itself for water-management detail.", "https://opac.kedahlib.gov.my/bib/196116"),
    "M3": ("Ministry reply (Dewan Rakyat)", "Written reply on integrated pest management (PPB) in paddy: MARDI-led PPB in granary areas; spraying only when pest "
           "populations exceed the economic threshold.",
           "https://pardocs.sinarproject.org/documents/2015-may-june-parliamentary-session/written-replies-soalan-bertulis/soalan-397.pdf"),
    "M4": ("MARDI (via news report)", "MARDI SOP for keladi agas (weed) control, built on integrated weed management; the weed can cause large yield loss if "
           "not controlled (RTM Berita).", "https://berita.rtm.gov.my/?p=177953"),
    # --- Water management
    "W1": ("IRRI-based factsheet", "Alternate wetting and drying (AWD): re-irrigate when the water table is about 15 cm below the soil surface, and keep the field "
           "flooded from about a week before to a week after flowering (IRRI Rice Knowledge Bank material, Assam).",
           "https://rkbassam.aau.ac.in/rkbassam.in/httpdocs/uploads/factsheets/pdf/6657ef1c105e4.pdf"),
    "W2": ("Peer-reviewed meta-analysis", "Rice yields and water use under alternate wetting and drying irrigation: a meta-analysis of 56 studies (mild AWD "
           "mostly kept yield; severe AWD lost about 23%).",
           "https://pure.psu.edu/en/publications/rice-yields-and-water-use-under-alternate-wetting-and-drying-irri/"),
    "W3": ("IRRI field trial", "Effect of water deficit stress on yield of wet-seeded rice: stress at heading/flowering was more damaging than early in the "
           "vegetative stage (Bangladesh Rice Journal).", "https://banglajol.info/index.php/BRJ/article/view/37359/25285"),
    # --- Growth stages
    "G1": ("IRRI-based factsheet", "Growth stages of the rice plant: vegetative, reproductive (about 35 days) and ripening (about 30 days) phases.",
           "https://rkbassam.aau.ac.in/uploads/factsheets/pdf/66576da6e47e2.pdf"),
    "G2": ("JICA training material", "Rice production training: the reproductive and ripening stages each last about 30 days.",
           "https://openjicareport.jica.go.jp/pdf/1000025769_04.pdf"),
    # --- Temperature
    "T1": ("Peer-reviewed review", "Influence of high-temperature stress on rice growth and development: spikelet sterility and yield loss; temperatures above "
           "33 C have reduced rice production in many places.", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9834771/"),
    "T2": ("Peer-reviewed experiment", "Asymmetric day/night warming on rice (Frontiers in Plant Science, 2017): night warming reduced pollen viability and grain "
           "yield; flowering is the stage most sensitive to heat.", "https://www.frontiersin.org/articles/10.3389/fpls.2017.01908/pdf"),
    "T3": ("Peer-reviewed review", "Heat stress at flowering and spikelet fertility (Frontiers in Plant Science, 2022): yields fell about 10% per 1 C rise in "
           "night-time minimum temperature in historical data.", "https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.983070/pdf"),
    "T4": ("Peer-reviewed review", "Impact of high-temperature stress in rice: temperatures above 35 C in the reproductive stage, especially at flowering, "
           "raise spikelet sterility.", "https://ojs.pphouse.org/index.php/IJEP/article/download/4637/2095/4212"),
    # --- Solar radiation
    "L1": ("Peer-reviewed (tropical irrigated rice)", "Yield gap between dry and wet season rice: lower radiation during ripening explained much of the lower wet-season "
           "yield.", "https://www.ukdr.uplb.edu.ph/journal-articles/2684"),
    "L2": ("Peer-reviewed", "Yield and morphophysiological indices of irrigated rice in contrasting ecosystems (Pesquisa Agropecuaria Tropical, 2017): more "
          "solar radiation in the reproductive and grain-filling phases raised yield.", "https://www.alice.cnptia.embrapa.br/alice/handle/doc/1075245"),
    "L3": ("Peer-reviewed", "Growth and yield of rice sown on different dates: adjust the planting date so the reproductive stage and grain filling do not fall "
           "mainly in low-radiation periods in the tropics.", "https://ojs-az.lib.uc.cl/index.php/ijanr/article/download/38401/29977/95953"),
    # --- Satellite indicators
    "S1": ("Peer-reviewed", "Xiao et al. (2005), Mapping paddy rice agriculture in southern China using multi-temporal MODIS images, Remote Sensing of "
           "Environment 95(4): flooding and transplanting are detected with LSWI together with EVI and NDVI.", "https://scholars.unh.edu/earthsci_facpub/315"),
    "S2": ("Conference paper", "Rice monitoring with Sentinel-1 on Google Earth Engine: VV and VH backscatter follow the growth stages of the crop.",
           "https://sciforum.net/paper/6206"),
    "S3": ("Peer-reviewed", "Xiao et al. (2021), Mapping paddy rice with Sentinel-1/2 and a phenology-based algorithm, Remote Sensing 13(5):990: transplanting "
           "is located from the EVI time series and flooding from radar and optical signals.",
           "https://mdpi-res.com/d_attachment/remotesensing/remotesensing-13-00990/article_deploy/remotesensing-13-00990.pdf"),
    # --- Resilience from satellite time series
    "R1": ("Peer-reviewed", "Resilience of vegetation to drought from a satellite NDVI time series, Remote Sensing of Environment 255 (2021) 112270: resilience is "
           "assessed as resistance to the disturbance and recovery after it.",
           "https://research-portal.uu.nl/en/publications/resilience-of-vegetation-to-drought-studying-the-effect-of-grazin/"),
    "R2": ("Peer-reviewed", "Changes in vegetation resistance and resilience under different drought disturbances, NDVI and SPEI time series, Remote Sensing "
           "15(13):3280 (2023): recovery speed and recovery time are read from NDVI anomalies.", "https://www.mdpi.com/2072-4292/15/13/3280"),
    # --- Soil
    "P1": ("Peer-reviewed review", "Chemistry of lowland rice soils and nutrient availability: flooding raises the pH of acid soils and lowers that of alkaline ones; "
           "iron and manganese become more soluble.", "https://www.bdpa.cnptia.embrapa.br/consulta/marc?id=897588"),
    "P2": ("IRRI-authored paper", "Iron toxicity of rice as a multiple soil stress (IRRI authors, JIRCAS TARS 15): iron toxicity depends on several soil factors, "
           "low pH among them.", "https://www.jircas.go.jp/sites/default/files/publication/tars/tars15-_167-179.pdf"),
    "P3": ("Practitioner summary", "Iron toxicity in rice (Plantix library): seen in flooded soils with low pH; liming, soil fertility (especially potassium) and drainage "
           "at certain stages are the usual measures.", "https://plantix.net/en/library/plant-diseases/900010/iron-toxicity-in-rice/"),
}


def reference_table():
    return pd.DataFrame([dict(ID=k, Type=v[0], Source=v[1], Link=v[2]) for k, v in REFERENCES.items()])


# what each collected feature means, how the app uses it, what can be done about it, and where that comes from
FEATURES = [
    # feature, what it measures, what it means for paddy, how the app uses it, rules, sources, limit
    ("NDVI", "Greenness of the canopy (optical satellite).", "Rises as the crop grows, peaks around heading, falls at ripening; the planting cycle is "
     "visible in it.", "Crop status (behind / on track / ahead), the 10- and 30-day forecast, stage estimate, and the resilience index.",
     "R01, R03, R06, R20, R21", "S1, S3, R1", "Cannot say WHY greenness differs; clouds leave gaps."),
    ("EVI", "A greenness index that saturates less over dense canopy.", "Used with LSWI to find flooding and transplanting; its time profile "
     "locates planting.", "Planting timing and stage estimate; model input.", "R03, R11", "S1, S3", "Same limits as NDVI."),
    ("LSWI", "Water content of the canopy and surface (shortwave infrared).", "Rises when fields are flooded and transplanted, so it marks the start "
     "of a crop.", "Model input; marks flooding and planting.", "none (a detection signal)", "S1", "Sources support detecting flooding, not a "
     "management action, so no rule uses it directly."),
    ("MNDWI", "Open-water signal.", "High over standing water, such as flooded fields at establishment.", "Model input; marks flooding.",
     "none (a detection signal)", "S1, S3", "Same as LSWI. Often missing in cloudy months."),
    ("SAR_VV_dB", "Radar backscatter (Sentinel-1, sees through cloud).", "Changes with flooding and with crop structure through the growth stages.",
     "Model input; keeps the forecast working when optical data is missing.", "none (a signal)", "S2, S3", "Hard to interpret by eye; a model input only."),
    ("Mean_Max_Temp_C", "Average daily maximum temperature in the 10-day period.", "Hot spells at flowering raise spikelet sterility; the literature "
     "gives about 33 C as a level where production starts to fall and above 35 C at flowering as clearly harmful.", "Heat flag when the recent "
     "average is 33 C or more.", "R14", "T1, T4", "A 10-day average hides short hot days."),
    ("Mean_Min_Temp_C", "Average daily minimum (night) temperature.", "Warm nights are linked to lower yield: about 10% per 1 C in historical "
     "data.", "Warm-night flag when it is far above this granary's normal for the time of year.", "R15", "T2, T3", "The 10% figure comes from "
     "other regions and seasons."),
    ("Mean_Temp_C", "Average temperature.", "General growing conditions.", "Context only.", "none", "T1", ""),
    ("Daily_Rain_mm (10-day total)", "Rainfall in the 10-day period.", "Water supply, and drainage risk when very high. Keeping the field "
     "flooded around flowering matters most; a water deficit then costs more yield than early in the season.", "Wet and dry flags when far from "
     "this granary's normal for the time of year.", "R17, R18", "W1, W3, M2", "Irrigated granaries depend on scheme water as well as rain."),
    ("Solar_Radiation_MJ", "Sunlight energy.", "Low radiation during the reproductive and ripening stages lowers yield; plan sowing so those "
     "stages avoid cloudy months.", "Low-sun flag during the reproductive or ripening stages.", "R16", "L1, L2, L3", "Reanalysis data is coarse."),
    ("Soil_pH", "Topsoil acidity (map-based estimate).", "Low pH in flooded soils is linked to iron toxicity; flooding itself raises the pH of "
     "acid soils. Liming and balanced potassium are the usual measures.", "A planning reminder to run a soil test where pH is below about 5.8.",
     "R19", "P1, P2, P3, W2", "A coarse estimate; confirm with a soil test before liming."),
    ("Stage, DAP, Cropping_Season, Month", "Estimated growth stage and days after planting, from the planting calendar.", "Vegetative, reproductive "
     "(about 35 days) and ripening (about 30 days) phases decide which actions make sense.", "Selects which rules apply (water, harvest "
     "logistics, next-cycle readiness).", "R02, R04, R05", "G1, G2", "Estimated from the NDVI cycle, not observed in the field."),
    ("NDVI_Growth_Velocity", "How fast NDVI is changing.", "A falling NDVI before maturity is not the normal pattern.", "Trend (rising / flat / "
     "falling) in the status and rule R06.", "R06", "S3", "Noisy where observations are missing."),
    ("Target_Yield", "Reported annual yield of the granary.", "The outcome the programme cares about.", "Yield maps, graph and the outlook.",
     "R08, R09", "none", "About 7 values per granary; the yield series of districts in one granary is identical."),
]


def feature_table():
    return pd.DataFrame(FEATURES, columns=["Feature", "What it measures", "What it means for paddy", "How the app uses it",
                                           "Rules", "Sources", "Limit"])
