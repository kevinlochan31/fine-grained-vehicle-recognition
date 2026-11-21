"""Download freely licensed photos of ~180 modern car models from Wikimedia Commons (world-car layer).

For each target (make, model, generation start year) the script finds the Commons category of that
generation, lists its photos (and those of its plain sub-categories, skipping interiors, engines, details,
police/taxi/racing versions, models/toys), and downloads 500 px thumbnails. Licence, author and source URL
of every file go to <out>/photos.csv. Photos are selected and split later (scripts/build_world_gallery.py).

Output goes to WORK_DIR/world_cars (outside the project). A descriptive User-Agent names the project.
Usage:  python -m scripts.fetch_world_photos [--max-per-model 40]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from src import config as C

API = "https://commons.wikimedia.org/w/api.php"
UA = "AutoLens-WorldCars/1.1 (student computer-vision research project) python-requests"
OUT = C.WORK_DIR / "world_cars"
SKIP = re.compile(r"interior|engine|badge|logo|emblem|detail|dashboard|wheel|light|seat|police|taxi|ambulance|"
                  r"scale model|model car|\bmodels\b|toy|lego|\bart\b|race|racing|rally|competition|accident|crash|wreck|military|"
                  r"advert|brochure|drawing|prototype|concept|camouflag|tuning|modified|limousine|hearse|"
                  r"convoy|museum|production|factory|assembly|by colour|by color|in .* colou?r|colou?red", re.I)

# (make, model, generation start year, Commons category or None to search)
TARGETS = [
    # ---------------------------------------------------------------- India
    ("Maruti Suzuki", "Swift", 2017, "Suzuki Swift (2017)"), ("Maruti Suzuki", "Swift", 2024, "Suzuki Swift (2024)"),
    ("Maruti Suzuki", "Dzire", 2017, None), ("Maruti Suzuki", "Baleno", 2015, "Suzuki Baleno (2015)"),
    ("Maruti Suzuki", "Wagon R", 2019, None), ("Maruti Suzuki", "Alto K10", 2022, "Maruti Suzuki Alto (2022)"), ("Maruti Suzuki", "Celerio", 2014, "Suzuki Celerio (2nd generation)"),
    ("Maruti Suzuki", "Ertiga", 2018, "Suzuki Ertiga (2nd generation)"), ("Maruti Suzuki", "Vitara Brezza", 2016, None),
    ("Maruti Suzuki", "Brezza", 2022, None), ("Maruti Suzuki", "Ciaz", 2014, "Suzuki Ciaz"),
    ("Maruti Suzuki", "Grand Vitara", 2022, None), ("Maruti Suzuki", "Fronx", 2023, "Suzuki Fronx"),
    ("Maruti Suzuki", "Ignis", 2017, "Suzuki Ignis (2016)"), ("Maruti Suzuki", "S-Presso", 2019, None),
    ("Maruti Suzuki", "XL6", 2019, None), ("Maruti Suzuki", "Eeco", 2010, None), ("Maruti Suzuki", "Gypsy", 1985, None),
    ("Maruti Suzuki", "Omni", 1984, None), ("Maruti Suzuki", "800", 1983, None), ("Maruti Suzuki", "Zen", 1993, None),
    ("Tata", "Nexon", 2017, None), ("Tata", "Harrier", 2019, None), ("Tata", "Safari", 2021, None),
    ("Tata", "Tiago", 2016, None), ("Tata", "Tigor", 2017, None), ("Tata", "Altroz", 2020, None),
    ("Tata", "Punch", 2021, None), ("Tata", "Indica", 2008, "Tata Indica Vista"), ("Tata", "Nano", 2008, None),
    ("Tata", "Sumo", 1994, None), ("Tata", "Hexa", 2017, None), ("Tata", "Curvv", 2024, None),
    ("Mahindra", "Scorpio", 2002, "Mahindra Scorpio (1st generation)"), ("Mahindra", "Scorpio-N", 2022, "Mahindra Scorpio (2nd generation)"),
    ("Mahindra", "Bolero", 2000, None), ("Mahindra", "XUV500", 2011, None), ("Mahindra", "XUV700", 2021, None),
    ("Mahindra", "XUV300", 2019, None), ("Mahindra", "Thar", 2010, "Mahindra Thar (1st generation)"), ("Mahindra", "Thar", 2020, "Mahindra Thar (2nd generation)"),
    ("Mahindra", "Marazzo", 2018, None), ("Mahindra", "TUV300", 2015, None), ("Mahindra", "KUV100", 2016, None),
    ("Mahindra", "Xylo", 2009, None), ("Mahindra", "BE 6", 2025, None), ("Mahindra", "XEV 9e", 2025, None),
    ("Hyundai", "Creta", 2015, "Hyundai Creta (GS)"), ("Hyundai", "Creta", 2020, "Hyundai Creta (SU2)"),
    ("Hyundai", "Venue", 2019, None), ("Hyundai", "i20", 2020, None), ("Hyundai", "i20", 2014, "Hyundai i20 (GB)"),
    ("Hyundai", "Grand i10", 2013, "Hyundai i10 (BA)"), ("Hyundai", "Grand i10 Nios", 2019, "Hyundai i10 (AI3)"), ("Hyundai", "Verna", 2017, "Hyundai Verna (HC)"),
    ("Hyundai", "Verna", 2023, "Hyundai Verna (BN7)"), ("Hyundai", "Santro", 2018, "Hyundai Santro (AH2)"), ("Hyundai", "Alcazar", 2021, None),
    ("Hyundai", "Exter", 2023, None), ("Hyundai", "Aura", 2020, None), ("Hyundai", "Eon", 2011, None),
    ("Kia", "Seltos", 2019, None), ("Kia", "Sonet", 2020, None), ("Kia", "Carens", 2022, None), ("Kia", "Carnival", 2020, None),
    ("Honda", "City", 2014, "Honda City (6th generation)"), ("Honda", "City", 2020, "Honda City (7th generation)"), ("Honda", "Amaze", 2018, None),
    ("Honda", "Elevate", 2023, None), ("Honda", "WR-V", 2016, None), ("Honda", "Jazz", 2014, "Honda Jazz (3rd generation)"),
    ("Toyota", "Innova", 2004, "Toyota Innova (AN40)"), ("Toyota", "Innova Crysta", 2016, "Toyota Innova (AN140)"), ("Toyota", "Innova Hycross", 2022, "Toyota Innova (AG10)"),
    ("Toyota", "Fortuner", 2005, "Toyota Fortuner (AN50/AN60)"), ("Toyota", "Fortuner", 2015, "Toyota Fortuner (AN150/AN160)"),
    ("Toyota", "Etios", 2010, None), ("Toyota", "Glanza", 2022, None), ("Toyota", "Urban Cruiser Hyryder", 2022, None),
    ("MG", "Hector", 2019, None), ("MG", "Astor", 2021, None), ("MG", "ZS EV", 2019, None), ("MG", "Comet EV", 2023, None),
    ("Skoda", "Kushaq", 2021, None), ("Skoda", "Slavia", 2022, None),
    ("Skoda", "Kylaq", 2025, None), ("Volkswagen", "Taigun", 2021, None), ("Volkswagen", "Virtus", 2022, None),
    ("Volkswagen", "Polo", 2009, "Volkswagen Polo V"), ("Renault", "Kwid", 2015, None), ("Renault", "Triber", 2019, None),
    ("Renault", "Kiger", 2021, None), ("Renault", "Duster", 2010, None), ("Nissan", "Magnite", 2020, None),
    ("Citroen", "C3 (India)", 2022, None), ("Citroen", "Basalt", 2024, None), ("Jeep", "Compass", 2016, None),
    ("Jeep", "Meridian", 2022, None), ("Force", "Gurkha", 2020, None), ("Hindustan", "Ambassador", 1958, None),
    ("Premier", "Padmini", 1964, None), ("Datsun", "redi-GO", 2016, None), ("Chevrolet", "Beat", 2010, None),
    ("Chevrolet", "Tavera", 2004, None),
    # ---------------------------------------------------------------- Japan / Korea
    ("Toyota", "Corolla", 2018, None), ("Toyota", "Camry", 2017, None), ("Toyota", "RAV4", 2018, None),
    ("Toyota", "Hilux", 2015, None), ("Toyota", "Yaris", 2020, None), ("Toyota", "Prius", 2015, None),
    ("Toyota", "Prius", 2022, None), ("Toyota", "Land Cruiser", 2021, None), ("Toyota", "Corolla Cross", 2020, None),
    ("Toyota", "Alphard", 2015, None), ("Honda", "Civic", 2021, None), ("Honda", "Civic", 2015, None),
    ("Honda", "CR-V", 2016, None), ("Honda", "CR-V", 2022, None), ("Honda", "Accord", 2017, None), ("Honda", "HR-V", 2021, None),
    ("Nissan", "Qashqai", 2021, None), ("Nissan", "Leaf", 2017, None), ("Nissan", "X-Trail", 2022, None),
    ("Nissan", "Juke", 2019, None), ("Mazda", "CX-5", 2017, None), ("Mazda", "Mazda3", 2019, None),
    ("Mazda", "MX-5", 2015, None), ("Subaru", "Forester", 2018, None), ("Subaru", "Outback", 2019, None),
    ("Mitsubishi", "Outlander", 2021, None), ("Suzuki", "Jimny", 2018, None), ("Suzuki", "Vitara", 2015, None),
    ("Hyundai", "Tucson", 2020, None), ("Hyundai", "Ioniq 5", 2021, None), ("Hyundai", "Ioniq 6", 2022, None),
    ("Hyundai", "Kona", 2017, None), ("Hyundai", "Santa Fe", 2023, None), ("Kia", "EV6", 2021, None),
    ("Kia", "Sportage", 2021, None), ("Kia", "Niro", 2022, None), ("Kia", "Picanto", 2017, None),
    # ---------------------------------------------------------------- Europe
    ("Volkswagen", "Golf", 2019, "Volkswagen Golf VIII"), ("Volkswagen", "Golf", 2012, "Volkswagen Golf VII"),
    ("Volkswagen", "ID.4", 2020, None), ("Volkswagen", "Tiguan", 2016, None), ("Volkswagen", "T-Roc", 2017, None),
    ("Volkswagen", "Polo", 2017, "Volkswagen Polo VI"), ("Skoda", "Octavia", 2020, None), ("Skoda", "Enyaq", 2020, None),
    ("Audi", "A3", 2020, None), ("Audi", "Q5", 2017, None), ("Audi", "e-tron", 2018, None),
    ("BMW", "3 Series", 2019, "BMW G20"), ("BMW", "X5", 2018, "BMW G05"), ("BMW", "i4", 2021, None), ("BMW", "iX", 2021, None),
    ("Mercedes-Benz", "C-Class", 2021, "Mercedes-Benz W206"), ("Mercedes-Benz", "E-Class", 2016, "Mercedes-Benz W213"),
    ("Mercedes-Benz", "GLC", 2015, "Mercedes-Benz X253"), ("Porsche", "Taycan", 2019, None), ("Porsche", "Macan", 2014, None),
    ("Peugeot", "208", 2019, None), ("Peugeot", "3008", 2016, None), ("Renault", "Clio", 2019, None),
    ("Renault", "Captur", 2019, None), ("Dacia", "Sandero", 2020, None), ("Dacia", "Duster", 2017, None),
    ("Fiat", "500", 2007, None), ("Fiat", "Panda", 2011, None), ("Opel", "Corsa", 2019, None),
    ("Volvo", "XC40", 2017, None), ("Volvo", "XC60", 2017, None), ("Volvo", "XC90", 2014, None),
    ("Land Rover", "Defender", 2020, None), ("Land Rover", "Range Rover", 2022, None), ("Mini", "Hatch", 2013, None),
    ("Polestar", "2", 2019, "Polestar 2"),
    # ---------------------------------------------------------------- USA
    ("Tesla", "Model 3", 2017, None), ("Tesla", "Model Y", 2020, None), 
    ("Tesla", "Cybertruck", 2023, None), ("Ford", "F-150", 2021, None), ("Ford", "Mustang", 2015, None),
    ("Ford", "Mustang Mach-E", 2020, None), ("Ford", "Ranger", 2022, None), ("Ford", "Bronco", 2021, None),
    ("Ford", "Puma", 2019, None), ("Chevrolet", "Silverado", 2019, None), ("Chevrolet", "Corvette", 2020, None),
    ("Jeep", "Wrangler", 2018, "Jeep Wrangler (JL)"), ("Ram", "1500", 2019, None), ("Rivian", "R1T", 2021, None),
    # ---------------------------------------------------------------- China / SE Asia
    ("BYD", "Atto 3", 2022, None), ("BYD", "Seal", 2022, None), ("BYD", "Dolphin", 2021, None), ("BYD", "Han", 2020, None),
    ("BYD", "Seagull", 2023, None), ("Geely", "Coolray", 2018, None), ("MG", "MG4", 2022, "MG4 EV"),
    ("XPeng", "P7", 2020, None), ("NIO", "ES8", 2023, None), ("Li Auto", "L9", 2022, None), ("Zeekr", "001", 2021, None),
    ("Haval", "H6", 2021, None), ("Great Wall", "Poer", 2019, None), ("Chery", "Tiggo 8 Pro", 2020, None),
    ("Lynk & Co", "01", 2017, None), ("VinFast", "VF 8", 2022, None), ("Xiaomi", "SU7", 2024, None),
    ("Hongqi", "H9", 2020, None),
    # ---------------------------------------------------------------- India, round 2 (older generations, more models)
    *[(mk, mo, y, None) for mk, mo, y in [
        ("Maruti Suzuki", "Alto 800", 2012), ("Maruti Suzuki", "Alto", 2000), ("Maruti Suzuki", "Wagon R", 2010),
        ("Maruti Suzuki", "Wagon R", 2006), ("Maruti Suzuki", "Swift", 2005), ("Maruti Suzuki", "Swift", 2011),
        ("Maruti Suzuki", "Dzire", 2012), ("Maruti Suzuki", "Dzire", 2024), ("Maruti Suzuki", "Esteem", 1994),
        ("Maruti Suzuki", "Ritz", 2009), ("Maruti Suzuki", "SX4", 2007), ("Maruti Suzuki", "S-Cross", 2015),
        ("Maruti Suzuki", "Celerio", 2021), ("Maruti Suzuki", "Baleno", 2022), ("Maruti Suzuki", "Ertiga", 2012),
        ("Maruti Suzuki", "Jimny", 2023), ("Maruti Suzuki", "Invicto", 2023), ("Maruti Suzuki", "Estilo", 2009),
        ("Maruti Suzuki", "A-Star", 2008), ("Maruti Suzuki", "e Vitara", 2025), ("Maruti Suzuki", "1000", 1990),
        ("Tata", "Indica", 1998), ("Tata", "Indigo", 2002), ("Tata", "Safari", 1998), ("Tata", "Safari", 2005),
        ("Tata", "Safari Storme", 2012), ("Tata", "Nexon", 2023), ("Tata", "Nexon EV", 2020), ("Tata", "Harrier", 2023),
        ("Tata", "Tiago EV", 2022), ("Tata", "Punch EV", 2024), ("Tata", "Zest", 2014), ("Tata", "Bolt", 2014),
        ("Tata", "Sierra", 1991), ("Tata", "Sierra", 2025), ("Tata", "Ace", 2005), ("Tata", "Sumo Grande", 2008),
        ("Tata", "Aria", 2010), ("Tata", "Manza", 2009), ("Tata", "Estate", 1992), ("Tata", "Xenon", 2007),
        ("Mahindra", "Scorpio", 2014), ("Mahindra", "Scorpio Classic", 2022), ("Mahindra", "Bolero Neo", 2021),
        ("Mahindra", "Thar Roxx", 2024), ("Mahindra", "XUV 3XO", 2024), ("Mahindra", "XUV400", 2023),
        ("Mahindra", "Alturas G4", 2018), ("Mahindra", "Verito", 2010), ("Mahindra", "Logan", 2007),
        ("Mahindra", "Quanto", 2012), ("Mahindra", "Nuvosport", 2016), ("Mahindra", "e2o", 2013),
        ("Mahindra", "Reva", 2001), ("Mahindra", "Armada", 1993), ("Mahindra", "Commander", 1991),
        ("Mahindra", "Bolero Pik-Up", 2008), ("Mahindra", "Major", 1999),
        ("Hyundai", "Santro", 1998), ("Hyundai", "i10", 2007), ("Hyundai", "i20", 2008), ("Hyundai", "Accent", 1999),
        ("Hyundai", "Verna", 2011), ("Hyundai", "Getz", 2002), ("Hyundai", "Venue", 2025), ("Hyundai", "Xcent", 2014),
        ("Hyundai", "Elantra", 2015), ("Hyundai", "Tucson", 2015),
        ("Kia", "Syros", 2025), ("Kia", "Seltos", 2025),
        ("Honda", "City", 1996), ("Honda", "City", 2002), ("Honda", "City", 2008), ("Honda", "Amaze", 2013),
        ("Honda", "Amaze", 2024), ("Honda", "Brio", 2011), ("Honda", "BR-V", 2015), ("Honda", "Mobilio", 2014),
        ("Honda", "Jazz", 2020), ("Honda", "Civic", 2005),
        ("Toyota", "Qualis", 2000), ("Toyota", "Etios Liva", 2011), ("Toyota", "Corolla Altis", 2008),
        ("Toyota", "Urban Cruiser Taisor", 2024), ("Toyota", "Rumion", 2023), ("Toyota", "Glanza", 2019),
        ("Toyota", "Camry", 2011), ("Toyota", "Land Cruiser Prado", 2009),
        ("MG", "Gloster", 2020), ("MG", "Windsor EV", 2024), ("MG", "Hector", 2023), ("MG", "Hector Plus", 2020),
        ("Skoda", "Rapid", 2011), ("Skoda", "Laura", 2004), ("Skoda", "Superb", 2015), ("Skoda", "Kodiaq", 2016),
        ("Skoda", "Fabia", 2007), ("Volkswagen", "Vento", 2010), ("Volkswagen", "Ameo", 2016),
        ("Renault", "Pulse", 2012), ("Renault", "Scala", 2012), ("Renault", "Lodgy", 2012),
        ("Nissan", "Micra", 2010), ("Nissan", "Sunny", 2011), ("Nissan", "Terrano", 2013), ("Nissan", "Kicks", 2016),
        ("Ford", "EcoSport (India)", 2013), ("Ford", "Figo", 2010), ("Ford", "Figo", 2015), ("Ford", "Endeavour", 2015),
        ("Ford", "Ikon", 1999), ("Ford", "Fiesta (India)", 2005), ("Ford", "Freestyle", 2018), ("Ford", "Figo Aspire", 2015),
        ("Chevrolet", "Spark (India)", 2007), ("Chevrolet", "Cruze", 2008), ("Chevrolet", "Sail", 2012),
        ("Chevrolet", "Aveo U-VA", 2006), ("Chevrolet", "Enjoy", 2013), ("Chevrolet", "Optra", 2003),
        ("Fiat", "Palio", 1996), ("Fiat", "Linea", 2007), ("Fiat", "Grande Punto", 2005), ("Fiat", "Uno", 1983),
        ("Hindustan", "Contessa", 1984), ("Premier", "118NE", 1985), ("Premier", "Rio", 2009),
        ("Datsun", "GO", 2014), ("Datsun", "GO+", 2015), ("Force", "Gurkha", 2008), ("Force", "Trax Cruiser", 2008),
        ("Force", "Urbania", 2024), ("Citroen", "C3 (India) Aircross", 2023), ("Citroen", "eC3", 2023),
    ]],
]

S = requests.Session()
S.headers["User-Agent"] = UA


def api(**p):
    p.update(action="query", format="json", formatversion=2)
    for k in range(8):
        try:
            r = S.get(API, params=p, timeout=60)
        except requests.RequestException:
            time.sleep(10 * (k + 1)); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (k + 1)); continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Commons API keeps failing")


def members(cat, kind="subcat|file", limit=500):
    out, cont = [], {}
    while True:
        j = api(list="categorymembers", cmtitle=f"Category:{cat}", cmlimit=limit, cmtype=kind, **cont)
        out += j.get("query", {}).get("categorymembers", [])
        if "continue" not in j or len(out) >= 1500:
            return out
        cont = {"cmcontinue": j["continue"]["cmcontinue"]}


def exists(cat):
    j = api(prop="categoryinfo", titles=f"Category:{cat}")
    pg = j["query"]["pages"][0]
    return not pg.get("missing") and pg.get("categoryinfo", {}).get("size", 0) > 0


ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
ORDW = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}


# --auto-from: makes sold in India (plus the big global ones); trims that look like their base car are skipped
AUTO_MAKES = {"Maruti Suzuki", "Tata", "Mahindra", "Hyundai", "Kia", "Honda", "Toyota", "MG", "Skoda", "Volkswagen",
              "Renault", "Nissan", "Citroen", "Jeep", "Force", "BYD", "Isuzu", "Mercedes-Benz", "BMW", "Audi", "Volvo",
              "Land Rover", "Jaguar", "Lexus", "Porsche", "Mini", "Ford", "Chevrolet", "Fiat", "Datsun", "Mitsubishi",
              "Suzuki", "Lamborghini", "Ferrari"}
TRIM = re.compile(r"\b(hybrid|phev|plug-in|n line|n|gt|gti|gts|turbo|electric|ev|e-tron|sport|sports|plus|max|pro|amg|"
                  r"rs\d*|s\d|m\d+|type r|nismo|tdi|diesel|cng|lwb|long|cabrio|cabriolet|convertible|coupe|"
                  r"estate|wagon|touring|sportback|black|dark|edition|x-line|gt-line|trophy|tomahawk|classic)\b", re.I)
VARIANTS = {}        # (make, model) -> names of other catalogue models that extend this one ("Thar Roxx", "Harrier EV")


def other_model(title, make, model):
    """True if a category / file title names a different catalogue model that extends this one
    (e.g. "Mahindra Thar ROXX" while looking for the Thar, "Tata Harrier EV" while looking for the Harrier)."""
    t = title.lower()
    return any(v.lower() in t for v in VARIANTS.get((make, model), ()))


def ordinal(tail):
    """Generation number written in a sub-category name: "(2nd generation)", "(second generation)", " II"."""
    m = re.search(r"\((\d+)(?:st|nd|rd|th) generation\)", tail, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\((\w+) generation\)", tail, re.I)
    if m and m.group(1).lower() in ORDW:
        return ORDW[m.group(1).lower()]
    m = re.fullmatch(r"\s+([IVX]+)", tail)
    return ROMAN.get(m.group(1)) if m else None


def search_files(make, model, y0, y1, strict, limit=150):
    """Commons full-text file search, for models whose category holds few photos (files are often filed
    elsewhere, e.g. under a motor-show category). Title must name make and model and no other variant."""
    mdl = model.replace(" (India)", "")
    mk = "Suzuki" if make == "Maruti Suzuki" else make
    out = []
    for q in dict.fromkeys([f"{make} {mdl}", f"{mk} {mdl}"]):
        j = api(list="search", srsearch=f'intitle:"{mdl}" {q}', srnamespace=6, srlimit=limit)
        for h in j.get("query", {}).get("search", []):
            t = h["title"]
            low = t.lower()
            if mdl.lower() not in low or mk.lower().split()[-1] not in low or SKIP.search(t) or other_model(t, make, model):
                continue
            y = title_year(t)
            if (y is None and strict) or (y is not None and not y0 - 1 <= y <= y1 + 1):
                continue
            out.append(t)
    return list(dict.fromkeys(out))


def resolve(make, model, start, gen_no, n_gens, y0, y1):
    """Find the Commons category of this generation. Returns (category, how) or (None, reason)."""
    makes = ["Suzuki", "Maruti Suzuki", "Maruti"] if make == "Maruti Suzuki" else ["Citroën", "Citroen"] if make == "Citroen" else [make]
    mdl = model.replace(" (India)", "")
    base = None
    for mk in makes:
        for cand in (f"{mk} {mdl}", f"{mk} {mdl.replace('-', ' ')}"):
            if exists(cand):
                base = cand; break
        if base:
            break
    if base is None:                                     # fall back to category search
        j = api(list="search", srsearch=f"{makes[0]} {mdl}", srnamespace=14, srlimit=10)
        for h in j["query"]["search"]:
            t = h["title"].split(":", 1)[1]
            if mdl.lower() in t.lower() and not SKIP.search(t) and not other_model(t, make, model):
                base = t; break
    if base is None:
        return None, "no category"
    subs = [m["title"].split(":", 1)[1] for m in members(base, "subcat")]
    gen_subs = []
    for s in subs:
        if SKIP.search(s[len(base):]) or not s.lower().startswith(base.lower()) or other_model(s, make, model):
            continue
        tail = s[len(base):]
        ys = [int(y) for y in re.findall(r"^\s*\(((?:19|20)\d\d)", tail)]
        if ys and abs(ys[0] - start) <= 1:
            return s, "generation sub-category (year in name)"
        if n_gens > 1 and ordinal(tail) == gen_no:
            return s, "generation sub-category (ordinal in name)"
        if re.match(r"\s*\(", tail) or re.fullmatch(r"\s+[IVX]+", tail):
            gen_subs.append(s)
    # Date every generation-like sub-category (chassis code, "(2nd generation)", roman numeral) by the years
    # in its file names and pick the one whose photos fall inside this generation's years.
    best, best_score = None, 0.0
    for s in gen_subs:
        ys = [title_year(t) for t in files_of(s)[:300]]
        ys = [y for y in ys if y]
        if len(ys) < 4:
            continue
        inside = float(np.mean([y0 - 1 <= y <= y1 + 1 for y in ys]))
        # prefer the generation whose earliest photos start near the generation's first year
        score = inside - 0.05 * abs(float(np.percentile(ys, 10)) - start)
        if inside >= 0.5 and score > best_score:
            best, best_score = s, score
    if best:
        return best, f"generation sub-category (dated {best_score:.0%})"
    if not gen_subs and n_gens <= 1:
        return base, "model category (single generation)"
    if not gen_subs:
        return base, "model category (files filtered by year)"
    return None, f"generation not found under {base}"


def title_year(t):
    m = re.search(r"(?<!\d)(19[5-9]\d|20[0-2]\d)(?!\d)", t)
    return int(m.group(1)) if m else None


def files_of(cat, depth=1, mm=None):
    out = [m["title"] for m in members(cat, "file")]
    if depth:
        for s in members(cat, "subcat"):
            t = s["title"].split(":", 1)[1]
            if not SKIP.search(t[len(cat):] or t) and not (mm and other_model(t, *mm)):
                out += files_of(t, depth - 1, mm)
    if mm:
        out = [f for f in out if not other_model(f, *mm)]
    return out


def strip(h):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", h or "")).split())


def info(titles):
    out = []
    for i in range(0, len(titles), 50):
        j = api(prop="imageinfo", titles="|".join(titles[i:i + 50]), iiprop="url|size|sha1|extmetadata|mime",
                iiurlwidth=500, iiextmetadatafilter="LicenseShortName|Artist|LicenseUrl|DateTimeOriginal|UsageTerms")
        for pg in j["query"]["pages"]:
            ii = (pg.get("imageinfo") or [None])[0]
            if ii and ii.get("mime") == "image/jpeg":
                md = ii.get("extmetadata", {})
                out.append(dict(title=pg["title"], thumb=ii.get("thumburl"), width=ii["width"], height=ii["height"],
                                sha1=ii["sha1"], source=ii["descriptionurl"],
                                license=strip(md.get("LicenseShortName", {}).get("value")),
                                license_url=md.get("LicenseUrl", {}).get("value", ""),
                                author=strip(md.get("Artist", {}).get("value"))[:200],
                                date=strip(md.get("DateTimeOriginal", {}).get("value"))[:40]))
    return out


def download(row, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for k in range(4):
        try:
            r = S.get(row["thumb"], timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (k + 1)); continue
            r.raise_for_status()
            dest.write_bytes(r.content); return True
        except requests.RequestException:
            time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-model", type=int, default=14)
    ap.add_argument("--retry-below", type=int, default=0,
                    help="re-fetch finished models that got fewer photos than this (or none)")
    ap.add_argument("--auto-from", type=int, default=0,
                    help="also target every catalogue generation of AUTO_MAKES starting in or after this year "
                         "(trim variants that look like their base car are left out)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="download nothing; just write photos.csv from the models finished so far")
    a = ap.parse_args()
    cat = json.loads((C.META_DIR / "world_catalogue.json").read_text(encoding="utf8"))["entries"]
    targets = list(TARGETS)
    if a.auto_from:
        have = {(m, mo, s) for m, mo, s, _ in targets}
        targets += [(e["make"], e["model"], e["years"][0], None) for e in cat
                    if e["make"] in AUTO_MAKES and (e["years"][0] or 0) >= a.auto_from
                    and (e["make"], e["model"], e["years"][0]) not in have and not TRIM.search(e["model"])]
        # cars sold in India first, so a stopped run still has the ones the app is most often shown
        first = ["Maruti Suzuki", "Suzuki", "Tata", "Mahindra", "MG", "Toyota", "Hyundai", "Kia", "Honda",
                 "Skoda", "Volkswagen", "Renault", "Nissan"]
        targets.sort(key=lambda t: first.index(t[0]) if t[0] in first else len(first))
    names = {}
    for e in cat:
        names.setdefault(e["make"], set()).add(e["model"])
    for mk, ms in names.items():
        for m in ms:
            VARIANTS[(mk, m)] = [o for o in ms if o.lower().startswith(m.lower() + " ")]
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    rows, log = [], []
    seen_sha = set()
    # resumable: each finished model is appended to progress.jsonl (its photos already downloaded)
    prog = OUT / "progress.jsonl"
    done = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf8").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:   # half-written last line of a stopped run
                continue
            done[d["target"]] = d
    if a.retry_below and not a.no_fetch:   # entries already made by this version (v2) are not retried again
        done = {k: d for k, d in done.items() if len(d["rows"]) >= a.retry_below or d.get("v") == 2}
    for make, model, start, cat_name in targets:
        tkey = f"{make}|{model}|{start}"
        if tkey in done:
            rows += done[tkey]["rows"]; log.append(done[tkey]["log"])
            seen_sha.update(r["sha1"] for r in done[tkey]["rows"]); continue
        if a.no_fetch:
            continue
        gens = [e for e in cat if e["make"] == make and e["model"] == model]
        g = next((e for e in gens if e["years"][0] == start), None)
        if g is None:
            log.append(f"SKIP {make} {model} {start}: not in catalogue"); continue
        gen_no = int(re.match(r"\d+", g["gen"]).group()) if g["gen"] else 1
        y0, y1 = g["years"][0], g["years"][1] or 2026
        if cat_name and exists(cat_name):
            c, how = cat_name, "given"
        else:
            c, how = resolve(make, model, start, gen_no, len(gens), y0, y1)
        titles = []
        if c is not None:
            titles = list(dict.fromkeys(files_of(c, mm=(make, model))))
            # drop photos whose file name dates them outside this generation (+-1 model year); in a mixed model
            # category keep only photos dated inside the generation
            strict = "filtered by year" in how
            titles = [t for t in titles if (title_year(t) is None and not strict)
                      or (title_year(t) is not None and y0 - 1 <= title_year(t) <= y1 + 1)]
        if len(titles) < a.max_per_model:
            # few photos in the category (or none found): add a file-title search. With several generations
            # a file must carry a year inside this one.
            extra = search_files(make, model, y0, y1, strict=len(gens) > 1)
            if extra:
                titles = list(dict.fromkeys(titles + extra))
                how = (how if c else "no category") + f" + file search ({len(extra)})"
                c = c or "file search"
        if c is None:
            log.append(f"MISS {make} {model} {start}: {how}"); print(log[-1], flush=True)
            with open(prog, "a", encoding="utf8") as f:
                f.write(json.dumps(dict(target=tkey, rows=[], log=log[-1], v=2)) + "\n")
            continue
        inf = [r for r in info(titles[:300]) if min(r["width"], r["height"]) >= 400 and r["thumb"] and r["sha1"] not in seen_sha]
        # spread over authors: round-robin so one prolific photographer does not fill the quota
        by_author = {}
        for r in inf:
            by_author.setdefault(r["author"], []).append(r)
        pick, lists = [], list(by_author.values())
        while len(pick) < a.max_per_model and any(lists):
            for L in lists:
                if L and len(pick) < a.max_per_model:
                    pick.append(L.pop(0))
        key = re.sub(r"[^a-z0-9]+", "_", f"{make}_{model}_{start}".lower()).strip("_")
        for r in pick:
            seen_sha.add(r["sha1"])
            r.update(key=key, make=make, model=model, gen=g["gen"], years=g["years_label"], category=c,
                     file=f"img/{key}__{r['sha1'][:12]}.jpg")
        with ThreadPoolExecutor(4) as ex:
            ok = list(ex.map(lambda r: download(r, OUT / r["file"]), pick))
        pick = [r for r, k in zip(pick, ok) if k]
        rows += pick
        log.append(f"OK   {make} {model} {start} -> {c} ({how}): {len(inf)} files, {len(by_author)} authors, took {len(pick)}")
        print(log[-1], flush=True)
        with open(prog, "a", encoding="utf8") as f:
            f.write(json.dumps(dict(target=tkey, rows=pick, log=log[-1], v=2)) + "\n")
    cols = ["key", "make", "model", "gen", "years", "file", "title", "source", "license", "license_url", "author",
            "date", "sha1", "width", "height", "category"]
    with open(OUT / "photos.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (OUT / "fetch_log.txt").write_text("\n".join(log), encoding="utf8")
    print(f"downloaded {len(rows)} photos for {len({r['key'] for r in rows})} models")


if __name__ == "__main__":
    main()
