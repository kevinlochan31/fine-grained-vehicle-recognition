"""World-car layer, extra data: Indian street photos from FGVD (Khoba et al., "A Fine-Grained Vehicle
Detection (FGVD) Dataset for Unconstrained Roads", ICVGIP 2022; Zenodo record 7488960, CC BY 4.0).

FGVD labels every vehicle box in IDD street scenes as <type>_<Manufacturer>_<Model>, e.g. car_MarutiSuzuki_Dzire.
This script crops every car box (8% context), maps the label to a world-catalogue model, and picks the
generation:
  * only generations launched before IDD was recorded (start year <= 2019) are candidates (old ones still drive);
  * one candidate -> that generation;
  * several -> the current world-car layer chooses among them (softmax restricted to those rows), and the crop is
    kept only if it is >= 0.6 sure (otherwise the generation label would be a guess).
Crops smaller than MIN_SIDE px or labelled Others / Covered are skipped.

FGVD train + val crops become reference (training) photos; FGVD test crops are kept apart as a street-photo test
set (set "test_fgvd"), so the Commons test numbers stay comparable. Rows are appended to
WORK_DIR/world_cars/photos_fgvd.csv; scripts/build_world_gallery.py merges them.

Usage:  python -m scripts.import_fgvd --root C:/Users/Kevin/cvh/fgvd/IDD_FGVD
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import config as C

W = C.WORK_DIR / "world_cars"
MIN_SIDE = 80
REC_YEAR = 2019            # IDD was recorded in 2017-2018; allow one model year
CITE = "Khoba, Parikh, Saluja, Sarvadevabhatla, Jawahar (FGVD, ICVGIP 2022)"
SRC = "https://zenodo.org/records/7488960"
MAKES = {"MarutiSuzuki": "Maruti Suzuki", "TataMotors": "Tata", "Bmw": "BMW"}
ALIAS = {("Maruti Suzuki", "esteem2000"): "Esteem", ("Maruti Suzuki", "1000"): "1000", ("Honda", "brv"): "BR-V",
         ("Honda", "wrv"): "WR-V", ("Honda", "crv"): "CR-V", ("Ford", "ecosporttitanium"): "EcoSport",
         ("Ford", "ecosport"): "EcoSport", ("Force", "traxtoofan"): "Trax", ("Mercedes-Benz", "glaclass"): "GLA",
         ("Mercedes-Benz", "eclass"): "E-Class", ("Mercedes-Benz", "cclass"): "C-Class",
         ("Mercedes-Benz", "gclass"): "G-Class", ("BMW", "3series"): "3 Series", ("BMW", "2series220d"): "2 Series",
         ("Fiat", "puntoevo"): "Punto Evo", ("Toyota", "corolla"): "Corolla Altis"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def boxes(root: Path):
    for split in ("train", "val", "test"):
        for x in sorted((root / split / "annos").glob("*.xml")):
            t = ET.parse(x).getroot()
            img = root / split / "images" / t.findtext("filename", x.stem + ".jpg")
            if not img.exists():
                img = next((root / split / "images").glob(x.stem + ".*"), None)
            if img is None:
                continue
            for i, o in enumerate(t.iter("object")):
                name = o.findtext("name", "")
                bb = o.find("bndbox")
                if bb is None or not name.startswith("car_"):
                    continue
                yield split, img, i, name, [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    cat = json.loads((C.META_DIR / "world_catalogue.json").read_text(encoding="utf8"))["entries"]
    models = {}
    for e in cat:
        models.setdefault((e["make"], norm(e["model"])), []).append(e)
    from src.worldcars import WorldCars, describe_world   # current world-car layer chooses the generation
    WCm = WorldCars(device="cuda")
    out_dir = W / "img_fgvd"; out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], Counter()
    for split, img, i, name, (x1, y1, x2, y2) in boxes(root):
        _, mk, md = (name.split("_", 2) + ["", ""])[:3]
        if mk in ("Others", "Covered", "") or not md:
            skipped["other/covered"] += 1; continue
        make = MAKES.get(mk, mk)
        key_m = ALIAS.get((make, norm(md)))
        gens = models.get((make, norm(key_m) if key_m else norm(md)))
        if not gens:
            skipped[f"unmapped {name}"] += 1; continue
        cands = [g for g in gens if (g["years"][0] or 0) <= REC_YEAR]          # old cars still drive in India
        if not cands:
            skipped["no generation on sale"] += 1; continue
        w, h = x2 - x1, y2 - y1
        if min(w, h) < MIN_SIDE:
            skipped["too small"] += 1; continue
        im = Image.open(img).convert("RGB")
        px, py = 0.08 * w, 0.08 * h
        crop = im.crop((max(0, x1 - px), max(0, y1 - py), min(im.width, x2 + px), min(im.height, y2 + py)))
        arr = np.asarray(crop)
        if len(cands) == 1:
            g, how = cands[0], "single generation on sale"
        else:
            ks = [WCm._find_class(c["make"], c["model"], c["years"][0]) for c in cands]
            ok = [(k, c) for k, c in zip(ks, cands) if k is not None]
            if not ok:
                skipped["no reference generation to choose from"] += 1; continue
            f = WCm.embed(arr, None)
            l = np.array([float(f @ WCm.W[k] + WCm.b[k]) for k, _ in ok]); p = np.exp(l - l.max()); p /= p.sum()
            j = int(p.argmax())
            if len(ok) > 1 and p[j] < 0.6:
                skipped["generation unclear"] += 1; continue
            g, how = ok[j][1], f"generation chosen by the world layer ({p[j]:.2f})"
        y0 = g["years"][0]
        key = re.sub(r"[^a-z0-9]+", "_", f"{g['make']} {g['model']} {y0}".lower()).strip("_")
        data = crop.tobytes()
        sha = hashlib.sha1(f"{img.name}:{i}".encode() + data[:4096]).hexdigest()
        fn = f"img_fgvd/{key}__{sha[:12]}.jpg"
        crop.save(W / fn, quality=92)
        years = f"{y0}–{g['years'][1] or 'present'}"
        rows.append(dict(key=key, make=g["make"], model=g["model"], gen=g["gen"], years=years, file=fn,
                         title=f"FGVD {split}/{img.name} box {i} ({name})", source=SRC, license="CC BY 4.0",
                         license_url="https://creativecommons.org/licenses/by/4.0/", author=f"FGVD: {CITE}",
                         date="2018", sha1=sha, width=crop.width, height=crop.height, category=how,
                         preset="ref" if split in ("train", "val") else "test_fgvd"))
    df = pd.DataFrame(rows)
    df.to_csv(W / "photos_fgvd.csv", index=False)
    print(f"{len(df)} car crops kept ({(df.preset == 'ref').sum()} reference, {(df.preset == 'test_fgvd').sum()} test) "
          f"for {df.key.nunique()} generations")
    print(df.groupby(["make", "model", "years"]).size().sort_values(ascending=False).head(40).to_string())
    for k, v in skipped.most_common(40):
        print(f"skipped {v:5d}  {k}")


if __name__ == "__main__":
    main()
