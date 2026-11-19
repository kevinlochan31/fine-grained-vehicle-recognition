"""Build data_meta/world_catalogue.json: ~3,000 make / model / generation entries up to 2026.

Each entry: {"make", "model", "gen", "years": [start, end or null], "label"}. The source table below is
compact: per make, "Model: ranges" where ranges are comma-separated "yy-yy" spans (yy >= 50 means 19yy,
an open end such as "24-" means still on sale in 2026). "N@" before the first span starts the
generation count at N when older generations are left out. Generation labels are ordinal ("3rd gen").
The 954-entry routing catalogue (data_meta/car_catalogue.json) is merged in so this file is a superset;
its models without known generations get one entry with no year range.

This catalogue is used only by the world-car layer (naming cars outside the 196 Stanford classes). It
never changes the 196-class models or the open-set routing, which keep using car_catalogue.json.

Usage:  python -m scripts.build_world_catalogue
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

T = {
# ----------------------------------------------------------------------------------------------- India
"Maruti Suzuki": """800: 83-14; Alto: 00-12; Alto 800: 12-19, 19-23; Alto K10: 10-14, 14-20, 22-; S-Presso: 19-;
 Celerio: 14-21, 21-; Wagon R: 99-06, 06-10, 10-19, 19-; Zen: 93-06; Zen Estilo: 07-13; Ritz: 09-16; A-Star: 08-14;
 Swift: 05-11, 11-17, 17-24, 24-; Dzire: 08-12, 12-17, 17-24, 24-; Baleno: 99-07, 15-22, 22-; Ignis: 17-; Esteem: 94-08;
 Gypsy: 85-18; Omni: 84-19; Eeco: 10-; Versa: 01-10; Ertiga: 12-18, 18-; XL6: 19-; Ciaz: 14-; Vitara Brezza: 16-22;
 Brezza: 22-; S-Cross: 15-22; SX4: 07-14; Kizashi: 11-14; Grand Vitara: 22-; Jimny: 23-; Fronx: 23-; Invicto: 23-;
 e Vitara: 25-; Victoris: 25-""",
"Tata": """Indica: 98-08, 08-18; Indigo: 02-09, 09-18; Indigo CS: 08-18; Nano: 08-18; Sumo: 94-04; Sumo Victa: 04-11;
 Sumo Gold: 11-19; Safari: 98-05, 05-17, 21-; Safari Storme: 12-19; Aria: 10-17; Hexa: 17-20; Zest: 14-19; Bolt: 14-19;
 Tiago: 16-; Tigor: 17-; Nexon: 17-; Nexon EV: 20-; Harrier: 19-; Harrier EV: 25-;
 Altroz: 20-; Punch: 21-; Punch EV: 24-; Tiago EV: 22-; Tigor EV: 19-; Curvv: 24-; Curvv EV: 24-;
 Sierra: 91-03, 25-; Xenon: 07-16; Estate: 92-00; Indigo Marina: 06-10; Spacio: 00-11""",
"Mahindra": """Scorpio: 02-14, 14-22; Scorpio-N: 22-; Scorpio Classic: 22-; Bolero: 00-; Bolero Neo: 21-; XUV500: 11-21;
 XUV700: 21-25; XUV 7XO: 26-; XUV300: 19-24; XUV 3XO: 24-; XUV400: 23-; Thar: 10-20, 20-; Thar Roxx: 24-;
 Marazzo: 18-23; TUV300: 15-20; KUV100: 16-21; Xylo: 09-19; Quanto: 12-16; Verito: 10-19; Logan: 07-10; e2o: 13-17;
 Alturas G4: 18-22; BE 6: 25-; XEV 9e: 25-; XEV 9S: 25-; Armada: 93-01; Major: 99-10; Nuvosport: 16-18; Supro: 15-""",
"Force": "Gurkha: 08-20, 20-; Trax Cruiser: 08-; One: 11-15; Urbania: 24-",
"Hindustan": "Ambassador: 58-14; Contessa: 84-02",
"Premier": "Padmini: 64-01; 118NE: 85-01; Rio: 09-17",
# ----------------------------------------------------------------------------------------------- Korea
"Hyundai": """Santro: 98-14, 18-22; i10: 07-13, 13-19; Grand i10: 13-19; Grand i10 Nios: 19-; i20: 08-14, 14-20, 20-;
 Accent: 94-99, 99-05, 05-11, 11-17, 17-22; Verna: 06-11, 11-17, 17-23, 23-; Elantra: 90-95, 95-00, 00-06, 06-10,
 10-15, 15-20, 20-; Sonata: 98-04, 04-09, 09-14, 14-19, 19-; Creta: 15-20, 20-; Creta Electric: 25-; Venue: 19-25, 25-;
 Alcazar: 21-; Exter: 23-; Aura: 20-; Xcent: 14-20; Eon: 11-19; Getz: 02-11; Tucson: 04-09, 09-15, 15-20, 20-;
 Santa Fe: 00-06, 06-12, 12-18, 18-23, 23-; Kona: 17-23, 23-; Ioniq: 16-22; Ioniq 5: 21-; Ioniq 6: 22-; Ioniq 9: 25-;
 Palisade: 18-25, 25-; Veloster: 11-17, 18-22; Genesis Coupe: 08-16; Azera: 05-11, 11-17; Grandeur: 17-22, 22-;
 Staria: 21-; Starex: 97-07, 07-21; Casper: 21-; Bayon: 21-; i30: 07-11, 11-16, 16-; i40: 11-19; ix20: 10-19;
 Santa Cruz: 21-; Matrix: 01-10; Terracan: 01-07; Veracruz: 06-15; Equus: 09-16; Atos: 97-08; Inster: 24-""",
"Kia": """Seltos: 19-25, 25-; Sonet: 20-; Carens: 99-06, 06-12, 13-19, 22-; Carnival: 98-05, 05-14, 14-20, 20-; EV6: 21-;
 EV9: 23-; EV3: 24-; EV4: 25-; EV5: 23-; Syros: 25-; Picanto: 04-11, 11-17, 17-; Rio: 00-05, 05-11, 11-17, 17-;
 Ceed: 06-12, 12-18, 18-; Sportage: 93-04, 04-10, 10-15, 15-21, 21-; Sorento: 02-09, 09-14, 14-20, 20-; Telluride: 19-;
 Stinger: 17-23; Niro: 16-22, 22-; Soul: 08-13, 13-19, 19-; Forte: 08-13, 13-18, 18-; Optima: 00-05, 05-10, 10-15,
 15-20; K5: 20-; Stonic: 17-; XCeed: 19-; Clavis: 25-; Mohave: 08-; K8: 21-; Ray: 11-; Cadenza: 09-16, 16-21;
 K900: 12-18, 18-; Venga: 09-19; Pride: 87-00; Spectra: 00-09; PV5: 25-; Tasman: 25-""",
"Genesis": "G70: 17-; G80: 16-20, 20-; G90: 15-22, 22-; GV60: 21-; GV70: 20-; GV80: 20-; GV80 Coupe: 24-",
"SsangYong": """Rexton: 01-17, 17-; Korando: 96-06, 10-19, 19-; Tivoli: 15-; Musso: 93-05, 18-; Actyon: 05-18, 24-;
 Kyron: 05-14; Rodius: 04-19; Torres: 22-; Chairman: 97-17""",
# ----------------------------------------------------------------------------------------------- Japan
"Toyota": """Innova: 04-15; Innova Crysta: 16-; Innova Hycross: 22-; Fortuner: 05-15, 15-; Corolla: 7@91-95, 95-00, 00-06,
 06-12, 12-18, 18-; Corolla Cross: 20-; Camry: 3@91-96, 96-01, 01-06, 06-11, 11-17, 17-24, 24-; Etios: 10-20;
 Etios Liva: 11-20; Glanza: 19-22, 22-; Urban Cruiser: 20-22; Urban Cruiser Hyryder: 22-; Urban Cruiser Taisor: 24-;
 Rumion: 23-; Qualis: 00-04; Land Cruiser: 98-07, 07-21, 21-; Land Cruiser Prado: 02-09, 09-23, 23-;
 Land Cruiser 70: 84-; RAV4: 94-00, 00-05, 05-12, 12-18, 18-25, 25-; Hilux: 6@97-05, 05-15, 15-; Yaris: 99-05, 05-11,
 11-20, 20-; Yaris Cross: 20-; Prius: 97-03, 03-09, 09-15, 15-22, 22-; C-HR: 16-23, 23-; Highlander: 00-07, 07-13,
 13-19, 19-; Grand Highlander: 23-; 4Runner: 3@95-02, 02-09, 09-24, 24-; Tacoma: 95-04, 04-15, 15-23, 23-;
 Tundra: 99-06, 06-21, 21-; Sienna: 97-03, 03-10, 10-20, 20-; Supra: 4@93-02, 19-; GR86: 12-21, 21-;
 Avalon: 94-99, 99-04, 04-12, 12-18, 18-22; Vellfire: 08-15, 15-23, 23-; Alphard: 02-08, 08-15, 15-23, 23-;
 Sequoia: 00-07, 07-22, 22-; Venza: 08-17, 20-24; bZ4X: 22-; Crown: 16@22-; Aygo: 05-14, 14-22; Aygo X: 22-;
 Auris: 06-12, 12-18; Avanza: 03-11, 11-21, 21-; Rush: 06-17, 17-; Vios: 02-07, 07-13, 13-22, 22-; FJ Cruiser: 06-22;
 GR Yaris: 20-; HiAce: 5@04-19, 19-; Mirai: 14-20, 20-; Celica: 6@93-99, 99-06; MR2: 3@99-07; Matrix: 02-08, 08-14;
 Estima: 90-00, 00-06, 06-19; Noah: 01-07, 07-14, 14-22, 22-; Harrier: 97-03, 03-13, 13-20, 20-; bZ3: 22-;
 Raize: 19-; Land Cruiser 250: 24-; Belta: 22-; Starlet: 20-; Corolla Altis: 9@01-08, 08-13, 13-19""",
"Lexus": """ES: 5@06-12, 12-18, 18-25, 25-; IS: 2@05-13, 13-; LS: 4@06-17, 17-; RX: 2@03-09, 09-15, 15-22, 22-;
 NX: 14-21, 21-; UX: 18-; GX: 09-23, 23-; LX: 2@07-21, 21-; LC: 17-; LM: 20-23, 23-; RZ: 22-; LBX: 23-; RC: 14-;
 CT: 10-22; LFA: 10-12; GS: 3@05-11, 12-20; SC: 2@01-10; HS: 09-18; TX: 23-""",
"Honda": """City: 3@96-02, 02-08, 08-13, 14-20, 20-; Amaze: 13-18, 18-24, 24-; Jazz: 01-08, 08-14, 14-20, 20-;
 Civic: 5@91-95, 95-00, 00-05, 05-11, 11-15, 15-21, 21-; Accord: 4@89-93, 93-97, 97-02, 02-07, 07-12, 12-17, 17-22, 22-;
 CR-V: 95-01, 01-06, 06-11, 11-16, 16-22, 22-; HR-V: 98-06, 13-21, 21-; WR-V: 16-23; Elevate: 23-; Brio: 11-18;
 BR-V: 15-21, 21-; Mobilio: 14-17; Pilot: 02-08, 08-15, 15-22, 22-; Odyssey: 98-04, 04-10, 10-17, 17-; Passport: 18-25,
 25-; Ridgeline: 05-14, 16-; Insight: 99-06, 09-14, 18-22; e: 20-24; Prologue: 24-; ZR-V: 22-; Prelude: 5@96-01, 25-;
 S2000: 99-09; NSX: 90-05, 16-22; Freed: 08-16, 16-24, 24-; N-Box: 11-17, 17-23, 23-; Vezel: 13-21, 21-;
 Legend: 4@04-12, 14-21; Crosstour: 10-15; Element: 03-11; Stream: 00-06, 06-14; CR-Z: 10-16; Fit: 01-08, 08-14,
 14-20, 20-; Integra: 3@93-01; e:Ny1: 23-""",
"Acura": """Integra: 22-; TLX: 14-20, 20-; ILX: 12-22; RLX: 13-20; RDX: 06-12, 12-18, 18-; MDX: 00-06, 06-13, 13-20, 21-;
 NSX: 2@16-22; ZDX: 09-13, 24-; TSX: 03-08, 08-14; TL: 3@03-08, 08-14; RSX: 01-06; ADX: 25-""",
"Nissan": """Micra: 2@92-02, 02-10, 10-17, 17-23, 25-; Sunny: 7@11-20; Terrano: 13-20; Kicks: 16-24, 24-; Magnite: 20-;
 X-Trail: 00-07, 07-13, 13-22, 22-; Qashqai: 06-13, 13-21, 21-; Juke: 10-19, 19-; Leaf: 10-17, 17-25, 25-; Ariya: 22-;
 Navara: 3@97-05, 05-14, 14-; Patrol: 5@97-10, 10-24, 24-; Altima: 3@01-06, 06-12, 12-18, 18-; Sentra: 5@00-06, 06-12,
 12-19, 19-25, 25-; Rogue: 07-13, 13-20, 20-; Murano: 02-07, 07-14, 14-24, 24-; Pathfinder: 3@04-12, 12-20, 21-;
 Frontier: 2@04-21, 21-; Titan: 03-15, 16-24; Maxima: 6@03-08, 08-15, 15-23; GT-R: 07-25; 370Z: 08-20; Z: 22-;
 350Z: 02-09; Note: 04-12, 12-20, 20-; Almera: 00-06, 12-19; Versa: 06-11, 11-19, 19-; Serena: 3@05-10, 10-16, 16-22,
 22-; Teana: 03-08, 08-14, 14-20; Armada: 03-16, 17-24, 25-; Cube: 3@08-19; Skyline: 10@98-02, 01-06, 06-14, 14-;
 Silvia: 6@93-98, 99-02; Tiida: 04-12, 12-; Sylphy: 12-19, 19-; Dayz: 13-19, 19-; Sakura: 22-; Gravite: 26-;
 Tekton: 26-; X-Trail e-Power: 22-; Evalia: 12-16""",
"Datsun": "redi-GO: 16-22; GO: 14-22; GO+: 15-22; on-DO: 14-20; mi-DO: 15-20",
"Infiniti": """Q50: 13-; Q60: 16-22; Q70: 13-19; QX50: 13-17, 18-; QX55: 21-; QX60: 12-20, 21-; QX70: 13-17; QX80: 10-24, 24-;
 G35: 02-07; G37: 08-13; FX: 03-08, 08-13; M: 05-10, 10-13; QX30: 16-19; Q30: 15-19; EX: 07-13; JX: 12-13""",
"Mazda": """Mazda2: 02-07, 07-14, 14-; Mazda3: 03-09, 09-13, 13-18, 19-; Mazda6: 02-07, 07-12, 12-24; CX-3: 15-;
 CX-30: 19-; CX-5: 12-17, 17-25, 25-; CX-50: 22-; CX-60: 22-; CX-70: 24-; CX-80: 24-; CX-90: 23-; CX-9: 06-15, 16-23;
 CX-7: 06-12; MX-5: 89-97, 98-05, 05-15, 15-; RX-8: 03-12; RX-7: 3@92-02; MX-30: 20-; Mazda5: 05-10, 10-18;
 BT-50: 06-11, 11-20, 20-; Tribute: 00-11; Mazda6 e: 25-; EZ-6: 24-; Premacy: 99-05; Demio: 96-02; MPV: 99-06, 06-16""",
"Mitsubishi": """Outlander: 01-06, 06-12, 12-21, 21-; Eclipse Cross: 17-; Pajero: 3@99-06, 06-21; Pajero Sport: 96-08, 08-15,
 15-; Lancer: 7@00-07, 07-17; Lancer Evolution: 7@01-03, 03-07, 07-16; Mirage: 6@12-; ASX: 10-22, 23-; Xpander: 17-;
 Triton: 4@05-15, 15-24, 24-; L200: 4@05-15, 15-24, 24-; Eclipse: 3@99-05, 05-12; Galant: 8@96-03, 03-12;
 Montero: 3@99-06, 06-21; Delica: 5@07-; i-MiEV: 09-21; eK: 01-06, 06-13, 13-19, 19-; Cedia: 00-07; Endeavor: 03-11;
 Outlander PHEV: 13-21, 21-; Destinator: 25-; Xforce: 23-""",
"Subaru": """Impreza: 92-00, 00-07, 07-11, 11-16, 16-23, 23-; WRX: 14-21, 21-; Forester: 97-02, 02-08, 08-13, 13-18, 18-25, 25-;
 Outback: 2@99-04, 04-09, 09-14, 14-19, 19-25, 25-; Legacy: 3@98-03, 03-09, 09-14, 14-19, 19-25; Crosstrek: 12-17,
 17-23, 23-; XV: 12-17, 17-23; Ascent: 18-; BRZ: 12-21, 21-; Solterra: 22-; Tribeca: 05-14; Levorg: 14-20, 20-;
 Baja: 02-06; SVX: 91-97; Justy: 84-94; Uncharted: 25-; Trailseeker: 25-""",
"Suzuki": """Swift: 1@83-88, 88-95, 95-00, 00-04, 04-10, 10-17, 17-24, 24-; Vitara: 88-98, 15-; Grand Vitara: 98-05, 05-15;
 Jimny: 2@81-98, 98-18, 18-; SX4: 06-14; S-Cross: 13-21, 21-; Baleno: 95-02, 15-22, 22-; Celerio: 14-21, 21-;
 Ignis: 00-08, 16-; Alto: 4@94-02, 98-14, 14-21, 21-; Ertiga: 12-18, 18-; Kizashi: 09-16; Splash: 08-15; Liana: 01-07;
 Wagon R: 93-98, 98-03, 03-08, 08-12, 12-17, 17-; Hustler: 14-19, 19-; Spacia: 13-17, 17-23, 23-; XL7: 20-;
 Across: 20-; Swace: 20-; Samurai: 85-95; Cappuccino: 91-98; Fronx: 23-; e Vitara: 25-; Equator: 09-12;
 Forenza: 04-08; Reno: 05-08; Aerio: 01-07; Carry: 99-13, 13-""",
"Daihatsu": """Terios: 97-06, 06-17, 17-; Sirion: 98-04, 04-10, 11-; Copen: 02-12, 14-; Mira: 7@06-18; Tanto: 03-07, 07-13,
 13-19, 19-; Move: 95-98, 98-02, 02-06, 06-10, 10-14, 14-23, 25-; Rocky: 19-; Xenia: 03-11, 11-21, 21-; Ayla: 13-23, 23-;
 Sigra: 16-; Gran Max: 07-; Charade: 4@93-00; Cuore: 7@03-07; Hijet: 10@04-21, 21-; Thor: 16-; Taft: 20-""",
"Isuzu": "D-Max: 02-12, 12-20, 20-; MU-X: 13-20, 20-; MU-7: 04-13; Trooper: 2@91-02; Rodeo: 2@98-04; Ascender: 02-08; VehiCROSS: 97-01; Panther: 91-",
# ----------------------------------------------------------------------------------------------- Germany
"Volkswagen": """Golf: 4@97-03, 03-08, 08-12, 12-19, 19-; Polo: 3@94-01, 01-09, 09-17, 17-; Vento: 10-22; Virtus: 22-;
 Taigun: 21-; Tiguan: 07-16, 16-24, 24-; Tiguan Allspace: 17-24; Passat: 5@96-05, 05-10, 10-14, 14-23, 23-;
 Jetta: 4@98-05, 05-10, 10-18, 18-; Ameo: 16-20; T-Roc: 17-25, 25-; T-Cross: 18-; Taigo: 21-; Nivus: 20-; Tera: 25-;
 Touareg: 02-10, 10-18, 18-; Atlas: 17-23, 24-; Atlas Cross Sport: 19-; ID.3: 19-; ID.4: 20-; ID.5: 21-; ID.7: 23-;
 ID. Buzz: 22-; ID.2all: 26-; Arteon: 17-24; Beetle: 2@97-11, 11-19; Scirocco: 3@08-17; Up: 11-23; Sharan: 95-10,
 10-22; Touran: 03-15, 15-; Caddy: 3@04-20, 20-; Amarok: 10-22, 22-; Multivan: 7@21-; Transporter: 5@03-15, 15-24, 24-;
 Phaeton: 02-16; CC: 08-17; Eos: 06-15; Lupo: 98-05; Fox: 03-21; Gol: 94-23; Corrado: 88-95; Routan: 08-14;
 Golf Plus: 05-14; Golf Sportsvan: 14-20; Teramont: 17-23, 24-; Lavida: 08-12, 12-18, 18-; Bora: 98-05; Santana: 12-22""",
"Audi": """A1: 10-18, 18-24; A3: 96-03, 03-12, 12-20, 20-; A4: 94-01, 01-04, 04-08, 08-15, 15-24; A5: 07-16, 16-24, 24-;
 A6: 2@97-04, 04-11, 11-18, 18-25, 25-; A7: 10-18, 18-; A8: 94-02, 02-10, 10-17, 17-; Q2: 16-; Q3: 11-18, 18-25, 25-;
 Q4 e-tron: 21-; Q5: 08-17, 17-24, 24-; Q6 e-tron: 24-; Q7: 05-15, 15-; Q8: 18-; Q8 e-tron: 22-; e-tron: 18-22;
 e-tron GT: 21-; TT: 98-06, 06-14, 14-23; R8: 06-15, 15-24; RS3: 11-12, 15-20, 21-; RS4: 2@05-08, 12-15, 18-;
 RS5: 10-16, 17-24; RS6: 2@08-10, 13-18, 19-; RS7: 13-18, 19-; S4: 4@03-08, 09-16, 16-; S5: 07-16, 17-; Q3 Sportback: 19-;
 Q5 Sportback: 20-; A6 e-tron: 24-; A3 Sportback e-tron: 14-20; 100: 3@82-94; 80: 4@86-96; allroad: 99-05, 06-11""",
"BMW": """1 Series: 04-11, 11-19, 19-24, 24-; 2 Series: 14-21, 21-; 2 Series Gran Coupe: 19-; 2 Series Active Tourer: 14-21,
 21-; 3 Series: 3@90-00, 98-06, 05-13, 12-19, 19-; 4 Series: 13-20, 20-; 5 Series: 3@88-96, 95-03, 03-10, 10-17, 17-23,
 23-; 6 Series: 2@03-10, 11-18; 6 Series GT: 17-23; 7 Series: 3@94-01, 01-08, 08-15, 15-22, 22-; 8 Series: 89-99, 18-;
 X1: 09-15, 15-22, 22-; X2: 17-23, 23-; X3: 03-10, 10-17, 17-24, 24-; X4: 14-18, 18-; X5: 99-06, 06-13, 13-18, 18-;
 X6: 08-14, 14-19, 19-; X7: 18-; XM: 22-; Z3: 95-02; Z4: 02-08, 09-16, 18-; Z8: 99-03; i3: 13-22; i4: 21-; i5: 23-;
 i7: 22-; i8: 14-20; iX: 21-; iX1: 22-; iX2: 23-; iX3: 20-25, 26-; M2: 15-21, 22-; M3: 3@00-06, 07-13, 14-20, 20-;
 M4: 14-20, 20-; M5: 3@98-03, 05-10, 11-16, 17-23, 24-; M8: 19-; 3 Series GT: 13-20; 1 Series M: 11-12;
 X3 M: 19-; X5 M: 09-13, 15-19, 20-; X6 M: 09-14, 15-19, 20-; 1600: 66-75; 2002: 68-76; 3 Series (E30): 82-94""",
"Mercedes-Benz": """A-Class: 97-04, 04-12, 12-18, 18-; B-Class: 05-11, 11-18, 18-; C-Class: 93-00, 00-07, 07-14, 14-21, 21-;
 CLA: 13-19, 19-25, 25-; CLS: 04-10, 11-17, 18-23; E-Class: 2@95-02, 02-09, 09-16, 16-23, 23-; S-Class: 4@91-98, 98-05,
 05-13, 13-20, 20-; GLA: 13-20, 20-; GLB: 19-; GLC: 15-22, 22-; GLC Coupe: 16-23, 23-; GLE: 15-19, 19-; GLE Coupe: 15-19,
 19-; GLS: 15-19, 19-; G-Class: 79-18, 18-; GLK: 08-15; ML: 97-05, 05-11, 11-15; GL: 06-12, 12-16; SL: 4@89-01, 01-11,
 12-20, 21-; SLK: 96-04, 04-11, 11-16; SLC: 16-20; SLS AMG: 10-14; AMG GT: 14-21, 23-; AMG GT 4-Door: 18-;
 EQA: 21-; EQB: 21-; EQC: 19-24; EQE: 22-; EQE SUV: 22-; EQS: 21-; EQS SUV: 22-; V-Class: 2@96-03, 14-24, 24-;
 Vito: 96-03, 03-14, 14-; Sprinter: 95-06, 06-18, 18-; CLK: 97-02, 02-10; R-Class: 05-13; Maybach S-Class: 15-20, 21-;
 Maybach GLS: 20-; X-Class: 17-20; CL-Class: 3@99-06, 06-14; E-Class Coupe: 09-16, 17-23; CLE: 23-;
 C-Class Coupe: 11-15, 15-23; 190: 82-93; W123: 76-86; GLC EV: 25-""",
"Porsche": """911: 5@97-05, 04-12, 11-19, 19-; Cayenne: 02-10, 10-17, 17-; Macan: 14-24; Macan Electric: 24-;
 Panamera: 09-16, 16-23, 23-; Taycan: 19-; Boxster: 96-04, 04-12, 12-16, 16-; Cayman: 05-12, 12-16, 16-;
 718 Boxster: 16-; 718 Cayman: 16-; Carrera GT: 03-06; 918 Spyder: 13-15; 924: 76-88; 928: 77-95; 944: 82-91;
 968: 92-95; 356: 48-65; 959: 86-93""",
"Opel": """Astra: 5@91-98, 98-09, 04-14, 09-15, 15-21, 21-; Corsa: 2@93-00, 00-06, 06-14, 14-19, 19-; Insignia: 08-17,
 17-22; Mokka: 12-19, 20-; Grandland: 17-24, 24-; Crossland: 17-24; Frontera: 91-98, 98-04, 24-; Zafira: 99-05, 05-14,
 11-19; Meriva: 03-10, 10-17; Adam: 13-19; Vectra: 3@88-95, 95-02, 02-08; Omega: 2@86-93, 94-03; Agila: 00-07, 08-14;
 Antara: 06-15; Combo: 18-; Vivaro: 01-14, 14-19, 19-; Ampera: 11-15; Ampera-e: 17-19; Calibra: 89-97; Tigra: 94-00,
 04-09; Manta: 70-88; Kadett: 79-91; Cascada: 13-19; Karl: 15-19; GT: 07-09; Signum: 03-08""",
"Vauxhall": """Astra: 5@91-98, 98-09, 04-14, 09-15, 15-21, 21-; Corsa: 2@93-00, 00-06, 06-14, 14-19, 19-; Insignia: 08-17,
 17-22; Mokka: 12-19, 20-; Grandland: 17-24, 24-; Crossland: 17-24; Frontera: 24-; Zafira: 99-05, 05-14, 11-19;
 Meriva: 03-10, 10-17; Adam: 13-19; Viva: 15-19; Vectra: 95-02, 02-08; Monaro: 04-06; VXR8: 07-17; Cavalier: 3@88-95""",
"MG": """Hector: 19-; Hector Plus: 20-; Astor: 21-; Gloster: 20-; ZS EV: 19-; ZS: 17-24, 24-; Comet EV: 23-;
 Windsor EV: 24-; MG4: 22-; MG3: 11-24, 24-; MG5: 20-; MG6: 10-16, 17-; HS: 18-24, 24-; Cyberster: 24-; M9: 25-;
 Marvel R: 21-; MG7: 23-; S5 EV: 25-; RX5: 16-; One: 22-; Majestor: 25-; TF: 02-11; MGF: 95-02; MGB: 62-80;
 Midget: 61-79; ZT: 01-05; ZR: 01-05""",
"Saab": "9-3: 98-02, 02-14; 9-5: 97-10, 10-12; 900: 78-93, 94-98; 9000: 84-98; 9-7X: 05-09; 99: 68-84; 96: 60-80",
"Daewoo": "Matiz: 98-05, 05-15; Cielo: 94-03; Nexia: 94-16; Lanos: 97-02; Nubira: 97-02; Leganza: 97-02; Tico: 91-01; Espero: 90-99; Tacuma: 00-08; Lacetti: 02-09",
"Smart": "Fortwo: 98-07, 07-14, 14-24; Forfour: 04-06, 14-21; #1: 22-; #3: 23-; #5: 25-; Roadster: 03-06",
# ----------------------------------------------------------------------------------------------- France / Italy / Spain
"Peugeot": """208: 12-19, 19-; 2008: 13-19, 19-; 308: 07-13, 13-21, 21-; 3008: 09-16, 16-23, 23-; 408: 10-, 22-;
 508: 10-18, 18-; 5008: 09-16, 17-23, 24-; 108: 14-21; 107: 05-14; 206: 98-12; 207: 06-14; 307: 01-08; 407: 04-11;
 607: 99-10; 806: 94-02; 807: 02-14; 106: 91-03; 205: 83-98; 306: 93-02; 405: 87-97; 406: 95-04; 605: 89-99;
 RCZ: 10-15; Partner: 96-08, 08-18, 18-; Rifter: 18-; Expert: 95-07, 07-16, 16-; Traveller: 16-; Landtrek: 20-;
 iOn: 10-20; 4007: 07-12; 4008: 12-17; 504: 68-05; 404: 60-75; 301: 12-""",
"DS": "DS 3: 09-16, 16-19; DS 3 Crossback: 19-22; DS 3 (crossover): 22-; DS 4: 11-18, 21-; DS 5: 11-18; DS 7: 17-; DS 9: 21-; N°4: 25-; N°8: 25-",
"Citroen": """C3: 02-09, 09-16, 16-24, 24-; C3 (India): 22-; eC3: 23-; C3 Aircross: 17-24, 24-; C3 Aircross (India): 23-;
 Basalt: 24-; Aircross: 25-; C5 Aircross: 18-25, 25-; C4: 04-10, 10-18, 20-; C4 X: 22-; C4 Picasso: 06-13, 13-22;
 C5: 01-08, 08-17; C5 X: 21-; Berlingo: 96-08, 08-18, 18-; C1: 05-14, 14-22; DS3: 09-16; Xsara: 97-06;
 Xsara Picasso: 99-12; Saxo: 96-04; C-Elysee: 12-; Ami: 20-; C2: 03-09; C6: 05-12; C8: 02-14; C-Crosser: 07-12;
 C4 Cactus: 14-20; SpaceTourer: 16-; Jumpy: 95-07, 07-16, 16-; 2CV: 48-90; DS: 55-75; CX: 74-91; XM: 89-00;
 Xantia: 93-01; ZX: 91-98; AX: 86-98; BX: 82-94; Jumper: 94-06, 06-""",
"Renault": """Kwid: 15-; Triber: 19-; Kiger: 21-; Duster: 10-17, 17-23, 24-; Captur: 13-19, 19-; Clio: 2@98-05, 05-12,
 12-19, 19-; Megane: 95-02, 02-08, 08-16, 16-23; Megane E-Tech: 22-; Scenic: 96-03, 03-09, 09-16, 16-22;
 Scenic E-Tech: 24-; Kadjar: 15-22; Koleos: 07-15, 16-23; Grand Koleos: 24-; Arkana: 19-; Austral: 22-;
 Espace: 3@96-02, 02-14, 15-23, 23-; Talisman: 15-22; Laguna: 2@01-07, 07-15; Twingo: 93-07, 07-14, 14-;
 Twingo E-Tech: 26-; Zoe: 12-24; Logan: 04-12, 12-22; Fluence: 09-16; Pulse: 12-17; Scala: 12-17; Lodgy: 12-22;
 Rafale: 24-; Symbioz: 24-; 5 E-Tech: 24-; 4 E-Tech: 25-; Kangoo: 97-08, 08-21, 21-; Trafic: 2@01-14, 14-;
 Master: 2@97-10, 10-; Sandero: 07-12, 12-20, 20-; Alaskan: 17-; Vel Satis: 01-09; Avantime: 01-03; Safrane: 92-00;
 Modus: 04-12; Wind: 10-13; Latitude: 10-15; Twizy: 12-23; 4: 61-94; 5: 72-96; 19: 88-00; 21: 86-95; 25: 84-92;
 Fuego: 80-92; Alpine A110: 61-77; Boreal: 25-; Kardian: 24-""",
"Dacia": "Sandero: 07-12, 12-20, 20-; Logan: 04-12, 12-20, 20-; Duster: 10-17, 17-23, 24-; Jogger: 21-; Spring: 21-; Lodgy: 12-22; Dokker: 12-21; Bigster: 25-",
"Alpine": "A110: 2@17-; A290: 24-; A390: 25-; A610: 91-95",
"Fiat": """500: 07-24; 500e: 20-; 500X: 14-24; 500L: 12-22; Panda: 2@03-12, 11-, 24-; Punto: 3@93-99, 99-10, 05-18;
 Grande Punto: 05-12; Punto Evo: 09-12; Avventura: 14-20; Linea: 07-18; Palio: 96-17; Siena: 96-16; Uno: 83-13, 10-21;
 Tipo: 88-95, 15-; Bravo: 95-01, 07-14; Doblo: 00-10, 10-22, 22-; Fiorino: 07-; Ducato: 2@94-06, 06-; Qubo: 08-19;
 Freemont: 11-16; Fullback: 16-19; Strada: 96-20, 20-; Toro: 16-; Pulse: 21-; Fastback: 22-; Argo: 17-; Mobi: 16-;
 Cronos: 18-; Multipla: 98-10; Stilo: 01-07; Croma: 05-11; Seicento: 98-10; Cinquecento: 91-98; Barchetta: 95-05;
 Coupe: 93-00; 124 Spider: 66-85, 16-20; 600: 55-69, 25-; Grande Panda: 24-; Topolino: 23-; Marea: 96-07;
 Sedici: 06-14; Idea: 03-12; Petra: 04-07; Palio Adventure: 99-17""",
"Abarth": "500: 08-24; 595: 16-24; 124 Spider: 16-19; 500e: 23-; 600e: 24-; Punto: 07-15",
"Alfa Romeo": """Giulia: 15-; Stelvio: 16-; Tonale: 22-; Junior: 24-; Giulietta: 10-20; MiTo: 08-18; 4C: 13-20;
 8C Competizione: 07-10; 159: 05-11; 156: 97-07; 147: 00-10; 166: 98-07; 164: 87-98; 155: 92-98; 145: 94-01; 146: 94-01;
 GT: 03-10; GTV: 95-06; Spider: 95-06, 66-93; Brera: 05-10; 33 Stradale: 67-69, 24-; Giulia (1962): 62-78;
 Alfasud: 71-89; 75: 85-92; 33: 83-95""",
"Lancia": "Ypsilon: 96-03, 03-11, 11-24, 24-; Delta: 79-94, 93-99, 08-14; Thema: 84-94, 11-14; Musa: 04-12; Phedra: 02-10; Thesis: 02-09; Kappa: 94-01; Lybra: 99-05; Stratos: 73-78; Fulvia: 63-76; Voyager: 11-15; Gamma: 26-",
"Maserati": """Ghibli: 13-24; Quattroporte: 5@04-12, 13-24; Levante: 16-24; Grecale: 22-; MC20: 20-; GranTurismo: 07-19, 23-;
 GranCabrio: 09-19, 24-; Coupe: 01-07; Spyder: 01-07; MC12: 04-05; Bora: 71-78; Merak: 72-83; 3200 GT: 98-02""",
"Ferrari": """Roma: 20-; Roma Spider: 23-; Amalfi: 25-; Portofino: 17-23; 296 GTB: 21-; 296 GTS: 22-; SF90 Stradale: 19-;
 F8 Tributo: 19-23; 488 GTB: 15-19; 458 Italia: 09-15; F430: 04-09; 360: 99-04; 355: 94-99; 812 Superfast: 17-24;
 12Cilindri: 24-; Purosangue: 22-; LaFerrari: 13-16; Enzo: 02-04; F12berlinetta: 12-17; FF: 11-16; GTC4Lusso: 16-20;
 California: 08-17; 599 GTB Fiorano: 06-12; 612 Scaglietti: 04-11; 575M Maranello: 02-06; 550 Maranello: 96-02;
 Testarossa: 84-96; F40: 87-92; F50: 95-97; 308: 75-85; 328: 85-89; 348: 89-95; Dino 246: 69-74; 250 GTO: 62-64;
 Daytona SP3: 22-25; F80: 25-; 849 Testarossa: 25-; Elettrica: 26-""",
"Lamborghini": """Huracan: 14-24; Temerario: 24-; Aventador: 11-22; Revuelto: 23-; Urus: 18-; Gallardo: 03-13; Murcielago: 01-10;
 Diablo: 90-01; Countach: 74-90, 22-; Miura: 66-73; Espada: 68-78; Reventon: 08-09; Sian: 20-22; Veneno: 13-14;
 Jalpa: 81-88; LM002: 86-93; Fenomeno: 25-""",
"Pagani": "Zonda: 99-19; Huayra: 11-; Utopia: 22-",
"SEAT": "Ibiza: 2@93-02, 02-08, 08-17, 17-; Leon: 99-05, 05-12, 12-20, 20-; Arona: 17-; Ateca: 16-; Tarraco: 18-24; Alhambra: 96-10, 10-20; Altea: 04-15; Toledo: 2@98-04, 04-09, 12-19; Mii: 11-21; Cordoba: 93-09; Exeo: 08-13; Arosa: 97-04",
"Cupra": "Formentor: 20-; Born: 21-; Leon: 20-; Ateca: 18-; Tavascan: 24-; Terramar: 24-; Raval: 26-",
# ----------------------------------------------------------------------------------------------- UK / Sweden / others in Europe
"Volvo": """XC40: 17-; EX30: 23-; EX40: 24-; EC40: 24-; XC60: 08-17, 17-; XC90: 02-14, 14-; EX90: 24-; ES90: 25-;
 S60: 00-09, 10-18, 18-; S90: 2@16-; V40: 2@12-19; V60: 10-18, 18-; V90: 2@16-; S40: 95-04, 04-12; V50: 04-12;
 V70: 2@00-07, 07-16; S80: 98-06, 06-16; C30: 06-13; C40: 21-23; C70: 96-05, 06-13; XC70: 2@00-07, 07-16, 25-;
 240: 74-93; 740: 84-92; 850: 91-96; 940: 90-98; P1800: 61-73; Amazon: 56-70""",
"Polestar": "1: 19-21; 2: 19-; 3: 24-; 4: 24-; 5: 25-",
"Mini": "Hatch: 01-06, 06-13, 13-24, 24-; Cooper: 01-06, 06-13, 13-24, 24-; Countryman: 10-16, 17-24, 24-; Clubman: 07-14, 15-24; Paceman: 13-16; Coupe: 11-15; Roadster: 12-15; Convertible: 04-08, 09-15, 16-24, 25-; Aceman: 24-; Cooper SE: 20-24; Mini (classic): 59-00",
"Land Rover": """Defender: 83-16, 20-; Discovery: 89-98, 98-04, 04-09, 09-16, 17-; Discovery Sport: 14-; Freelander: 97-06, 06-15;
 Range Rover: 2@94-02, 02-12, 12-21, 22-; Range Rover Sport: 05-13, 13-22, 22-; Range Rover Evoque: 11-18, 19-;
 Range Rover Velar: 17-; Series: 48-85; LR2: 06-15; LR3: 04-09; LR4: 09-16""",
"Jaguar": """F-Pace: 16-; E-Pace: 17-; I-Pace: 18-25; XE: 15-24; XF: 07-15, 15-24; XJ: 6@03-09, 09-19; F-Type: 13-24;
 XK: 96-06, 06-15; S-Type: 99-08; X-Type: 01-09; E-Type: 61-75; XJS: 75-96; XJ220: 92-94; Mark 2: 59-67;
 Type 00: 26-; XJ6 (Series): 68-92""",
"Aston Martin": """DB11: 16-23; DB12: 23-; DBS Superleggera: 18-23; DBS: 07-12; DBX: 20-; Vantage: 05-17, 18-;
 Vanquish: 01-07, 12-18, 24-; Valkyrie: 21-; Rapide: 10-20; DB9: 04-16; DB7: 94-04; Virage: 11-12, 89-00; Valhalla: 25-;
 One-77: 09-12; Cygnet: 11-13; DB5: 63-65; DB4: 58-63; Lagonda: 76-90""",
"Bentley": "Continental GT: 03-11, 11-18, 18-24, 24-; Continental Flying Spur: 05-13; Flying Spur: 13-19, 19-; Bentayga: 15-; Mulsanne: 80-92, 10-20; Arnage: 98-09; Azure: 95-03, 06-10; Brooklands: 92-98, 08-11; Continental R: 91-03; Turbo R: 85-97",
"Rolls-Royce": "Phantom: 7@03-17, 17-; Ghost: 09-20, 20-; Wraith: 13-23; Dawn: 15-23; Cullinan: 18-; Spectre: 23-; Silver Shadow: 65-80; Silver Spirit: 80-98; Silver Seraph: 98-02; Corniche: 71-96; Silver Cloud: 55-66",
"McLaren": "720S: 17-23; 750S: 23-; 570S: 15-21; 540C: 15-21; 600LT: 18-21; 650S: 14-17; 12C: 11-14; P1: 13-15; Senna: 18-19; Artura: 21-; GT: 19-24; GTS: 24-; Elva: 20-22; Speedtail: 20-21; 765LT: 20-22; F1: 92-98; W1: 25-; MP4-12C: 11-14",
"Lotus": "Elise: 96-01, 01-11, 11-21; Exige: 00-02, 04-11, 12-21; Evora: 09-21; Emira: 22-; Eletre: 23-; Emeya: 24-; Esprit: 76-04; Evija: 21-; Elan: 62-75, 89-95; Europa: 66-75, 06-10; Seven: 57-72",
"Bugatti": "Veyron 16.4: 05-15; Chiron: 16-24; Divo: 19-21; Tourbillon: 26-; Bolide: 24-; Centodieci: 21-22; EB110: 91-95",
"Koenigsegg": "Agera: 10-18; Regera: 16-22; Jesko: 21-; Gemera: 26-; CCX: 06-10; One:1: 14-15",
"Morgan": "Plus Four: 50-69, 20-; Plus Six: 19-23; Super 3: 22-; 3 Wheeler: 11-21; Aero 8: 01-18; Plus 8: 68-04, 12-18; 4/4: 36-19",
"Lada": "Niva: 77-; Niva Travel: 20-; Granta: 11-; Vesta: 15-; XRAY: 15-22; Largus: 12-; Samara: 84-13; Riva: 80-12; Priora: 07-18; Kalina: 04-18; Iskra: 25-",
"Skoda": """Octavia: 96-04, 04-13, 13-20, 20-; Superb: 2@01-08, 08-15, 15-23, 23-; Fabia: 99-07, 07-14, 14-21, 21-;
 Rapid: 11-21; Slavia: 22-; Kushaq: 21-; Kylaq: 25-; Kodiaq: 16-23, 23-; Karoq: 17-; Kamiq: 19-; Yeti: 09-17;
 Laura: 04-13; Scala: 19-; Enyaq: 20-; Elroq: 24-; Citigo: 11-20; Roomster: 06-15; Felicia: 94-01; Favorit: 87-94;
 Epiq: 26-""",
# ----------------------------------------------------------------------------------------------- USA
"Ford": """EcoSport: 03-12, 12-22; Figo: 10-15, 15-19; Figo Aspire: 15-21; Freestyle: 18-21; Endeavour: 03-15, 15-22;
 Everest: 3@22-; Ikon: 99-11; Fiesta: 5@02-08, 08-17, 17-23; Focus: 98-04, 04-11, 11-18, 18-25; Mondeo: 2@96-00,
 00-07, 07-14, 14-22; Fusion: 05-12, 12-20; Mustang: 4@94-04, 05-14, 15-23, 23-; Mustang Mach-E: 20-; F-150: 10@97-03,
 04-08, 09-14, 15-20, 21-; F-150 Lightning: 22-; Super Duty: 99-07, 08-16, 17-22, 23-; Explorer: 3@02-05, 06-10,
 11-19, 20-; Escape: 01-07, 08-12, 13-19, 20-; Kuga: 08-12, 12-19, 19-; Edge: 06-14, 15-23; Expedition: 97-02, 03-17,
 18-; Ranger: 98-11, 11-22, 22-; Bronco: 6@21-; Bronco Sport: 20-; Maverick: 21-; Puma: 19-; Puma Gen-E: 25-;
 Ka: 96-08, 08-16, 16-21; Galaxy: 95-06, 06-15, 15-23; S-Max: 06-15, 15-23; Transit: 6@00-14, 13-; Transit Custom: 12-23,
 23-; Tourneo Connect: 02-13, 13-22, 22-; Taurus: 6@10-19; Flex: 09-19; GT: 04-06, 16-22; Territory: 04-16, 23-;
 Crown Victoria: 92-12; Thunderbird: 02-05; Five Hundred: 05-07; Freestar: 04-07; Excursion: 99-05; Ka+: 16-21;
 Explorer EV: 24-; Capri: 69-86, 24-; Model T: 08-27; Model A: 27-32; Escort: 68-04; Sierra: 82-93; Cortina: 62-82;
 Granada: 72-94; Probe: 89-97; Courier: 98-02""",
"Chevrolet": """Beat: 10-18; Spark: 98-05, 05-10, 10-15, 15-22; Cruze: 08-16, 16-19; Sail: 12-17; Enjoy: 13-17; Tavera: 04-17;
 Aveo: 06-11, 11-20, 23-; Optra: 03-13; Captiva: 06-18, 19-; Trailblazer: 02-09, 12-20, 20-; Silverado: 99-06, 07-13,
 14-18, 19-; Malibu: 6@04-07, 08-12, 13-15, 16-25; Camaro: 4@93-02, 10-15, 16-24; Corvette: 5@97-04, 05-13, 14-19, 20-;
 Tahoe: 2@00-06, 07-14, 15-20, 21-; Suburban: 9@00-06, 07-14, 15-20, 21-; Equinox: 05-09, 10-17, 18-24, 25-;
 Equinox EV: 24-; Traverse: 09-17, 18-23, 24-; Colorado: 04-12, 12-22, 23-; Blazer: 19-; Blazer EV: 24-; Bolt EV: 17-23,
 26-; Bolt EUV: 21-23; Trax: 13-22, 23-; Impala: 8@00-05, 06-13, 14-20; Volt: 11-15, 16-19; Onix: 12-19, 19-;
 Tracker: 19-; Spin: 12-; Express: 96-; Avalanche: 01-06, 07-13; HHR: 06-11; Cobalt: 05-10; Cavalier: 3@95-05;
 Monte Carlo: 6@00-07; SSR: 03-06; Uplander: 05-09; Sonic: 11-20; Orlando: 11-18; Silverado EV: 24-; Montana: 23-;
 Bel Air: 50-81; Chevelle: 64-77; Nova: 62-88; Corvair: 60-69""",
"GMC": "Sierra: 99-06, 07-13, 14-18, 19-; Sierra EV: 24-; Yukon: 2@00-06, 07-14, 15-20, 21-; Acadia: 07-16, 17-23, 24-; Terrain: 10-17, 18-24, 25-; Canyon: 04-12, 15-22, 23-; Savana: 96-; Hummer EV: 21-; Envoy: 98-09; Jimmy: 70-05",
"Cadillac": """Escalade: 99-00, 02-06, 07-14, 15-20, 21-; Escalade IQ: 24-; CT4: 19-; CT5: 19-; CT6: 16-20; XT4: 18-25;
 XT5: 16-25; XT6: 19-25; Lyriq: 22-; Optiq: 24-; Vistiq: 25-; Celestiq: 24-; ATS: 12-19; CTS: 02-07, 08-13, 14-19;
 XTS: 12-19; SRX: 04-09, 10-16; DTS: 06-11; STS: 04-11; ELR: 14-16; XLR: 03-09; DeVille: 59-05; Eldorado: 53-02;
 Seville: 75-04; Fleetwood: 23-96""",
"Buick": "Enclave: 07-17, 18-24, 25-; Encore: 12-22; Encore GX: 19-; Envision: 14-20, 21-; Envista: 23-; LaCrosse: 04-09, 09-16, 16-19; Regal: 5@08-17, 17-20; Verano: 11-17; Lucerne: 05-11; Rendezvous: 01-07; Riviera: 63-99; Park Avenue: 91-05; Electra E5: 24-; GL8: 99-",
"Lincoln": "Navigator: 97-02, 03-06, 07-17, 18-24, 25-; Aviator: 02-05, 19-; Nautilus: 18-23, 24-; Corsair: 19-; MKZ: 06-12, 13-20; MKX: 06-15, 16-18; MKC: 14-19; MKS: 08-16; MKT: 09-19; Continental: 9@95-02, 16-20; Town Car: 81-11; LS: 99-06; Mark LT: 05-08",
"Chrysler": "300: 04-10, 11-23; Pacifica: 04-08, 16-; Voyager: 19-25, 25-; Town & Country: 4@01-07, 08-16; Sebring: 95-10; 200: 10-14, 14-17; PT Cruiser: 00-10; Crossfire: 03-08; Aspen: 06-09; Concorde: 93-04; Grand Caravan: 21-",
"Dodge": """Charger: 6@05-10, 11-23, 24-; Challenger: 3@08-23; Durango: 98-03, 04-09, 11-; Journey: 08-20; Grand Caravan: 5@08-20;
 Viper: 92-95, 96-02, 03-10, 13-17; Dart: 13-16; Avenger: 08-14; Caliber: 06-12; Nitro: 07-11; Magnum: 05-08;
 Neon: 95-99, 00-05, 16-20; Hornet: 22-; Stratus: 95-06; Intrepid: 93-04; Ram: 81-10; Dakota: 87-11; Coronet: 65-76;
 Charger Daytona: 24-""",
"Ram": "1500: 4@09-18, 19-; 2500: 4@10-18, 19-; 1500 Classic: 19-24; ProMaster: 14-; ProMaster City: 15-22; 700: 14-; Rampage: 23-; Dakota: 26-",
"Jeep": """Compass: 06-16, 16-25, 25-; Meridian: 22-; Wrangler: 2@96-06, 06-18, 18-; Grand Cherokee: 2@98-04, 04-10, 11-21,
 21-; Grand Cherokee L: 21-; Cherokee: 84-01, 13-23, 26-; Liberty: 01-07, 07-12; Renegade: 14-; Gladiator: 19-;
 Commander: 05-10, 21-; Avenger: 23-; Wagoneer: 21-; Grand Wagoneer: 63-91, 21-; Wagoneer S: 24-; Patriot: 07-17;
 Recon: 26-; CJ: 44-86; Willys MB: 41-45""",
"Tesla": "Model S: 12-; Model 3: 17-23, 23-; Model X: 15-; Model Y: 20-25, 25-; Cybertruck: 23-; Roadster: 08-12; Semi: 22-; Cybercab: 26-",
"Rivian": "R1T: 21-; R1S: 22-; R2: 26-; R3: 27-; EDV: 22-",
"Lucid": "Air: 21-; Gravity: 24-",
"Fisker": "Karma: 11-12; Ocean: 23-24",
"Hummer": "H1: 92-06; H2: 02-09; H3: 05-10",
"Pontiac": "Firebird: 3@82-92, 93-02; GTO: 04-06, 64-74; G6: 04-10; G8: 08-09; Solstice: 06-10; Vibe: 02-10; Aztek: 01-05; Grand Prix: 6@97-03, 04-08; Bonneville: 00-05; Torrent: 06-09; Sunfire: 95-05; Fiero: 84-88",
"Saturn": "Ion: 03-07; Aura: 07-09; Vue: 02-07, 08-10; Outlook: 07-10; Sky: 07-10; Astra: 08-09; Relay: 05-07; S-Series: 91-02",
"Mercury": "Grand Marquis: 92-11; Mountaineer: 97-10; Milan: 06-11; Mariner: 05-11; Sable: 86-09",
"AM General": "Hummer: 92-06",
"Scion": "tC: 04-10, 11-16; xB: 03-07, 08-15; xD: 07-14; FR-S: 12-16; iQ: 11-15; iA: 15-16; iM: 15-16",
"Plymouth": "Neon: 95-01; Prowler: 97-02; Voyager: 74-00; Barracuda: 64-74; Road Runner: 68-80; Fury: 56-78",
"Oldsmobile": "Alero: 99-04; Aurora: 94-03; Intrigue: 98-02; Bravada: 91-04; Cutlass: 61-99; 88: 49-99",
# ----------------------------------------------------------------------------------------------- China
"BYD": """Atto 3: 22-; Seal: 22-; Sealion 7: 24-; Seal U: 23-; Dolphin: 21-; Dolphin Mini: 23-; Seagull: 23-; Han: 20-;
 Tang: 15-18, 18-; Song Plus: 20-; Song Pro: 19-; Qin Plus: 21-; Qin L: 24-; Yuan Plus: 22-; Yuan Up: 24-; e6: 10-;
 eMAX 7: 24-; M6: 24-; Shark: 24-; Seal 06: 24-; Sealion 06: 24-; Atto 2: 25-; Song L: 23-; F3: 05-21; Seal 5: 25-;
 Destroyer 05: 22-; Frigate 07: 22-; Atto 1: 25-; Sealion 05: 25-""",
"Denza": "D9: 22-; N7: 23-; N8: 22-; Z9 GT: 24-; X: 19-",
"Yangwang": "U8: 23-; U9: 24-; U7: 24-",
"Fangchengbao": "Bao 5: 23-; Bao 8: 25-; Tai 3: 25-",
"Geely": "Coolray: 18-; Emgrand: 09-; Azkarra: 19-; Okavango: 20-; Monjaro: 21-; Tugella: 19-; Geometry C: 20-; Galaxy E5: 24-; Galaxy L7: 23-; Preface: 20-; Starray: 23-; EX5: 24-; Boyue: 16-; Xingyue L: 21-; Panda Mini: 23-; GX3 Pro: 21-",
"Chery": "Tiggo 2: 17-; Tiggo 4 Pro: 20-; Tiggo 7: 16-20, 20-; Tiggo 7 Pro: 20-; Tiggo 8: 18-; Tiggo 8 Pro: 20-; Tiggo 9: 23-; Arrizo 5: 16-; Arrizo 8: 22-; QQ: 03-13; QQ Ice Cream: 21-; eQ1: 17-; Omoda 5: 22-",
"Omoda": "5: 22-; C5: 22-; E5: 23-; 7: 25-; C9: 25-",
"Jaecoo": "7: 23-; 8: 24-; 5: 25-; J7: 23-",
"Jetour": "X70: 18-; Dashing: 22-; Traveller: 23-; X90: 19-; T2: 24-",
"Exeed": "TXL: 19-; VX: 21-; RX: 22-; Sterra ES: 24-",
"Great Wall": "Wingle: 06-; Poer: 19-; Cannon: 20-; Hover: 05-11; Voleex: 08-",
"Haval": "H6: 11-17, 17-21, 21-; H9: 14-24, 24-; Jolion: 20-; H2: 14-21; Dargo: 20-; F7: 18-; Big Dog: 20-; Raptor: 23-; H7: 24-",
"Tank": "300: 20-; 500: 21-; 400: 23-; 700: 24-",
"Ora": "Good Cat: 20-; 03: 20-; 07: 23-; Ballet Cat: 22-; Lightning Cat: 22-",
"Wey": "VV5: 17-; VV7: 17-; Coffee 01: 21-; Lanshan: 23-",
"Changan": "CS35 Plus: 18-; CS55 Plus: 19-; CS75 Plus: 19-; CS85: 19-; UNI-T: 20-; UNI-K: 21-; UNI-V: 22-; Eado: 12-; Hunter: 23-; Alsvin: 14-; Lumin: 22-",
"Deepal": "S7: 23-; SL03: 22-; G318: 24-; S05: 24-; L07: 24-",
"Avatr": "11: 22-; 12: 23-; 07: 24-; 06: 25-",
"NIO": "ES8: 18-22, 23-; ES6: 18-; EC6: 20-; ET7: 21-; ET5: 22-; ES7: 22-; EL6: 23-; ET9: 25-; Onvo L60: 24-; Firefly: 25-; EP9: 16-",
"XPeng": "P7: 20-; P7+: 24-; G9: 22-; G6: 23-; P5: 21-; G3: 18-; X9: 23-; Mona M03: 24-; G7: 25-",
"Li Auto": "One: 19-22; L6: 24-; L7: 23-; L8: 22-; L9: 22-; Mega: 24-; i8: 25-; i6: 25-",
"Zeekr": "001: 21-; 007: 23-; 009: 22-; X: 23-; 7X: 24-; Mix: 24-; 9X: 25-",
"Lynk & Co": "01: 17-; 02: 18-; 03: 18-; 05: 19-; 06: 20-; 08: 23-; 09: 21-; Z10: 24-",
"Hongqi": "H5: 17-; H9: 20-; HS5: 19-; HS7: 19-; E-HS9: 20-; L5: 14-; EH7: 24-; HQ9: 24-",
"Wuling": "Hongguang Mini EV: 20-; Air EV: 22-; Bingo: 23-; Almaz: 19-; Cortez: 18-; Confero: 17-; Starlight: 23-; Binguo: 23-",
"Baojun": "530: 18-; 510: 17-; E300: 20-; Yep: 23-; KiWi EV: 21-",
"GAC": "GS8: 16-; GS4: 15-; GS3: 17-; M8: 17-; Emkoo: 22-; Empow: 21-; Emzoom: 23-",
"Aion": "S: 19-; Y: 21-; V: 23-; LX: 19-; Hyper GT: 23-; Hyper SSR: 23-; UT: 25-",
"Xiaomi": "SU7: 24-; YU7: 25-; SU7 Ultra: 25-",
"Leapmotor": "C10: 23-; C11: 21-; C16: 24-; T03: 20-; B10: 25-; C01: 22-",
"Aito": "M5: 22-; M7: 22-; M9: 23-; M8: 25-",
"Neta": "V: 20-; U: 19-; S: 22-; GT: 23-; X: 23-; Aya: 24-",
"Voyah": "Free: 21-; Dreamer: 22-; Passion: 23-; Courage: 24-",
"Dongfeng": "Aeolus Shine: 22-; Fengon 580: 16-; Box: 24-; Rich 7: 24-",
"JAC": "S2: 15-; S3: 14-; T8: 18-; JS4: 20-; iEV7S: 18-",
"Roewe": "RX5: 16-; i5: 18-; i6: 17-; 550: 08-16; D7: 24-",
"BAIC": "BJ40: 14-; BJ60: 22-; X55: 18-; EU5: 18-; Senova D50: 14-",
"Maxus": "T60: 17-; T90: 21-; D90: 17-; G10: 14-; MIFA 9: 21-; Deliver 9: 20-; eDeliver 3: 20-",
"Proton": "Saga: 85-08, 08-16, 16-; Persona: 07-16, 16-; Iriz: 14-; X50: 20-; X70: 18-; X90: 23-; Wira: 93-09; Perdana: 95-10, 13-; Exora: 09-; Preve: 12-; e.MAS 7: 24-",
"Perodua": "Myvi: 05-11, 11-17, 17-; Axia: 14-23, 23-; Bezza: 16-; Alza: 09-18, 22-; Aruz: 19-; Ativa: 21-; Kancil: 94-09; Kelisa: 01-07; Kenari: 00-09; Viva: 07-14",
"VinFast": "VF 3: 24-; VF 5: 23-; VF 6: 24-; VF 7: 24-; VF 8: 22-; VF 9: 23-; Lux A2.0: 19-22; Lux SA2.0: 19-22; Fadil: 19-22; VF e34: 21-23",
"Togg": "T10X: 23-; T10F: 25-",
}

# Extension (v2): more models, older generations and more markets. Merged into T by make; models already
# listed above are not repeated here (duplicate make/model/start entries are removed anyway).
EXTRA = {
"Maruti Suzuki": "Alto 800 (2nd): 19-23; Swift Dzire Tour: 18-; Wagon R Stingray: 13-19; Estilo: 09-14; 1000: 90-00; S-Cross (Nexa): 17-22; Super Carry: 16-; Tour S: 18-",
"Tata": "Sumo Grande: 08-16; Movus: 14-16; Manza: 09-16; Vista: 08-15; Venture: 10-19; Yodha: 17-; Xenon Yodha: 17-; Intra: 19-; Ace: 05-; Nexon iCNG: 24-; Altroz Racer: 24-; Avinya: 26-",
"Mahindra": "Bolero Camper: 03-; Bolero Pik-Up: 08-; Scorpio Getaway: 07-; Imperio: 19-; Jeeto: 15-; e-Verito: 16-21; Reva: 01-12; MM540: 79-10; Commander: 91-01; Voyager: 97-02; BE 6 Formula E: 25-; XUV 3XO EV: 25-; XUV.e8: 26-",
"Hyundai": "Elantra N: 21-; i20 N: 20-24; i30 N: 17-; Kona N: 21-; Ioniq 5 N: 23-; Tiburon: 96-01, 01-08; Coupe: 96-08; Trajet: 99-08; Galloper: 91-03; Pony: 75-90; Stellar: 83-97; Excel: 85-94; Scoupe: 90-95; Lavita: 01-10; Click: 02-11; Sonata Hybrid: 11-; Elantra Touring: 07-12; Entourage: 06-09; Nexo: 18-; Mufasa: 23-; Custo: 23-; Grand Starex: 07-21; Verna Fluidic: 11-17",
"Kia": "Sephia: 92-03; Shuma: 96-04; Magentis: 00-10; Opirus: 03-11; Retona: 98-03; Borrego: 08-; Rio 5: 05-17; Pro Ceed: 08-; Sportage EV: 26-; Sonet X-Line: 22-; Seltos X-Line: 23-; Carens Clavis EV: 25-; K3: 18-24; K4: 24-; Pegas: 17-; Soluto: 19-; Rondo: 13-19",
"Toyota": "Corona: 9@87-01; Carina: 84-01; Cressida: 76-92; Starlet (classic): 73-99; Tercel: 78-99; Paseo: 91-99; Echo: 99-05; Previa: 90-00, 00-19; Land Cruiser 80: 89-97; Land Cruiser 100: 98-07; T100: 92-98; Hilux Surf: 84-09; Mark II: 68-04; Mark X: 04-19; Chaser: 77-01; Cresta: 80-01; Soarer: 81-05; Century: 67-97, 97-17, 18-; Sienta: 03-15, 15-22, 22-; Passo: 04-10, 10-16, 16-; Porte: 04-20; Ractis: 05-16; Wish: 03-17; Isis: 04-17; Esquire: 14-22; Voxy: 01-07, 07-14, 14-22, 22-; Granvia: 95-02, 19-; Allion: 01-21; Premio: 01-21; Corolla Axio: 06-; Corolla Fielder: 00-; Blade: 06-12; Mark X Zio: 07-13; Kluger: 00-; Tacoma TRD Pro: 15-; GR Corolla: 22-; Urban Cruiser EV: 25-; bZ5: 25-; bZ7: 26-; Corolla Hatchback: 18-; Fortuner Legender: 21-; Hilux Champ: 23-; Innova Zenix: 22-; Veloz: 21-; Wigo: 13-23, 23-; Agya: 13-23, 23-; Calya: 16-; Raize Hybrid: 25-",
"Lexus": "LS 400: 89-00; SC 400: 91-00; GS 300: 93-05; ES 300: 91-01; RX 300: 98-03; IS 200: 98-05; GX 470: 02-09; LX 470: 98-07; RC F: 14-; IS F: 07-14; GS F: 15-20",
"Honda": "Beat: 91-96; Today: 85-98; Logo: 96-01; Capa: 98-02; HR-V (1st): 98-06; Avancier: 99-04, 16-; Inspire: 89-; Rafaga: 93-97; Ascot: 89-97; Torneo: 97-02; Edix: 04-09; FR-V: 04-09; Airwave: 05-10; Partner: 96-10; Life: 97-14; Zest: 06-12; N-WGN: 13-19, 19-; N-One: 12-20, 20-; N-Van: 18-; S660: 15-22; Shuttle: 11-22; Grace: 14-20; Breeze: 19-; Envix: 19-; Crider: 13-; e:NP1: 22-; e:NS1: 22-; Accord Hybrid: 13-; Civic Type R: 97-; Odyssey (Japan): 94-; Orthia: 96-02; Concerto: 88-94; Quint: 80-85; Prelude (classic): 78-96; CR-X: 83-91; City Hybrid e:HEV: 22-; 0 Series SUV: 26-; 0 Series Saloon: 26-",
"Nissan": "Bluebird: 9@91-01; Bluebird Sylphy: 00-12; Primera: 90-08; Pulsar: 78-00, 13-18; Laurel: 68-02; Cedric: 60-04; Gloria: 59-04; Cima: 88-; Fuga: 04-22; Stagea: 96-07; Elgrand: 97-; Presage: 98-09; Lafesta: 04-17; Wingroad: 96-18; AD: 82-; March: 82-; Moco: 02-16; Roox: 09-; Kix: 08-12; Dualis: 07-14; Terrano II: 93-06; Mistral: 94-99; Prairie: 82-98; Stanza: 77-92; Sunny (classic): 66-06; Patrol Y60: 87-97; 300ZX: 83-00; 240SX: 88-98; Skyline GT-R: 69-02; Fairlady Z: 69-; Leopard: 80-99; Avenir: 90-05; Expert: 99-06; Liberty: 98-04; Rasheen: 94-00; Hypermini: 99-03; Note e-Power: 16-; Kicks e-Power: 20-; Almera Turbo: 20-; Navara PRO-4X: 21-; N7: 25-; Frontier Pro: 25-; Townstar: 21-",
"Mazda": "323: 80-03; 626: 79-02; 929: 73-97; Xedos 6: 92-99; Xedos 9: 93-02; Millenia: 94-02; Protege: 89-03; MX-3: 91-98; MX-6: 87-97; Familia: 63-03; Capella: 70-02; Atenza: 02-; Axela: 03-; Verisa: 04-15; Biante: 08-18; Carol: 89-; Flair: 12-; AZ-1: 92-95; Cosmo: 67-96; RX-3: 71-78; B-Series: 61-06; Navajo: 91-94; CX-30 EV: 21-; CX-4: 16-; CX-8: 17-23; Mazda2 Hybrid: 22-; Mazda6e: 25-",
"Mitsubishi": "Lancer (classic): 73-00; Colt: 62-13, 23-; Carisma: 95-04; Sigma: 77-96; Diamante: 90-05; 3000GT: 90-00; Starion: 82-90; FTO: 94-00; GTO: 90-01; Space Star: 98-05; Space Wagon: 83-04; Space Runner: 91-02; Grandis: 03-11; Chariot: 83-03; Dion: 00-06; RVR: 91-; Airtrek: 01-08; Pajero iO: 98-07; Pajero Mini: 94-12; Minica: 62-11; Toppo: 98-13; i: 06-13; Strada: 96-; Raider: 05-09; Attrage: 13-; Mirage G4: 13-; Xpander Cross: 19-; Delica D:5: 07-; eK X: 19-; eK Space: 14-; eK X EV: 22-",
"Subaru": "Leone: 71-94; Alcyone: 85-91; Vivio: 92-98; Pleo: 98-; R1: 05-10; R2: 03-10; Stella: 06-; Exiga: 08-18; Trezia: 10-16; Dex: 08-12; Sambar: 61-; Rex: 72-92, 22-; Libero: 83-98; Traviq: 01-04; Impreza WRX STI: 94-21; Levorg Layback: 23-; Forester Wilderness: 22-; Crosstrek Wilderness: 24-; e-Outback: 26-",
"Suzuki": "Esteem: 95-02; Cultus: 83-; Every: 82-; Escudo: 88-; X-90: 95-97; Kei: 98-09; Cervo: 77-09; MR Wagon: 01-16; Lapin: 02-; Palette: 08-13; Solio: 10-; Landy: 07-; Twin: 03-05; Grand Vitara XL-7: 98-06; XL-7: 06-09; APV: 04-; Karimun: 99-; Dzire (global): 17-; S-Presso (global): 19-; Ertiga Hybrid: 22-",
"Daihatsu": "Applause: 89-00; Feroza: 88-99; Rocky (classic): 89-02; YRV: 00-05; Materia: 06-12; Boon: 04-; Esse: 05-11; Sonica: 06-09; Max: 01-05; Naked: 99-04; Atrai: 81-; Wake: 14-22; Cast: 15-23; Mira e:S: 11-; Mira Tocot: 18-; Luxio: 09-; Terios Kid: 98-12",
"Volkswagen": "Golf GTI: 76-; Golf R: 02-; Polo GTI: 98-; Jetta (classic): 79-98; Vento (Europe): 92-98; Santana (classic): 81-; Corrado VR6: 91-95; Karmann Ghia: 55-74; Type 2: 50-79; Type 3: 61-73; Kombi: 50-13; Golf Cabriolet: 79-02, 11-16; Golf Variant: 93-; Passat Variant: 73-; Polo Sedan: 10-20; Polo Vivo: 10-; Up GTI: 17-23; e-Up: 13-23; e-Golf: 14-20; ID.6: 21-; ID. Unyx: 24-; ID. Every1: 27-; T-Roc Cabriolet: 20-; Atlas Peak: 25-; Saveiro: 82-; Voyage: 81-; Parati: 82-12; Quantum: 85-02; Pointer: 94-96; Logus: 93-97; Brasilia: 73-82; SP2: 72-76; Thing: 69-83; Iltis: 78-88; Taro: 89-97; LT: 75-06; Crafter: 06-; Tavendor: 24-; Talagon: 21-; Viloran: 20-; Sagitar: 06-; Phideon: 16-; Tharu: 18-; Tayron: 18-, 24-; T-Cross (India): 25-",
"Audi": "Q2 L e-tron: 19-; Q5 e-tron: 21-; A4 allroad: 09-; A6 allroad: 00-; A8 L: 94-; S3: 99-; S5 Sportback: 09-; S6: 94-; S7: 12-; S8: 96-; SQ5: 13-; SQ7: 16-; SQ8: 19-; RS Q3: 13-; RS Q8: 19-; RS e-tron GT: 21-; TT RS: 09-23; R8 Spyder: 10-24; quattro: 80-91; V8: 88-94; 50: 74-78; 90: 84-91; Cabriolet: 91-00; Coupe (B3): 88-96; A2: 99-05; A4 Cabriolet: 02-09; A3 Cabriolet: 08-20; A6 Avant e-tron: 24-; Q9: 26-; E5 Sportback: 25-",
"BMW": "Z1: 89-91; M1: 78-81; 3.0 CS: 68-75; Isetta: 55-62; 1 Series Coupe: 07-13; 2 Series Coupe: 14-; M2 CS: 20-; M3 CS: 18-; M4 CSL: 22-; M5 CS: 21-; 3 Series Touring: 87-; 5 Series Touring: 91-; 3 Series Compact: 93-04; 6 Series Gran Coupe: 12-18; 8 Series Gran Coupe: 19-; X5 M50i: 19-; X7 M60i: 22-; iX5 Hydrogen: 23-; i3s: 17-22; iX M60: 22-; i5 Touring: 24-; i7 M70: 23-; 3 Series Gran Turismo: 13-20; 5 Series Gran Turismo: 09-17; Neue Klasse iX3: 25-; 3 Series Long Wheelbase: 12-; 7 Series Long Wheelbase: 01-; M3 Touring: 22-; 2002 Turbo: 73-75",
"Mercedes-Benz": "300 SL: 54-63; Pagoda SL: 63-71; W124: 84-97; W126: 79-91; W140: 91-98; W201: 82-93; W108: 65-72; W114: 68-76; W115: 68-76; Ponton: 53-62; Fintail: 59-68; 600: 63-81; SLR McLaren: 03-10; CLK GTR: 98-99; SLK 55 AMG: 04-; C 63 AMG: 07-; E 63 AMG: 06-; S 63 AMG: 06-; G 63 AMG: 12-; GLA 45 AMG: 14-; A 45 AMG: 13-; CLA 45 AMG: 13-; AMG GT Black Series: 20-21; AMG One: 22-; G-Class Electric: 24-; Citan: 12-; T-Class: 22-; EQV: 20-; EQT: 22-; Vaneo: 02-05; X-Class (pickup): 17-20; Unimog: 48-; EQG: 24-; CLA Shooting Brake: 15-; C-Class Estate: 96-; E-Class All-Terrain: 17-; GLB EV: 26-; VLE: 26-",
"Porsche": "912: 65-69; 914: 69-76; 930: 75-89; 964: 89-94; 993: 94-98; 996: 97-05; 997: 04-12; 991: 11-19; 992: 19-; 911 GT3: 99-; 911 Turbo: 75-; 911 Targa: 67-; Cayenne Coupe: 19-; Taycan Cross Turismo: 21-; Panamera Sport Turismo: 17-24; Cayenne Electric: 26-; 718 Spyder: 19-; Cayman GT4: 15-; 550 Spyder: 53-56",
"Opel": "Kadett E: 84-91; Ascona: 70-88; Rekord: 53-86; Senator: 78-93; Monza: 78-86; GT (classic): 68-73; Speedster: 00-05; Sintra: 96-99; Frontera (classic): 91-04; Monterey: 91-99; Astra GTC: 05-18; Insignia Country Tourer: 13-22; Mokka-e: 20-; Corsa-e: 19-; Astra Electric: 23-; Grandland Electric: 24-; Movano: 98-",
"Peugeot": "e-208: 19-; e-2008: 19-; e-308: 22-; e-3008: 24-; e-5008: 24-; 2008 (India): 25-; 309: 85-93; 305: 77-89; 505: 79-99; 604: 75-85; 104: 72-88; 1007: 04-09; 206 CC: 00-07; 207 CC: 07-15; 307 CC: 03-08; 308 CC: 09-15; 407 Coupe: 05-11; 508 RXH: 12-18; 408 X: 22-; Boxer: 94-; Bipper: 08-17; 3008 Hybrid4: 11-16; e-Rifter: 21-",
"Renault": "Clio V6: 01-05; Clio RS: 00-; Megane RS: 04-23; Spider: 95-99; Laguna Coupe: 08-15; Megane CC: 03-15; Scenic RX4: 00-03; Grand Scenic: 04-22; Kangoo Z.E.: 11-; Zoe E-Tech: 20-24; Dacia-built Logan: 04-; Kwid EV: 26-; Triber (2nd): 25-; Duster (India 2nd): 26-; Captur (Russia): 16-; Kaptur: 16-; Oroch: 15-; Stepway: 10-; Symbol: 99-13; Thalia: 99-13; 9: 81-89; 11: 81-89; 12: 69-80; 16: 65-80; 18: 78-86; 20: 75-84; 30: 75-84; Dauphine: 56-67; Floride: 58-68; R8: 62-73; R10: 65-71",
"Fiat": "Uno (classic): 83-95; Ritmo: 78-88; Regata: 83-90; Croma (classic): 85-96; Tempra: 90-98; Brava: 95-01; Albea: 02-12; Palio Weekend: 96-20; Siena (classic): 96-16; 127: 71-83; 128: 69-85; 131: 74-84; 126: 72-00; X1/9: 72-89; 130: 69-77; Dino: 66-73; 850: 64-73; 1100: 53-69; Grande Punto Abarth: 07-10; 500 Hybrid: 25-; 500X Sport: 19-24; Titano: 23-; Egea: 15-; Tipo Cross: 20-; Doblo Cargo: 01-",
"Alfa Romeo": "Montreal: 70-77; Alfetta: 72-87; Giulietta (classic): 54-65, 77-85; 90: 84-87; SZ: 89-91; RZ: 92-93; 8C Spider: 08-10; 4C Spider: 15-20; Giulia Quadrifoglio: 16-; Stelvio Quadrifoglio: 17-",
"Volvo": "S70: 96-00; V40 (classic): 95-04; 440: 87-96; 460: 88-97; 480: 86-95; 340: 76-91; 760: 82-92; 960: 90-98; 140: 66-74; 164: 68-75; 260: 74-82; 780: 85-90; PV444: 44-58; P1800 ES: 71-73; XC90 Recharge: 20-; EM90: 24-",
"Land Rover": "Freelander 2: 06-15; Range Rover Classic: 70-96; Range Rover P38: 94-02; Discovery 3: 04-09; Discovery 4: 09-16; Defender 90: 83-; Defender 110: 83-; Defender 130: 22-; Range Rover Electric: 26-",
"Jaguar": "XJ (X350): 03-09; XJ (X351): 09-19; XK8: 96-06; XKR: 98-15; S-Type (classic): 63-68; Mark X: 61-70; XK120: 48-54; XK140: 54-57; XK150: 57-61; C-X75: 10-; Project 7: 15-16; XE SV Project 8: 17-19",
"Ford": "Fiesta ST: 04-22; Focus RS: 02-18; Focus ST: 05-25; Mustang Shelby GT500: 07-; Mustang Shelby GT350: 15-20; Ranger Raptor: 18-; F-150 Raptor: 10-; Bronco (classic): 66-96; Model 48: 35-36; Fairlane: 55-70; Falcon: 60-70, 60-16; Galaxie: 59-74; LTD: 65-86; Torino: 68-76; Pinto: 71-80; Maverick (classic): 69-77; Fairmont: 78-83; Granada (US): 75-82; Tempo: 84-94; Contour: 94-00; Aspire: 94-97; Festiva: 86-02; Windstar: 94-03; Aerostar: 86-97; Excursion: 99-05; Explorer Sport Trac: 00-10; Freestyle (US): 05-07; Taurus X: 07-09; C-Max: 03-19; B-Max: 12-17; Fusion (Europe): 02-12; Ka (Brazil): 14-21; Figo (global): 10-21; Fiesta (India): 05-15; Ikon (India): 99-11; EcoSport (India): 13-22; Endeavour (India): 03-22; Mondeo (China): 22-; Equator: 21-; Evos: 21-; Territory (China): 18-; Ranger PHEV: 25-; Mustang Dark Horse: 24-; Tourneo Custom: 12-; Transit Connect: 02-; Courier (Brazil): 97-13",
"Chevrolet": "Optra Magnum: 07-13; Aveo U-VA: 06-12; Captiva (India): 08-16; Forester (India): 03-07; Cruze (India): 09-17; Spark (India): 07-17; Sail Hatchback: 13-17; Trailblazer (India): 15-17; Camaro ZL1: 12-24; Corvette Z06: 01-; Corvette ZR1: 90-95, 09-13, 25-; Silverado HD: 01-; Chevette: 73-98; Citation: 80-85; Celebrity: 82-90; Lumina: 90-01; Beretta: 87-96; Corsica: 87-96; Caprice: 65-20; Biscayne: 58-72; El Camino: 59-87; S-10: 82-12; Astro: 85-05; Lumina APV: 90-96; Venture: 97-05; Tracker (classic): 89-04; Prizm: 98-02; Metro: 98-01; Opala: 69-92; Monza (Brazil): 82-96; Kadett (Brazil): 89-98; Vectra (Brazil): 93-11; Celta: 00-15; Prisma: 06-19; Cobalt (Brazil): 11-19; Groove: 21-; Seeker: 22-; Menlo: 20-",
"Dodge": "Viper ACR: 99-17; Challenger SRT Hellcat: 15-23; Charger SRT Hellcat: 15-23; Durango SRT: 18-; Dart (classic): 60-76; Polara: 60-73; Monaco: 65-77; Diplomat: 77-89; Aries: 81-89; Daytona: 84-93; Shadow: 87-94; Spirit: 89-95; Dynasty: 88-93; Omni: 78-90; Stealth: 91-96; Super Bee: 68-71; Attitude: 06-; Vision: 16-",
"Jeep": "Commander (India): 22-; Compass Trailhawk: 17-; Wrangler Rubicon: 03-; Wrangler 4xe: 21-; Cherokee (SJ): 74-83; Wagoneer (classic): 63-91; Grand Cherokee (ZJ): 93-98; Comanche: 86-92; Scrambler: 81-86; Willys Jeep Wagon: 46-65; Avenger 4xe: 24-",
"Tesla": "Model 3 Performance: 18-; Model Y Juniper: 25-; Model S Plaid: 21-; Model X Plaid: 21-; Model Y L: 25-",
"Cadillac": "Allante: 87-93; Catera: 96-01; Cimarron: 82-88; BLS: 05-10; XT4 (China): 18-; CT4-V Blackwing: 21-; CT5-V Blackwing: 21-; Escalade ESV: 03-; Escalade EXT: 02-13; Series 62: 40-64",
"Rolls-Royce": "Silver Ghost: 06-26; Phantom (classic): 25-91; Camargue: 75-86; Silver Wraith: 46-59; Silver Dawn: 49-55; Park Ward: 00-02; Black Badge Cullinan: 19-",
"Bentley": "S2: 59-62; T-Series: 65-80; Eight: 84-92; Continental GTC: 06-; Bentayga EWB: 22-; Batur: 23-; Bacalar: 21-; Blower: 29-31",
"Mini": "Countryman E: 24-; Cooper S: 02-; John Cooper Works: 08-; Clubvan: 13-14; Mini Moke: 64-93; Mini Electric: 20-24",
"BYD": "e2: 19-; e3: 20-; e5: 15-20; Qin: 13-; Song: 16-; Song Max: 17-; Yuan: 16-; Tang EV: 18-; S6: 11-17; S7: 14-17; G3: 09-14; L3: 10-14; F0: 08-15; F6: 07-12; M3: 21-; T3: 15-; Han L: 25-; Tang L: 25-; Sealion 8: 25-; Atto 8: 25-; Yuan Pro: 21-; Song Plus DM-i: 21-; Qin Pro: 18-; Destroyer 07: 25-",
"Geely": "Emgrand GT: 15-; Emgrand X7: 11-; Vision: 06-; Binyue: 18-; Haoyue: 20-; Icon: 20-; Xingrui: 20-; Xingyue: 19-; GX7: 12-; MK: 06-; CK: 05-; Atlas Pro: 20-; Geometry A: 19-; Geometry E: 22-; Galaxy E8: 24-; Galaxy Starship 7: 24-; Panda: 23-; Jiaji: 19-; Boyue L: 23-",
"Chery": "Tiggo 3: 14-; Tiggo 5: 14-; Tiggo 4: 17-; Tiggo 5x: 17-; Tiggo 7 Plus: 21-; Arrizo 3: 14-; Arrizo 6: 21-; Arrizo 7: 13-; Fulwin: 08-; A1: 07-15; A3: 08-17; A5: 06-; Eastar: 03-; Cowin: 03-; Fengyun: 08-; Exeed LX: 19-; Fulwin X3: 25-",
"Great Wall": "Deer: 96-13; Safe: 01-10; Florid: 08-; C30: 10-; M4: 12-; Steed: 06-; Coolbear: 09-; King Kong Poer: 21-; Shanhai Poer: 23-",
"Changan": "Benni: 06-; CS15: 15-; CS35: 12-; CS55: 17-; CS75: 13-; CS95: 17-; Raeton: 13-; Alsvin V7: 15-; Eado XT: 13-; Oshan X7: 20-; Nevo A07: 23-; Qiyuan A05: 24-",
"Haval": "M6: 17-; H5: 10-; H8: 13-; F5: 18-; H6 Coupe: 15-; Xiaolong: 23-; Menglong: 23-; H6 GT: 22-; Shenshou: 24-; M6 Plus: 21-",
"NIO": "Onvo L90: 25-; ES8 (3rd): 25-; EC7: 23-; ET5 Touring: 23-",
"XPeng": "G9 (2025): 24-; Mona M03 Max: 24-; P7i: 23-; X9 EREV: 25-",
"Zeekr": "001 FR: 23-; 7GT: 25-; Mix: 24-",
"Proton": "Waja: 00-11; Gen-2: 04-12; Satria: 94-05; Satria Neo: 06-15; Arena: 02-10; Putra: 96-01; Tiara: 96-00; Juara: 01-03; Savvy: 05-11; Inspira: 10-15; Suprima S: 13-; S70: 23-; X50 Flagship: 24-",
"Perodua": "Rusa: 96-07; Nautica: 08-09; Kembara: 98-08; Traz: 25-; QV-E: 26-",
"VinFast": "VF 3 Plus: 25-; President: 21-23; Limo Green: 25-; Minio Green: 25-; Herio Green: 25-; Nerio Green: 25-",
"Lada": "2101: 70-88; 2104: 84-12; 2105: 79-10; 2106: 76-06; 2107: 82-12; 110: 95-14; 111: 98-09; 112: 99-08; Granta Cross: 18-; Vesta Cross: 17-; Largus Cross: 14-; Niva Legend: 21-; Aura: 24-",
"Holden": "Commodore: 78-20; Statesman: 71-10; Caprice: 73-17; Monaro: 68-77, 01-05; Kingswood: 68-80; Torana: 67-80; Gemini: 75-87; Camira: 82-89; Barina: 85-18; Astra: 84-20; Cruze: 09-16; Captiva: 06-18; Colorado: 08-20; Rodeo: 72-08; Ute: 00-17; Calais: 84-20; Equinox: 17-20; Acadia: 18-20; Trailblazer: 12-20; Spark: 10-18; Trax: 13-20; Malibu: 13-16; Volt: 12-15; Berlina: 84-05; Premier: 62-80; FJ: 53-56; EH: 63-65",
"HSV": "GTS: 92-17; Clubsport: 90-17; Maloo: 90-17; Senator: 89-17; Grange: 96-17; Coupe: 01-05; Avalanche: 03-05",
"Ssangyong": "",
"KGM": "Torres EVX: 23-; Actyon (2024): 24-; Rexton Sports: 18-; Korando Turismo: 13-19; Tivoli Grand: 22-; Musso Grand: 19-",
"Ashok Leyland": "Dost: 11-; Stile: 13-15; Partner: 14-; Bada Dost: 21-",
"Eicher": "Polaris Multix: 15-20",
"Mahindra Renault": "Logan: 07-10",
"San Motors": "Storm: 98-03",
"Reva": "i: 01-12; NXR: 09-12",
"Sipani": "Dolphin: 82-95; Montana: 82-95",
"Standard": "Herald: 61-71; Gazel: 72-77; 2000: 85-88",
"Hindustan Motors": "Landmaster: 54-58; Trekker: 97-01; Porter: 97-01; Pushpak: 98-01",
"Rover": "75: 99-05; 45: 99-05; 25: 99-05; 200: 84-99; 400: 90-99; 600: 93-99; 800: 86-99; SD1: 76-86; P6: 63-77; Metro: 80-98; Streetwise: 03-05; CityRover: 03-05",
"Triumph": "Spitfire: 62-80; TR6: 68-76; TR7: 74-81; Stag: 70-77; Dolomite: 72-80; Herald: 59-71; GT6: 66-73",
"Austin": "Mini: 59-00; Healey 3000: 59-67; Allegro: 73-83; Maxi: 69-81; Metro: 80-90; Princess: 75-82; Cambridge: 54-71",
"Sunbeam": "Alpine: 59-68; Tiger: 64-67; Imp: 63-76",
"Hillman": "Imp: 63-76; Minx: 32-70; Avenger: 70-81; Hunter: 66-79",
"Daimler": "Sovereign: 66-92; Double-Six: 72-92; SP250: 59-64; Super V8: 97-09",
"TVR": "Griffith: 91-02, 26-; Chimaera: 92-03; Cerbera: 96-03; Tuscan: 99-06; Sagaris: 04-06; Tamora: 02-07",
"Caterham": "Seven: 73-; 21: 94-99; Project V: 26-",
"Ariel": "Atom: 00-; Nomad: 15-",
"Noble": "M12: 00-08; M400: 04-07; M600: 10-",
"Ginetta": "G40: 10-; G60: 11-15; Akula: 20-",
"Radical": "SR3: 02-; SR8: 05-; Rapture: 16-",
"De Tomaso": "Pantera: 71-92; Mangusta: 67-71; Guara: 93-04; P72: 20-",
"Iso": "Grifo: 63-74; Rivolta: 62-70",
"Lancia ": "",
"Seat": "",
"Zastava": "Yugo: 80-08; Florida: 88-08; 101: 71-08; 10: 05-08",
"Trabant": "601: 64-90; 1.1: 90-91",
"Wartburg": "353: 66-88; 1.3: 88-91",
"Skoda ": "",
"Tatra": "603: 56-75; 613: 74-96; 700: 96-99; 87: 36-50",
"Moskvitch": "412: 67-98; 2141: 86-01; 3: 22-; 408: 64-76",
"GAZ": "Volga: 46-10; Chaika: 59-88; Gazelle: 94-; Sobol: 98-; Siber: 08-10",
"UAZ": "Patriot: 05-; Hunter: 03-; Bukhanka: 65-; Pickup: 08-; Profi: 17-",
"ZAZ": "Zaporozhets: 60-94; Tavria: 87-07; Sens: 02-14; Vida: 12-17; Forza: 11-17",
"Dacia ": "",
"Aro": "10: 80-06; 24: 72-06",
"Iran Khodro": "Samand: 02-; Dena: 11-; Paykan: 67-05; Runna: 08-; Tara: 22-",
"Saipa": "Pride: 93-20; Tiba: 09-; Quick: 17-; Shahin: 20-; Saina: 15-",
"Kantanka": "Omama: 16-; Onantefo: 16-",
"Innoson": "IVM Fox: 14-; IVM Umu: 15-; IVM G80: 19-",
"Troller": "T4: 99-21; Pantanal: 06-08",
"Gurgel": "BR-800: 88-91; X-12: 75-89; Supermini: 92-94",
"Holden ": "",
"Tata ": "",
"Bajaj": "Qute: 18-",
"Kia ": "",
"Isuzu ": "",
"Hino": "",
"Maxus ": "",
"Polestar ": "",
"Fisker ": "",
"Canoo": "Lifestyle Vehicle: 22-24",
"Faraday Future": "FF 91: 23-",
"Aptera": "Launch Edition: 25-",
"Lordstown": "Endurance: 22-23",
"Karma": "Revero: 17-; GS-6: 21-; Kaveya: 26-",
"DeLorean": "DMC-12: 81-83; Alpha5: 25-",
"Shelby": "Cobra: 62-67; GT350: 65-70; GT500: 67-70",
"Saleen": "S7: 00-09; S1: 19-",
"SSC": "Ultimate Aero: 04-13; Tuatara: 20-",
"Hennessey": "Venom F5: 21-; Venom GT: 11-17",
"Studebaker": "Champion: 39-58; Lark: 59-66; Avanti: 62-63; Hawk: 56-64",
"AMC": "Gremlin: 70-78; Pacer: 75-80; Javelin: 68-74; Hornet: 70-77; Eagle: 80-88; Matador: 71-78; Concord: 78-83; Spirit: 79-83",
"DeSoto": "Firedome: 52-59; Adventurer: 56-60",
"Packard": "Clipper: 41-57; Caribbean: 53-56",
"Edsel": "Corsair: 58-59; Citation: 58",
"Hudson": "Hornet: 51-57; Commodore: 41-52",
"Nash": "Rambler: 50-55; Metropolitan: 53-61",
"Imperial": "Crown: 55-70; LeBaron: 57-75",
"Polski Fiat": "125p: 67-91; 126p: 73-00",
"FSO": "Polonez: 78-02; Syrena: 57-83",
"Dongfeng ": "",
"Hongqi ": "",
"Lynk & Co ": "",
"Jetour ": "",
"Leapmotor ": "",
"IM Motors": "L7: 22-; LS7: 23-; L6: 24-; LS6: 23-",
"Rising Auto": "R7: 22-; F7: 23-",
"HiPhi": "X: 21-24; Z: 22-24; Y: 23-24",
"Seres": "SF5: 19-; 3: 20-; 5: 23-; 7: 24-",
"Luxeed": "S7: 23-; R7: 24-",
"Stelato": "S9: 24-",
"Maextro": "S800: 25-",
"Smart ": "",
"Tank ": "",
"Jaecoo ": "",
"Forthing": "T5 EVO: 21-; U-Tour: 24-; Friday: 23-",
"Soueast": "DX3: 16-; DX7: 15-; S07: 25-",
"Haima": "M3: 13-; S5: 14-; 7X: 20-; 8S: 19-",
"Brilliance": "V5: 12-; H530: 11-; FRV: 08-; BS4: 06-",
"Zotye": "T600: 13-; Z100: 13-; SR9: 16-",
"Hawtai": "Boliger: 07-; Santa Fe (Hawtai): 02-",
"JMC": "Vigus: 17-; Yusheng: 10-",
"Foton": "Tunland: 12-; Sauvana: 15-; Gratour: 15-; View: 08-",
"Kaiyi": "X3: 20-; X7: 23-; Showjet: 22-",
"Baic ": "",
"Arcfox": "Alpha S: 21-; Alpha T: 20-; Kaola: 24-",
"Yuanhang": "Y6: 24-; H8: 24-",
"Livan": "X3 Pro: 23-; 7: 23-",
"Radar": "RD6: 22-; Horizon: 25-",
"Smart EQ": "Fortwo EQ: 17-24; Forfour EQ: 17-21",
"Hummer EV": "",
# ---- batch 3: more current and recent models per market
"Toyota ": "Crown Crossover: 22-; Crown Sport: 23-; Crown Signia: 24-; Crown Estate: 25-; Land Cruiser Mini: 26-; Hilux Revo: 15-; Hilux Vigo: 04-15; Corolla Rumion: 07-15; Sequoia TRD Pro: 22-; Tundra TRD Pro: 21-; Prius Prime: 12-; RAV4 Prime: 20-; Highlander Hybrid: 05-; Sienna Hybrid: 20-; Camry Hybrid: 06-; Yaris Ativ: 22-; Vios Hybrid: 23-; Probox: 02-; Succeed: 02-20; Pixis Epoch: 12-; Roomy: 16-; Tank: 16-20; Spade: 12-20; iQ: 08-16; Verso: 09-18; Avensis: 97-18; Celsior: 89-06; Aristo: 91-05; Altezza: 98-05; Comfort: 95-17; JPN Taxi: 17-; Coaster: 69-; Dyna: 59-; ToyoAce: 54-; Land Cruiser Prado 150: 09-23; Fortuner GR Sport: 22-; Glanza (2019): 19-22; Urban Cruiser (2025): 25-",
"Honda ": "Civic e:HEV: 22-; Accord e:HEV: 23-; CR-V e:HEV: 23-; HR-V e:HEV: 21-; Fit e:HEV: 20-; Stepwgn: 96-; Elysion: 04-16; Vamos: 99-18; Acty: 77-21; Zest Spark: 08-12; Crossroad: 07-10, 13-; Jade: 13-20; UR-V: 17-; Spirior: 14-18; Gienia: 16-; Lingpai: 18-; Inspire (China): 18-; Avancier (China): 16-; Breeze (China): 19-; e:NP2: 24-; Ye P7: 25-; Ye S7: 25-; GT: 24-",
"Hyundai ": "Elantra Hybrid: 21-; Sonata N Line: 21-; Tucson Hybrid: 21-; Santa Fe Hybrid: 21-; Kona Electric: 18-; Ioniq Electric: 16-22; Creta N Line: 24-; Venue N Line: 22-; i20 N Line: 21-; Verna Turbo: 23-; Alcazar (2024): 24-; Stargazer: 22-; Creta Grand: 22-; Custin: 21-; Lafesta: 18-; Encino: 17-; Reina: 17-21; Celesta: 17-21; Mistra: 13-; ix25: 14-; ix35: 09-15; HB20: 12-; Grand Santa Fe: 13-18; Maxcruz: 13-18; Aslan: 14-17; Dynasty: 96-05; Marcia: 95-98; Grandeur XG: 98-05; Avante: 90-; Porter: 77-",
"Kia ": "EV6 GT: 22-; EV9 GT: 25-; Niro EV: 18-; Soul EV: 14-; Sportage Hybrid: 21-; Sorento Hybrid: 20-; Seltos (2nd): 25-; Sonet (2nd): 26-; Carnival Hybrid: 24-; K7: 09-21; K3 GT: 18-; Stinger GT: 17-23; Morning: 04-; Bongo: 80-; Mohave Borrego: 08-; KX1: 18-; KX3: 15-; KX5: 16-; KX7: 17-; Sportage R: 10-15; Picanto GT-Line: 17-; EV2: 26-; Syros EV: 26-",
"Nissan ": "Patrol Nismo: 21-; GT-R Nismo: 14-25; Leaf Nismo: 18-; Juke Nismo: 13-; Note Aura: 21-; Roox (Japan): 20-; Clipper: 03-; NV200: 09-; NV350: 12-; Caravan: 73-; Bluebird (Taiwan): 01-; Livina: 06-; Grand Livina: 06-; Lannia: 15-; Sylphy e-Power: 22-; Venucia D60: 17-; Venucia T70: 15-; Paladin: 99-15; Qashqai e-Power: 22-; X-Trail e-4orce: 22-; Magnite (2024): 24-; Murano (2025): 24-",
"Volkswagen ": "Golf GTE: 14-; Golf GTD: 09-; Passat GTE: 15-; Arteon Shooting Brake: 20-24; Tiguan R: 20-24; T-Roc R: 19-; Touareg R: 20-; ID.4 GTX: 21-; ID.5 GTX: 21-; ID.7 Tourer: 24-; Polo Sedan (Russia): 10-20; Jetta GLI: 84-; Atlas Cross Sport (2024): 24-; Taos: 21-; Tarok: 19-; Amarok V6: 16-22; Caddy California: 20-; California: 88-; Grand California: 19-; Multivan eHybrid: 21-",
"Mercedes-Benz ": "A-Class Sedan: 18-; A-Class L: 18-; C-Class L: 08-; E-Class L: 06-; GLA (China): 20-; GLC L: 22-; EQA (China): 21-; EQB (China): 21-; EQE AMG: 22-; EQS AMG: 22-; S-Class Coupe: 99-21; S-Class Cabriolet: 15-21; SL 63 AMG: 08-; AMG GT Coupe (2nd): 23-; AMG GT 63 S E Performance: 22-; Maybach SL: 23-; Maybach EQS SUV: 23-; CLA Coupe (EV): 25-; CLE Cabriolet: 24-; Vito Tourer: 14-; eVito: 18-; eSprinter: 19-",
"BMW ": "X1 (China L): 16-; X3 (China L): 24-; X5 Le: 22-; 5 Series Li: 05-; 3 Series Li: 12-; i3 (China): 22-; iX3 (China): 20-25; M3 Competition: 20-; M4 Competition: 20-; M8 Competition: 19-; X3 M Competition: 19-; XM Label: 23-; 1 Series Sedan: 17-; 2 Series Gran Tourer: 15-22; Z4 M40i: 18-; 4 Series Gran Coupe: 14-; i4 M50: 21-",
"Audi ": "A3 L: 23-; A4 L: 09-; A6 L: 99-; Q5 L: 18-; Q5 L Sportback: 21-; Q3 Sportback (China): 20-; Q2 L: 18-; Q4 e-tron Sportback: 21-; Q8 Sportback e-tron: 22-; A5 Sportback: 09-; A5 Cabriolet: 09-24; S1: 14-18; RS 6 Avant: 02-; RS 4 Avant: 99-; Q6 Sportback e-tron: 24-; A6 Sportback e-tron: 24-",
"Ford ": "Mustang EcoBoost: 15-; Mustang Mach 1: 21-; F-250: 99-; F-350: 99-; F-450: 08-; E-Series: 61-; Transit Connect (US): 10-23; Explorer ST: 20-; Edge ST: 19-23; Escape PHEV: 20-; Ranger Wildtrak: 12-; Everest Titanium: 22-; Raptor R: 23-; Expedition Max: 18-; Territory (2023): 23-; Equator Sport: 21-; Mondeo Sport: 23-; Kuga PHEV: 20-; Explorer (EV, 2024): 24-; Capri EV: 24-",
"Chevrolet ": "Silverado 2500HD: 01-; Silverado 3500HD: 01-; Colorado ZR2: 17-; Tahoe Z71: 00-; Suburban HD: 00-; Express Van: 96-; Trailblazer (2021): 20-; Trax (2024): 23-; Equinox (2025): 25-; Blazer RS: 19-; Camaro SS: 10-24; Corvette E-Ray: 24-; Brightdrop 400: 22-; Montana (2023): 23-; Onix Plus: 19-; Tracker (Brazil): 20-; Spin Activ: 14-",
"BYD ": "Seal (2025): 25-; Seal 07: 25-; Atto 3 (2025): 25-; Dolphin Surf: 25-; Song Pro DM-i: 23-; Qin Plus DM-i: 21-; Qin Plus EV: 21-; Han DM: 20-; Tang DM: 18-; Yuan Plus EV: 22-; Seagull (2025): 25-; Sealion 6: 24-; Shark 6: 24-; eMAX 9: 25-; Song L DM-i: 24-; Xia: 25-; Haishi 06: 24-",
"Geely ": "EX2: 25-; Galaxy E2: 25-; Galaxy M9: 25-; Galaxy A7: 25-; Galaxy Cruiser: 25-; Xingyue S: 22-; Monjaro (China): 21-; Okavango (2024): 24-; Coolray (2023): 23-; Emgrand (2021): 21-; Preface (2024): 24-; Tugella (2024): 24-; Jiaji L: 22-; Starray EM-i: 24-",
"Chery ": "Tiggo 9 PHEV: 24-; Tiggo 8 Pro Max: 22-; Tiggo 4 Hybrid: 24-; Tiggo 7 PHEV: 24-; Arrizo 8 Pro: 24-; Fulwin T6: 24-; Fulwin T9: 24-; Fulwin A8: 24-; iCar 03: 24-; iCar V23: 24-; QQ Ice Cream 2: 24-",
"Changan ": "UNI-Z: 24-; UNI-S: 25-; CS75 Pro: 22-; CS85 Coupe: 19-; X5 Plus: 22-; Eado Plus: 20-; Lamore: 23-; Qiyuan A07: 24-; Qiyuan Q05: 24-; Qiyuan E07: 25-; Kaicene F70: 19-",
"Leapmotor ": "C10 REEV: 24-; C16 REEV: 24-; B01: 25-; B05: 25-; C11 REEV: 23-; T03 (Europe): 24-",
"Li Auto ": "L6 Max: 24-; L7 Max: 23-; L8 Max: 22-; L9 Max: 22-; i8 Max: 25-",
"XPeng ": "G3i: 21-; P5 (2024): 24-; G6 (2025): 25-; G9 Max: 22-; P7 Wing: 23-",
"Aito ": "M5 EV: 23-; M7 Ultra: 24-; M9 EREV: 23-; M8 EREV: 25-",
"Avatr ": "11 EREV: 24-; 12 EREV: 24-; 07 EREV: 24-",
"Deepal ": "S07: 23-; L07: 24-; S05 EREV: 24-; E07: 24-; G318 EREV: 24-",
"Jetour ": "X70 Plus: 20-; X90 Plus: 21-; X50: 23-; Shanhai L7: 24-; Shanhai L9: 24-; Traveller C-DM: 25-",
"Exeed ": "LX (2024): 24-; Yaoguang: 24-; Sterra ET: 24-; Sterra ES (2025): 25-",
"Omoda ": "C7: 25-; E5 (2025): 25-; 9: 25-",
"Jaecoo ": "7 SHS: 24-; 6: 25-",
"Haval ": "H6 HEV: 21-; H6 PHEV: 23-; Jolion Pro: 24-; Jolion HEV: 21-; Big Dog 2: 23-; Raptor PHEV: 23-; Xiaolong Max: 23-",
"Tank ": "300 Hi4-T: 24-; 500 Hi4-T: 23-; 700 Hi4-T: 24-",
"Wuling ": "Hongguang S: 13-; Hongguang S3: 17-; Victory: 23-; Starlight S: 24-; Bingo Plus: 25-; Cloud EV: 24-",
"MG ": "Hector (2023): 23-; Astor (2024): 24-; ZS EV (2022): 22-; Gloster (2025): 25-; M9 EV: 25-; Majestor (2025): 25-; Cyberster (India): 25-; MG ES5: 25-; MG7 Trophy: 23-; MG4 XPower: 23-; MG5 EV: 20-; One (2022): 22-; Mulan: 22-; MGS5 EV: 25-; MGS6 EV: 26-; HS PHEV: 20-",
"Tata ": "Harrier (2023): 23-; Safari (2023): 23-; Nexon (2023): 23-; Nexon EV (2023): 23-; Punch (2025): 25-; Altroz (2025): 25-; Tiago NRG: 18-; Tigor iCNG: 22-; Tiago iCNG: 22-; Punch iCNG: 23-; Safari Dark: 21-; Harrier Dark: 21-; Sierra EV: 26-; Curvv Dark: 25-",
"Mahindra ": "XUV700 (2025): 25-; Scorpio-N Z8: 22-; Thar Earth: 24-; Bolero Maxx Pik-Up: 23-; Marazzo (2023): 23-; XUV400 Pro: 24-; Thar Roxx AX7L: 24-; BE 6 Batman Edition: 25-; XEV 7e: 26-",
"Maruti Suzuki ": "Swift (2024) Hybrid: 24-; Dzire (2024) CNG: 24-; Brezza CNG: 23-; Grand Vitara CNG: 23-; Fronx Hybrid: 26-; Baleno CNG: 22-; Ertiga CNG: 18-; XL6 CNG: 23-; Wagon R CNG: 19-; Celerio CNG: 21-; Alto K10 CNG: 22-; S-Presso CNG: 19-; Eeco CNG: 10-; Ignis (2020): 20-; Ciaz (2018): 18-",
"Skoda ": "Octavia RS: 00-; Superb iV: 20-; Kodiaq RS: 18-; Enyaq Coupe: 22-; Kushaq Monte Carlo: 22-; Slavia Monte Carlo: 24-; Fabia RS: 03-14; Kamiq (2024): 24-; Scala (2024): 24-; Karoq (2022): 22-",
"Kia India": "",
"Renault ": "Kiger (2025): 25-; Triber (2025): 25-; Kwid (2019): 19-; Duster (2026): 26-; Clio (2026): 26-; Captur (2024): 24-; Arkana (2025): 25-; Austral (2025): 25-",
"Citroen ": "C3 (India) Aircross: 23-; eC3 Shine: 24-; Basalt X: 25-; C3 X: 25-; Aircross X: 25-; C3 Aircross (2024 Europe): 24-; e-C3: 24-; e-C4: 20-; e-C4 X: 23-; e-Berlingo: 21-",
"Jeep ": "Compass (2021): 21-; Meridian (2025): 25-; Grand Cherokee (India): 22-; Wrangler (India 2024): 24-; Commander (Brazil): 21-; Avenger (2025): 25-; Compass (2025 Europe): 25-",
"Volvo ": "EX30 Cross Country: 25-; XC90 (2025): 25-; XC60 Recharge: 20-; V60 Cross Country: 15-; V90 Cross Country: 17-; S90 (China): 16-; ES90 (China): 25-",
"Lexus ": "ES 300h: 12-; NX 350h: 21-; RX 500h: 22-; LM 350h: 23-; LX 600: 21-; GX 550: 23-; UX 300e: 20-; RZ 450e: 22-; LBX Morizo RR: 24-; IS 500: 21-; LC 500 Convertible: 20-",
"Porsche ": "Taycan Turbo GT: 24-; Taycan 4S: 19-; Macan 4 Electric: 24-; Macan Turbo Electric: 24-; Cayenne E-Hybrid: 10-; Panamera 4 E-Hybrid: 13-; 911 Dakar: 22-24; 911 S/T: 23-; 911 GT3 RS: 03-; 718 Cayman GT4 RS: 21-",
"Land Rover ": "Defender 90 (2020): 20-; Defender 110 (2020): 20-; Defender Octa: 24-; Range Rover Sport SV: 23-; Range Rover SV: 22-; Range Rover Velar (2023): 23-; Range Rover Evoque (2023): 23-; Discovery Sport (2023): 23-",
"Tesla ": "Model 3 Highland: 23-; Model Y Standard: 25-; Model 3 Standard: 25-; Cybertruck Cyberbeast: 23-",
"Rivian ": "R1T Quad: 24-; R1S Quad: 24-",
"Lucid ": "Air Sapphire: 23-; Air Pure: 23-; Gravity X: 25-",
"Polestar ": "2 (2024): 24-; 4 (2025): 25-; 7: 27-",
"Zeekr ": "001 (2025): 25-; 007 GT: 25-; 7X (2025): 25-; X (Europe): 24-",
"NIO ": "ET5T: 23-; ES6 (2023): 23-; EC6 (2023): 23-; ES8 (2023): 23-; ET7 (2024): 24-; Onvo L80: 25-",
"Hongqi ": "H6: 22-; HS3: 24-; HS6: 25-; E-QM5: 22-; EH7 (2025): 25-; EHS7: 24-; Guoya: 25-",
"VinFast ": "VF 6 Plus: 25-; VF 7 Plus: 24-; VF 8 Plus: 22-; VF 9 Plus: 23-; EC Van: 25-",
"Proton ": "eMas 5: 25-; X50 (2024): 24-; Saga (2023): 23-; Persona (2022): 22-",
"Perodua ": "Bezza (2020): 20-; Myvi (2022): 22-; Axia (2023) E: 23-",
}


def ordinal(n: int) -> str:
    suf = "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def year(yy: str) -> int | None:
    if not yy:
        return None
    y = int(yy)
    return y if y > 1000 else (1900 + y if y >= 30 else 2000 + y)


def parse() -> list[dict]:
    merged = {k: v for k, v in T.items()}
    canon = {k.lower(): k for k in T}
    for make, body in EXTRA.items():
        mk = canon.get(make.strip().lower(), make.strip())
        if body.strip():
            merged[mk] = (merged.get(mk, "") + "; " + body).strip("; ")
    out, seen = [], set()
    for make, body in merged.items():
        for item in re.split(r";\s*", " ".join(body.split())):
            if not item.strip():
                continue
            model, spec = item.rsplit(":", 1) if re.search(r":\s*[\d@ ,\-]+$", item) else (item, "")
            model, spec = model.strip(), spec.strip()
            start = 1
            if "@" in spec:
                n, spec = spec.split("@", 1); start = int(n)
            spans = [s.strip() for s in spec.split(",") if s.strip()]
            for k, s in enumerate(spans):
                a, b = (s.split("-") + [""])[:2] if "-" in s else (s, s)
                ys, ye = year(a), year(b)
                if (make, model, ys) in seen:
                    continue
                seen.add((make, model, ys))
                gen = f"{ordinal(start + k)} gen" if len(spans) > 1 or start > 1 else "1st gen" if ys and ys >= 1990 else None
                out.append(dict(make=make, model=model, gen=gen, years=[ys, ye]))
    return out


def main():
    E = parse()
    seen = {(e["make"].lower(), e["model"].lower()) for e in E}
    # merge the 954-entry routing catalogue (model names only) so this is a superset
    old = json.loads((ROOT / "data_meta" / "car_catalogue.json").read_text(encoding="utf8"))
    canon = {m.lower(): m for m in T}                      # FIAT -> Fiat, MINI -> Mini, HUMMER -> Hummer, ...
    canon.update({"citroën": "Citroen", "mercedes": "Mercedes-Benz"})
    added = 0
    for make, models in old.items():
        mk = canon.get(make.lower(), make)
        for mo in models:
            if (mk.lower(), mo.lower()) not in seen:
                E.append(dict(make=mk, model=mo, gen=None, years=[None, None])); seen.add((mk.lower(), mo.lower())); added += 1
    for e in E:
        ys, ye = e["years"]
        e["years_label"] = "" if ys is None else f"{ys}–{ye if ye else 'present'}" if ye != ys else f"{ys}"
        e["label"] = f"{e['make']} {e['model']}"
    E.sort(key=lambda e: (e["make"].lower(), e["model"].lower(), e["years"][0] or 0))
    meta = dict(description="World car catalogue (make / model / generation, up to 2026) for the world-car layer. "
                            "Not used by the 196-class Stanford Cars models or the open-set routing.",
                n_entries=len(E), n_models=len({(e["make"], e["model"]) for e in E}), n_makes=len({e["make"] for e in E}),
                merged_from_routing_catalogue=added)
    p = ROOT / "data_meta" / "world_catalogue.json"
    p.write_text(json.dumps(dict(meta=meta, entries=E), ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    print(meta)


if __name__ == "__main__":
    main()
