"""
apps/ml_inference/test_pu_labels.py — parser label PU. Fixture diambil dari teks PDF yang
benar-benar tercetak pada recon Sesi D (K001 dan KA01 lengkap pada bagian geometri;
bagian "Operating conditions"/kerusakan untuk KA01, KA04, KI01, KI04, KB23).
"""
from django.test import SimpleTestCase

from apps.ml_inference import pu_labels as pl

GEOMETRY = """
Profile of rolling bearing damage (Bearing: {code})
        Category                          Unit        Specification/Value
        Bearing Type                        -         deep groove ball bearing
        Bearing designation
        (dimension series, bore code)       -         6203
        Pitch circle diameter              mm         29.05
        Number of rolling elements         pc.        8
        Rolling element diameter           mm         6.75
        Length of rolling element          mm         6.75
        Nominal pressure angle              °         0
        Bearing code                        -         {code}
"""

K001_TAIL = """
   Operating conditions
        Lifetime                            h         >50
"""

KA01_TAIL = """
 Operating conditions
 Lifetime h:min
 Load N
 artificial damage, bearing was not operated
 Dynamic equivalent load N
 Comment -
 Number of damages 1
 Category Damage 1 Damage 2 Damage 3
 Type of Damage
 Mode artificial
 Sub-mode n/a
 Symptom n/a
 Component OR
 Damage Position of damage raceway
 Damage method EDM machining
Legend
OR: outer ring
IR: inner ring
"""

KA04_TAIL = """
 Operating conditions
 Lifetime h:min 12:01
 Load N 3800
 Rotational speed min 2900
 Comment - n/a
 Number of damages 1
 Mode fatigue
 Sub-mode n/a
 Symptom Pitting
 Component OR
 Damage method lifetime test
"""

KI01_TAIL = """
 Operating conditions
 Lifetime h:min
 artificial damage, bearing was not operated
 Number of damages 1
 Mode artificial
 Component IR
 Damage method n/a
"""

KI04_TAIL = """
 Operating conditions
 Lifetime h:min 12:01
 Load N 3800
 Number of damages 2
 Category Damage 1 Damage 2 Damage 3
 Mode fatigue plastic deformation
 Sub-mode fatigue below the deformation by foreign
 Symptom Pitting
 Component IR OR
 Damage method lifetime test lifetime test
"""

KB23_TAIL = """
 Operating conditions
 Lifetime h:min 21:01
 Number of damages 3
 Mode fatigue fatigue fatigue
 Symptom Pitting Pitting Pitting
 Component IR IR OR
 Damage method lifetime test lifetime test lifetime test
"""


def _doc(code, tail):
    return GEOMETRY.format(code=code) + tail


class ParseTests(SimpleTestCase):
    def test_healthy_k001(self):
        e = pl.parse_pdf_text(_doc("K001", K001_TAIL))
        self.assertEqual((e["code"], e["label"], e["origin"], e["components"]), ("K001", "HEALTHY", "healthy", []))
        self.assertEqual(e["lifetime"], ">50")
        self.assertIsNone(e["n_damages"])

    def test_geometry_from_pdf(self):
        g = pl.parse_pdf_text(_doc("K001", K001_TAIL))["geometry"]
        self.assertEqual((g["ball_diameter_mm"], g["pitch_diameter_mm"], g["num_balls"], g["contact_angle_deg"]),
                         (6.75, 29.05, 8, 0.0))
        self.assertTrue(pl.geometry_complete(pl.parse_pdf_text(_doc("K001", K001_TAIL))))

    def test_artificial_outer_race(self):
        e = pl.parse_pdf_text(_doc("KA01", KA01_TAIL))
        self.assertEqual((e["label"], e["origin"], e["components"], e["lifetime"]), ("OUTER_RACE", "artificial", ["OR"], None))
        self.assertEqual(e["damage_method"], "EDM machining")

    def test_real_outer_race_with_lifetime(self):
        e = pl.parse_pdf_text(_doc("KA04", KA04_TAIL))
        self.assertEqual((e["label"], e["origin"], e["lifetime"], e["n_damages"]), ("OUTER_RACE", "real", "12:01", 1))

    def test_artificial_inner_race(self):
        e = pl.parse_pdf_text(_doc("KI01", KI01_TAIL))
        self.assertEqual((e["label"], e["origin"], e["components"]), ("INNER_RACE", "artificial", ["IR"]))

    def test_ki04_is_mixed_despite_ki_prefix(self):
        e = pl.parse_pdf_text(_doc("KI04", KI04_TAIL))
        self.assertEqual((e["label"], e["origin"], e["components"], e["n_damages"]), ("MIXED", "real", ["IR", "OR"], 2))

    def test_kb23_mixed_three_damages(self):
        e = pl.parse_pdf_text(_doc("KB23", KB23_TAIL))
        self.assertEqual((e["label"], e["components"], e["n_damages"], e["lifetime"]), ("MIXED", ["IR", "IR", "OR"], 3, "21:01"))

    def test_missing_code_raises(self):
        with self.assertRaises(ValueError):
            pl.parse_pdf_text("tidak ada kode bearing di sini")

    def test_component_regex_ignores_prose(self):
        text = _doc("KA01", KA01_TAIL).replace(" Component OR", " Component of the machine\n Component OR")
        self.assertEqual(pl.parse_pdf_text(text)["components"], ["OR"])


class DeriveLabelTests(SimpleTestCase):
    def _e(self, code, comps):
        return {"code": code, "components": comps}

    def test_rules(self):
        self.assertEqual(pl.derive_label(self._e("K003", [])), "HEALTHY")
        self.assertEqual(pl.derive_label(self._e("KA99", [])), "UNKNOWN")      # tanpa kerusakan tetapi bukan K00n
        self.assertEqual(pl.derive_label(self._e("KA09", ["OR"])), "OUTER_RACE")
        self.assertEqual(pl.derive_label(self._e("KI09", ["IR", "IR"])), "INNER_RACE")
        self.assertEqual(pl.derive_label(self._e("KX01", ["RE"])), "BALL")
        self.assertEqual(pl.derive_label(self._e("KX02", ["OR", "RE"])), "MIXED")

    def test_compact_lines(self):
        self.assertEqual(pl.compact_lines("  a   b \n\n   \n c"), ["a b", "c"])
