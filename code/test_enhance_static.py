"""Focused geography tests owned by enhance; no network or evaluation labels."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np

from enhance_static import (BUFFER_KM, CLASSES, EARTH_KM, city_elevation, densify_segment,
                           landcover_summary, spherical_km, tile_url, xyz)


class StructuralTests(unittest.TestCase):
    def test_spherical_distance_and_dateline(self):
        self.assertAlmostEqual(float(spherical_km(0, 0, 0, 0)), 0)
        self.assertAlmostEqual(float(spherical_km(0, 0, 0, 1)), np.pi * EARTH_KM / 180)
        self.assertAlmostEqual(float(spherical_km(0, 179.5, 0, -179.5)), np.pi * EARTH_KM / 180)

    def test_coastline_spacing(self):
        a, b = (-90, 30), (-89, 31)
        samples = np.vstack([densify_segment(a, b), xyz(b[1], b[0])])
        distances = 2 * EARTH_KM * np.arcsin(np.linalg.norm(np.diff(samples, axis=0), axis=1) / 2)
        self.assertLessEqual(distances.max(), .50000001)
        np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1)

    def test_negative_tile_coordinates_use_floor(self):
        self.assertIn('_N39W078_', tile_url(39.1, -77.04))
        self.assertIn('_N24W084_', tile_url(24.555, -81.7522))

    def test_landcover_nodata_is_not_a_real_class(self):
        # Four small pixels are inside the circle; half the valid area is built.
        transform = (.001, 0, -.001, 0, -.001, .001)
        values = np.array([[10, 50], [50, 0]], dtype='uint8')
        out = landcover_summary([(values, transform)], .0005, -.0005)
        self.assertEqual(out['landcover_station_class'], 10)
        self.assertAlmostEqual(out['landcover_valid_fraction_1km'], .75)
        self.assertAlmostEqual(out['landcover_built_fraction_1km'], 2/3, places=8)
        self.assertAlmostEqual(sum(out[f'landcover_{n}_fraction_1km'] for n in CLASSES.values()), 1)

    def test_outside_circle_is_excluded(self):
        transform = (1., 0, -.5, 0, -1., .5)
        values = np.array([[30, 50]], dtype='uint8')
        out = landcover_summary([(values, transform)], 0, 0)
        self.assertEqual(out['landcover_valid_pixels_1km'], 1)
        self.assertEqual(out['landcover_grass_fraction_1km'], 1)

    def test_unknown_class_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            landcover_summary([(np.array([[111]],dtype='uint8'), (.001,0,-.0005,0,-.001,.0005))], 0, 0)

    def test_elevation_cache_preserves_bad_source_date_without_inventing_one(self):
        city = SimpleNamespace(city_id='city', lat=39., lon=-77.)
        entry = {'parameters':{'x':-77.,'y':39.}, 'response':{
            'location':{'x':-77.,'y':39.}, 'value':'125.5','resolution':1.,
            'attributes':{'AcquisitionDate':'0/5/2024'}}}
        with TemporaryDirectory() as folder:
            (Path(folder)/'city.json').write_text(json.dumps(entry),encoding='utf-8')
            result=city_elevation(city,Path(folder),offline=True)
            self.assertEqual(result['elevation_m'],125.5)
            self.assertEqual(result['elevation_acquisition_date'],'unknown')
            self.assertEqual(result['elevation_acquisition_date_raw'],'0/5/2024')

    def test_elevation_response_coordinate_must_match(self):
        city = SimpleNamespace(city_id='city', lat=39., lon=-77.)
        entry = {'parameters':{'x':-77.,'y':39.}, 'response':{
            'location':{'x':-76.99,'y':39.}, 'value':'125.5','resolution':1.,'attributes':{}}}
        with TemporaryDirectory() as folder:
            (Path(folder)/'city.json').write_text(json.dumps(entry),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'coordinates'):
                city_elevation(city,Path(folder),offline=True)


if __name__ == '__main__':
    unittest.main()
