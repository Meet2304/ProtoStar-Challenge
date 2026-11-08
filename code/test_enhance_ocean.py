"""Enhancement-owned cutoff and aggregation tests; no network or labels."""
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from common import DataRuleViolation
from enhance_ocean import (BOXES, END, EXPECTED_MONTHS, load_ocean_for_origin,
                           origin_snapshot, parameters, regional_mean, validate_months)


def sample_history():
    dates = EXPECTED_MONTHS
    table = pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'year':dates.year,'month':dates.month})
    for name in BOXES:
        table[f'{name}_sst_c'] = (dates.year-1950)*.01 + dates.month*.1 + 20
    return table


class OceanTests(unittest.TestCase):
    def test_exact_endpoint(self):
        self.assertEqual(END, '2018-12-01T00:00:00Z')
        for box in BOXES.values():
            self.assertEqual(parameters(box)['time_end'], END)

    def test_date_cap_rejects_adjacent_future_month(self):
        with self.assertRaises(DataRuleViolation):
            validate_months(EXPECTED_MONTHS.append(pd.DatetimeIndex(['2019-01-01'])))

    def test_missing_or_duplicate_month_rejected(self):
        for dates in (EXPECTED_MONTHS[:-1], EXPECTED_MONTHS.append(EXPECTED_MONTHS[-1:])):
            with self.assertRaises(DataRuleViolation):
                validate_months(dates)

    def test_origin_reference_ignores_all_later_values(self):
        table = sample_history()
        first, _ = origin_snapshot(table, 1983)
        table.loc[table.year > 1983, [f'{n}_sst_c' for n in BOXES]] = 999
        second, annual = origin_snapshot(table, 1983)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(annual.year.max(), 1983)
        self.assertEqual(len(first), 34*12)
        for name in BOXES:
            reference=first.loc[first.year>=1954].groupby('month')[f'{name}_anomaly_c'].mean()
            np.testing.assert_allclose(reference, 0, atol=1e-12)

    def test_active_fold_and_invalid_origin_fail_closed(self):
        with self.assertRaises(DataRuleViolation):
            origin_snapshot(sample_history(), 2019)
        with patch.dict(os.environ, {'PROTOSTAR_FOLD_CUTOFF':'1983'}):
            with self.assertRaises(DataRuleViolation):
                load_ocean_for_origin('not-opened', 1990)
            with self.assertRaises(DataRuleViolation):
                validate_months(EXPECTED_MONTHS)

    def test_area_weighting_and_land_mask(self):
        lat,lon=np.array([0,60]),np.array([200,202])
        values=np.array([[[10,np.nan],[20,20]]],dtype=float)
        result, details=regional_mean(lat,lon,values,{'south':0,'north':60,'west':190,'east':240})
        self.assertAlmostEqual(float(result[0]),15)
        self.assertEqual(details['minimum_valid_cells'],3)

    def test_wrapping_atlantic_zero_meridian(self):
        result,_=regional_mean(np.array([0]),np.array([0,280,282]),np.array([[[30,10,20]]],dtype=float),BOXES['north_atlantic'])
        self.assertAlmostEqual(float(result[0]),20)


if __name__ == '__main__':
    unittest.main()
