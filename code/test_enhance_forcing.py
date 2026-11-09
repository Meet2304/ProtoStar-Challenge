"""Cutoff, source-range and component accounting tests; no network or labels."""
from io import BytesIO
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from common import DataRuleViolation
from enhance_forcing import (COMPONENTS, csv_prefix, cut, dated_table, declaration,
                             load_forcing_for_origin, normalize_ar6)


class ForcingTests(unittest.TestCase):
    def test_range_reader_does_not_request_the_next_year(self):
        content = b'year,a,b\n1750,0,0\n1751,1.123456789,2.987654321\n1752,2.222222222,3.333333333\n1753,999999999,999999999\n'
        calls = []
        def read(offset,count):
            calls.append((offset,count))
            return content[offset:offset+count]
        got,_ = csv_prefix(read,1750,1752,initial_bytes=1)
        boundary = content.index(b'1753,')
        self.assertEqual(got,content[:boundary])
        self.assertTrue(all(a+b <= boundary for a,b in calls))

    def test_range_reader_rejects_ignored_or_short_response(self):
        with self.assertRaises(ValueError):
            csv_prefix(lambda offset,count:b'',1750,1752)

    def test_no_missing_duplicate_or_future_source_years(self):
        valid=b'year,x\n1750,0\n1751,1\n'
        self.assertEqual(len(dated_table(valid,1750,1751)),2)
        for bad in (b'year,x\n1750,0\n1752,2\n',b'year,x\n1750,0\n1750,1\n',b'year,x\n2019,1\n'):
            with self.assertRaises(DataRuleViolation):
                dated_table(bad,1750,1751)

    def test_cut_drops_future_without_using_future_values(self):
        table=pd.DataFrame({'year':range(1750,2019),'x':np.arange(269,dtype=float)})
        first=cut(table,1983)
        table.loc[table.year>1983,'x']=1e30
        pd.testing.assert_frame_equal(first,cut(table,1983))
        with patch.dict(os.environ,{'PROTOSTAR_FOLD_CUTOFF':'1983'}):
            with self.assertRaises(DataRuleViolation):
                cut(table,1990)
            with self.assertRaises(DataRuleViolation):
                load_forcing_for_origin('not-opened',1990)

    def test_asset_metadata_is_origin_scoped(self):
        for origin in (1983,1990,1997,2004,2011,2018):
            item=declaration(origin)
            self.assertEqual(item['kind'],'weather_input')
            self.assertEqual(item['information_end_year'],origin)
            self.assertEqual(item['origins'],[origin])
            self.assertEqual(item['time_column'],'year')
        self.assertIn('design-to-1982',declaration(1982,design=True)['path'])

    def test_component_totals_without_double_counting(self):
        row={name:0. for name in COMPONENTS}
        row.update(year=1750,co2=2.,ch4=.5,n2o=.1,other_wmghg=.2,
                   **{'aerosol-radiation_interactions':-.3,'aerosol-cloud_interactions':-.7},
                   aerosol=-1.,volcanic=-.4,solar=.1,total_anthropogenic=1.8,total_natural=-.3,total=1.5)
        result=normalize_ar6(pd.DataFrame([row]))
        self.assertAlmostEqual(result.wmghg_erf_wm2.iloc[0],2.8)
        self.assertAlmostEqual(result.tropospheric_aerosol_erf_wm2.iloc[0],-1)
        row['total']=999
        with self.assertRaises(ValueError):
            normalize_ar6(pd.DataFrame([row]))


if __name__=='__main__':
    unittest.main()
