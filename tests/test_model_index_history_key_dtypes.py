"""Regression for string CatBoost categories vs numeric publication keys.

No database, trained model, or expensive fit required: exercise the exact
ModelIndexHistory builder and its follow-up reliability GROUP join.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from motorcycle_surface.core import MotorcyclePriceSurface, GROUP


def _fixtures():
    asof=pd.Timestamp('2026-11-22')
    ads, cells, surface = [], [], []
    for copy in ('0', '1'):
        for i in range(8):
            ads.append(dict(motorcycle_model_id=353,real_brand='Bajaj',
                real_model='Pulsar 180',is_copy=copy,
                model_variant='DEFAULT',date=pd.Timestamp('2026-11-15')+pd.Timedelta(days=i%3),
                price=100.+i))
        for i in range(3):
            cells.append(dict(motorcycle_model_id='353',real_brand='Bajaj',
                real_model='Pulsar 180',is_copy=copy,
                model_variant='DEFAULT',publish_state='MODEL_SUPPORT_REVIEW',
                price_monotonic=100., actual=101.,price_low=95.,price_high=105.))
        surface.append(dict(motorcycle_model_id='353',real_brand='Bajaj',
            real_model='Pulsar 180',is_copy=int(copy),model_variant='DEFAULT',
            shift_down_pct=2.,is_reliable=(copy=='0')))
    ads=pd.DataFrame(ads)
    cells=pd.DataFrame(cells)
    surface=pd.DataFrame(surface)
    builder=object.__new__(MotorcyclePriceSurface)
    builder.asof=asof
    builder.config=SimpleNamespace(history_days=120)
    builder.ads=ads
    builder.backtest_cells=cells
    return builder,surface


def test_model_index_history_merges_mixed_string_and_integer_is_copy():
    builder,surface=_fixtures()
    idx=builder._build_model_index_history(surface)
    assert len(idx)==2
    assert set(idx['is_copy'])=={0,1}
    assert idx.motorcycle_model_id.eq('353').all()
    assert idx.samples_last_month.eq(8).all()
    assert idx.tested_samples.eq(3).all()
    assert np.isfinite(idx['mape']).all()
    assert idx.is_reliable.all()
    # A downstream join in create_surface must not reintroduce str/int mismatch.
    group_valid=(surface.groupby(GROUP,as_index=False,dropna=False).is_reliable.any()
                 .rename(columns={'is_reliable':'_any_cell_reliable'}))
    group_valid=builder._normalize_history_group_keys(group_valid)
    joined=idx.merge(group_valid,on=GROUP,how='left',validate='one_to_one')
    assert len(joined)==2
    assert joined['_any_cell_reliable'].notna().all()
    assert (joined.set_index('is_copy')['_any_cell_reliable'].to_dict()
            == {0: True,1: False})


def test_model_history_normalization_does_not_change_training_ads_or_surface():
    builder,surface=_fixtures()
    original_ads=builder.ads.copy(deep=True)
    original_oof=builder.backtest_cells.copy(deep=True)
    original_surface=surface.copy(deep=True)
    builder._build_model_index_history(surface)
    pd.testing.assert_frame_equal(builder.ads, original_ads)
    pd.testing.assert_frame_equal(builder.backtest_cells, original_oof)
    pd.testing.assert_frame_equal(surface, original_surface)


def test_nonbinary_is_copy_fails_closed_without_colliding_groups():
    _, surface=_fixtures()
    surface.loc[0,'is_copy']=3
    with pytest.raises(ValueError,match='non-binary'):
        MotorcyclePriceSurface._normalize_history_group_keys(surface)
