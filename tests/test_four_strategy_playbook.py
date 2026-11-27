import pandas as pd

from four_strategy_playbook import opening


def test_orb_uses_closed_range_and_close_confirmation():
    bars=pd.DataFrame(dict(high=[11.,11.2,12.],low=[10.,10.,10.5],
                           close=[10.5,10.9,11.5],hour_atr=[1.,1.,1.]))
    assert opening(bars,1)[0][:3]==(2,1,10.5)
    bars["hour_atr"]=3.
    assert opening(bars,1)==[]
