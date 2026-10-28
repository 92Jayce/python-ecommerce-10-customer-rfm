"""第10讲 RFM与可解释客户分层
业务问题：如何同时看最近购买、购买频次和累计价值？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

# %% 同分保留同分，不用强行rank(method=first)制造不同客户等级
rfm=rfm_table()
for feature,label,ascending in [('recency_days','R',False),('frequency','F',True),('monetary_cents','M',True)]:
    pct=rfm[feature].rank(method='average',pct=True,ascending=ascending)
    rfm[label]=np.ceil(pct*5).clip(1,5).astype(int)
rfm['segment']=np.select([
    rfm.R.ge(4)&rfm.F.ge(4)&rfm.M.ge(4),
    rfm.R.le(2)&rfm.F.ge(4),rfm.R.ge(4)&rfm.F.le(2)],
    ['高活跃高价值','历史高频需关注','近期低频'],default='其他已购客户')
show(rfm);save_table(rfm,10,'rfm_customers')
profile=rfm.groupby('segment',as_index=False).agg(customers=('customer_id','size'),
    median_recency=('recency_days','median'),median_frequency=('frequency','median'),median_value_cents=('monetary_cents','median'))
show(profile);save_table(profile,10,'rfm_segment_profile')
fig,ax=plt.subplots(figsize=(7,4));ax.scatter(rfm.recency_days,rfm.monetary_cents/100,s=8,alpha=.5,color=BLUE)
ax.set(title='RFM: recency and observed net value',xlabel='Days since last positive-net order',ylabel='Observed net merchandise CNY')
save_figure(fig,10,'rfm_scatter')
print('RFM仅含至少一笔正净额订单的客户；未购买客户不能用R=0伪装为最近购买。')
