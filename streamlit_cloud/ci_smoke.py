"""CI smoke test of the public demo on a clean machine with the platform's Python (run from streamlit_cloud/).

Real first-start setup (model env via uv, pinned model code, sha256-checked checkpoints; LANE_MODELS limits the
download), then the built-in sample image through two CLRerNet checkpoints with persistent workers, a repeat run
(must reuse the loaded model), and a render of the page.
"""
import time
import demo_core as core

t = time.time(); avail = core.setup(); print(f'setup {time.time() - t:.0f} s, available: {avail}')
assert {'clrernet', 'clrernet_robust'} <= set(avail), core.setup_log()
img = core.sample('sample_frame.jpg')
t = time.time(); kind, res, comp = core.run_job(img, 'sample_frame.jpg', ['clrernet', 'clrernet_robust'])
first = time.time() - t
for m, r in res.items():
    print(m, 'lanes', r['summary']['mean_lanes'], 'ms', round(r['summary']['median_latency_ms']))
    assert kind == 'image' and r['summary']['mean_lanes'] >= 2, (m, r['summary'])
assert comp is not None and comp.is_file()
t = time.time(); core.run_job(img, 'sample_frame.jpg', ['clrernet']); again = time.time() - t
print(f'first run {first:.1f} s (loads 2 models), repeat {again:.1f} s (model already loaded)')
assert again < first / 2, (first, again)
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('streamlit_app.py').run(timeout=300)
assert not at.exception, [e.value for e in at.exception]
print('CI SMOKE OK')
