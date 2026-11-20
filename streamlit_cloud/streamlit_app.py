"""Public lane-detection demo on Streamlit Community Cloud (free CPU).

Pick models, upload a dashcam image or short video (or use a built-in sample), click Generate. Logic lives in
demo_core.py; inference is the benchmark's own verified code running on CPU, in persistent per-model workers.
"""
import threading
from pathlib import Path
import streamlit as st
import demo_core as core

st.set_page_config(page_title='Lane Detection Demo', page_icon='🛣️', layout='wide')
st.session_state.setdefault('runs', [])
st.session_state.setdefault('sample', None)


@st.cache_resource(show_spinner='First start: downloading the model code and checkpoints (a few minutes, once)…')
def ready_models():
    return core.setup()


@st.cache_resource
def job_lock():   # one job at a time across visitors (shared free CPU)
    return threading.Lock()


st.title('Lane detection demo')
st.write('Pick models, upload a dashcam **image** or a short **video** (or try a sample), and click **Generate**. '
         'Models trained on [CULane](https://xingangpan.github.io/projects/CULane.html). '
         '[Code, benchmark and full report on GitHub](https://github.com/AhireRohit/lane_detection_CuLane).')
available = ready_models()
if not available:
    st.error('Model setup failed. Please try again later.')
    with st.expander('Setup log'): st.code(core.setup_log())
    st.stop()
missing = [core.MODELS[k][0] for k in core.MODELS if k not in available]
if missing: st.warning('Not available right now (checkpoint download failed): ' + ', '.join(missing))

left, right = st.columns([3, 2])
with left:
    chosen = st.multiselect('Models', available, default=available[:1], format_func=lambda k: core.MODELS[k][0],
                            help='Pick several to compare them side by side.')
    for k in chosen: st.caption(f'**{core.MODELS[k][0]}**: {core.MODELS[k][1]}')
    up = st.file_uploader('Dashcam image or video', type=sorted(e.lstrip('.') for e in core.IMAGE_EXT | core.VIDEO_EXT))
    if up: st.session_state.sample = None
    s1, s2 = st.columns(2)
    for col, (name, label) in zip((s1, s2), core.SAMPLES.items()):
        if col.button(label, use_container_width=True, disabled=not core.SAMPLE_SRC.is_file()):
            st.session_state.sample = name
    if st.session_state.sample and not up:
        st.info(f'Using **{core.SAMPLES[st.session_state.sample]}**: the example clip shipped with the '
                '[Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) repository.')
with right:
    st.subheader('Options')
    adapter = st.radio('Frame adapter', ['auto', 'squash'], horizontal=True,
                       format_func=lambda a: {'auto': 'Crop to CULane shape (recommended)', 'squash': 'Squash whole frame'}[a],
                       help='The models expect 1640×590 road views. "Crop" takes a full-width band of that shape at the '
                            "road's vanishing point, so the road is not distorted.")
    tracker = st.checkbox('Temporal lane tracker (videos)', value=True,
                          help='Matches lanes across frames and smooths them; held dashed lanes are drawn dashed.')
    loaded = [core.MODELS[m][0] for m in core.loaded_models()]
    st.caption('Loaded and ready: ' + (', '.join(loaded) if loaded else 'none yet (the first run of a model loads it, '
                                       'about 10 to 30 s)'))
st.caption(f'Free shared CPU: about 0.25 s per frame for CLRerNet, ~1 s for SCNN, once a model is loaded. Videos are cut '
           f'to {core.MAX_S} s at {core.FPS_CAP} fps. Uploads are processed only for your request and deleted within an '
           'hour. Your media has no labels, so the numbers below are detections and stability, not accuracy.')

src = (up.getvalue(), up.name) if up else ((core.sample(st.session_state.sample), st.session_state.sample)
                                           if st.session_state.sample else None)
if st.button('Generate', type='primary', disabled=not (src and chosen)):
    bar = st.progress(0.0, text='Starting…')
    lock = job_lock()
    if lock.locked(): bar.progress(0.0, text="Another visitor's job is running; waiting for it to finish…")
    try:
        with lock:
            kind, res, comp = core.run_job(src[0], src[1], chosen, adapter, tracker,
                                           progress=lambda f, t: bar.progress(f, text=t))
        bar.progress(1.0, text='Done')
        st.session_state.runs.insert(0, dict(name=src[1], kind=kind, results=res, compare=comp, adapter=adapter,
                                             tracker=tracker))
    except Exception as e:   # show the reason instead of a blank failure
        st.error(str(e))

for ri, run in enumerate(st.session_state.runs):
    st.divider(); res = run['results']
    st.subheader(run['name'])
    st.caption(f"adapter: {run['adapter']} · tracker: {'on' if run['tracker'] and run['kind'] == 'video' else 'off'}")
    if len(res) > 1:
        st.dataframe([{'Model': core.MODELS[m][0], 'Frames with lanes': f"{r['summary']['frames_with_lanes']:.0%}",
                       'Mean lanes': round(r['summary']['mean_lanes'], 2), 'Lane-count changes': r['summary']['lane_count_changes'],
                       'Median ms/frame (CPU)': round(r['summary']['median_latency_ms'])} for m, r in res.items()],
                     hide_index=True, use_container_width=True)
    names = (['Side by side'] if run.get('compare') else []) + [core.MODELS[m][0] for m in res]
    tabs = st.tabs(names)
    if run.get('compare'):
        with tabs[0]:
            c = Path(run['compare'])
            if run['kind'] == 'image': st.image(str(c), use_container_width=True)
            else: st.video(str(c))
            st.download_button('Download comparison', c.read_bytes(), file_name=f'lanes_compare{c.suffix}', key=f'c{ri}')
        tabs = tabs[1:]
    for tab, (m, r) in zip(tabs, res.items()):
        with tab:
            sm = r['summary']
            if 'image' in r:
                st.image(str(r['image']), use_container_width=True)
                st.download_button('Download image', Path(r['image']).read_bytes(), file_name=f'{m}_lanes.png', key=f'i{ri}{m}')
            else:
                st.video(str(r['video']))
                st.download_button('Download video', Path(r['video']).read_bytes(), file_name=f'{m}_lanes.mp4', key=f'v{ri}{m}')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Frames', sm['frames']); c2.metric('Frames with lanes', f"{sm['frames_with_lanes']:.0%}")
            c3.metric('Mean lanes', f"{sm['mean_lanes']:.2f}"); c4.metric('Median ms/frame', round(sm['median_latency_ms']))
            a = sm['adapter']
            st.caption(f"Adapter: {a['mode']}" + (f", band rows {a['band'][0]}–{a['band'][1]}" if a['mode'] == 'crop' else ''))
