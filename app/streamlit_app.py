"""Lane-detection demo: pick a model, upload a dashcam image or video, click Generate.

Inference runs on a cluster GPU through HTCondor (see app/cluster.py), with the same model code that produced the
benchmark numbers. Start with:  streamlit run app/streamlit_app.py
"""
import json, time
from pathlib import Path
import streamlit as st
import cluster as C

MODELS = {
    'clrernet': dict(label='CLRerNet (DLA34)', blurb='Most accurate and fastest: test F1 81.55, 20.7 ms/frame on a P100.'),
    'clrernet_robust': dict(label='CLRerNet, robust fine-tune (ours)',
                            blurb='Fine-tuned with rain/blur/JPEG/dark augmentation: test F1 81.18, mean F1 on corrupted images 77.3 (vs 72.6).'),
    'ufldv2': dict(label='UFLDv2 (ResNet34)', blurb='Ultra-Fast-Lane-Detection v2: test F1 75.86, 53.3 ms/frame end to end.'),
    'scnn': dict(label='SCNN (VGG16)', blurb='Classic spatial-CNN baseline: test F1 70.07, 63.9 ms/frame.'),
}

st.set_page_config(page_title='Lane Detection Demo', page_icon='🛣️', layout='wide')
st.session_state.setdefault('runs', [])

# ---------------- sidebar: cluster connection ----------------
with st.sidebar:
    st.header('Cluster')
    s = C.load_settings()
    host = st.text_input('Login (user@host)', value=s['host'], help='Key-based SSH login to the HTCondor submit node.')
    root = st.text_input('Remote benchmark_culane folder', value=s['root'])
    if st.button('Test connection', use_container_width=True):
        try:
            st.success(C.check_connection(host, root))
        except C.ClusterError as e:
            st.error(str(e))
    st.caption('Each Generate submits one GPU job. Waiting for a free GPU usually takes under a minute, '
               'and loading each model takes about 10–30 s.')

# ---------------- main: inputs ----------------
st.title('Lane detection demo')
st.write('Choose one or more models, upload a dashcam **image or video**, and click **Generate**.')

left, right = st.columns([3, 2])
with left:
    chosen = st.multiselect('Models', list(MODELS), default=['clrernet'], format_func=lambda k: MODELS[k]['label'])
    for k in chosen:
        st.caption(f"**{MODELS[k]['label']}**: {MODELS[k]['blurb']}")
    up = st.file_uploader('Dashcam image or video', type=sorted(e.lstrip('.') for e in C.IMAGE_EXT | C.VIDEO_EXT))
with right:
    st.subheader('Options')
    adapter = st.radio('Frame adapter', ['auto', 'squash'], horizontal=True,
                       format_func=lambda a: {'auto': 'Crop to CULane shape (recommended)', 'squash': 'Squash whole frame'}[a],
                       help='The models were trained on 1640×590 road views. "Crop" takes a full-width band of that shape, '
                            'placed at the road\'s vanishing point, so the road is not distorted. "Squash" resizes the whole frame.')
    tracker = st.checkbox('Temporal lane tracker (videos)', value=True,
                          help='Matches lanes across frames and smooths them. It reduces flicker and briefly holds dashed '
                               'lanes, drawn dashed while held. Images are always processed frame by frame.')
    max_s = st.slider('Video length to process (seconds)', 5, 120, 30, 5,
                      help='Longer videos are cut to this length before uploading. Audio is removed.')

go = st.button('Generate', type='primary', disabled=not (up and chosen and host and root))
if not up:
    st.info('Upload a file to enable Generate.')

# ---------------- run ----------------
if go:
    box = st.status('Preparing upload…', expanded=True)
    try:
        work = C.RUNS / time.strftime('%Y%m%dT%H%M%S')
        inp, kind = C.prepare_input(up.getvalue(), up.name, max_s, work)
        box.write(f'Prepared {kind}: {inp.stat().st_size / 1e6:.1f} MB')
        box.update(label='Uploading to the cluster and submitting the GPU job…')
        job = C.submit(host, root, inp, kind, chosen, adapter, tracker)
        box.write(f"HTCondor job {job['cluster_id']} submitted (run `{job['job_id']}`)")
        bar = st.progress(0.0, text='Waiting for a GPU…')
        t0, misses = time.time(), 0
        while True:
            time.sleep(4)
            try:
                p = C.poll(host, job); misses = 0
            except C.ClusterError:   # brief network or VPN drop: the job keeps running on the cluster
                misses += 1
                if misses >= 8: raise
                bar.progress(0.0, text=f'Connection to the cluster lost, retrying ({misses}/8)…'); continue
            done_models = sum(1 for m in chosen if p['status'].get(m, {}).get('stage') == 'done')
            cur = next((p['status'][m] for m in chosen if m in p['status'] and p['status'][m].get('stage') != 'done'), None)
            state = 'finished' if p['done'] else p['condor']
            frac, text = done_models / len(chosen), f"{state} · {done_models}/{len(chosen)} models done"
            if cur:
                tot = cur.get('total') or 0
                within = (cur.get('frame', 0) / tot) if tot else 0.0
                frac += within / len(chosen)
                text += f" · {MODELS[cur['model']]['label']}: {cur['stage']}" + (f" {cur.get('frame', 0)}/{tot}" if tot > 1 else '')
            bar.progress(min(frac, 1.0), text=f'{text} · {int(time.time() - t0)} s')
            if p['done'] or p['failed'] or p['condor'] in ('held', 'removed') or (p['condor'] == 'left the queue' and time.time() - t0 > 30 and not cur):
                break
        if p['failed'] or not p['done']:
            box.update(label='The job did not finish cleanly', state='error')
            st.code(C.failure_logs(host, job))
        box.update(label='Downloading results…')
        res = C.fetch(host, job, work)
        if res:
            st.session_state.runs.insert(0, dict(job=job, name=up.name, kind=kind, results=res, adapter=adapter, tracker=tracker))
            box.update(label=f'Done in {int(time.time() - t0)} s', state='complete', expanded=False)
        else:
            box.update(label='No results came back', state='error')
    except C.ClusterError as e:
        box.update(label='Error', state='error')
        st.error(str(e))

# ---------------- results ----------------
for ri, run in enumerate(st.session_state.runs):
    st.divider()
    job, res = run['job'], run['results']
    st.subheader(f"{run['name']}")
    st.caption(f"Run `{job['job_id']}` · adapter: {run['adapter']} · tracker: {'on' if run['tracker'] and run['kind'] == 'video' else 'off'} · "
               'No labels exist for uploaded media, so the numbers below describe detections and stability, not accuracy.')
    if len(res) > 1:
        rows = [dict(Model=MODELS[m]['label'], **{
            'Frames with lanes': f"{r['summary']['frames_with_lanes']:.0%}", 'Mean lanes': round(r['summary']['mean_lanes'], 2),
            'Lane-count changes': r['summary']['lane_count_changes'], 'Median ms/frame (P100)': round(r['summary']['median_latency_ms'], 1)})
            for m, r in res.items()]
        st.dataframe(rows, hide_index=True, use_container_width=True)
    tabs = st.tabs([MODELS[m]['label'] for m in res])
    for tab, (m, r) in zip(tabs, res.items()):
        with tab:
            sm = r['summary']
            if 'image' in r:
                st.image(str(r['image']), use_container_width=True)
                st.download_button('Download image', r['image'].read_bytes(), file_name=f'{m}_lanes.png', key=f'i{ri}{m}')
            else:
                st.video(str(r['video']))
                st.download_button('Download video', r['video'].read_bytes(), file_name=f'{m}_lanes.mp4', key=f'v{ri}{m}')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Frames', sm['frames']); c2.metric('Frames with lanes', f"{sm['frames_with_lanes']:.0%}")
            c3.metric('Mean lanes', f"{sm['mean_lanes']:.2f}"); c4.metric('Median ms/frame', f"{sm['median_latency_ms']:.1f}")
            a = sm['adapter']
            st.caption(f"Adapter: {a['mode']}" + (f", band rows {a['band'][0]}–{a['band'][1]} ({a['method']})" if a['mode'] == 'crop' else f" ({a.get('note', '')})"))
            st.download_button('Lane coordinates (JSON lines)', Path(r['lanes']).read_bytes(), file_name=f'{m}_lanes.jsonl', key=f'j{ri}{m}')
    if st.button('Delete this upload from the cluster', key=f'del{ri}'):
        try:
            C.delete_remote(host, job); st.success('Deleted from the cluster.')
        except (C.ClusterError, AssertionError) as e:
            st.error(str(e))
