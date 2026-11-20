# Fine-tune CONTROL: authors' CLRerNet DLA34-EMA recipe unchanged, short schedule from the released EMA checkpoint.
# Fixed in advance: 3 epochs, AdamW lr 6e-5 cosine to 6e-6, batch 24, seed 0; the LAST epoch is the result (no selection).
_base_ = ['/path/to/Lane_Detection/benchmark_culane/vendors/clrernet/configs/clrernet/culane/clrernet_culane_dla34_ema.py']
load_from = '/path/to/Lane_Detection/benchmark_culane/checkpoints/clrernet/clrernet_culane_dla34_ema.pth'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)
train_dataloader = dict(batch_size=24, dataset=dict(diff_file='/path/to/Lane_Detection/benchmark_culane/data_aux/train_diffs.npz'))
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=6e-5))
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=3, T_max=3, eta_min=6e-6)]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
