"""Official TimeVAE per activity, with train-only PCA and circular BFA encoding.

Not a conditional reimplementation: imports the pinned author's architecture.
PCA reduces its very large final dense decoder at 234 subcarriers.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

UPSTREAM = '9914576bd44a6facbc94a594a82dd14724969a50'


def encode(x):
    phi = (x[..., :2].astype(np.float32) + .5) * (2*np.pi/512)
    psi = x[..., 2:].astype(np.float32) / 127 * 2 - 1
    return np.concatenate((np.cos(phi), np.sin(phi), psi), axis=-1).reshape(len(x), 10, 1404)


def decode(x):
    x = np.asarray(x).reshape(-1, 10, 234, 6)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite generated features')
    phi = np.mod(np.arctan2(x[..., 2:4], x[..., :2]), 2*np.pi)
    phi = np.rint(phi * (512/(2*np.pi)) - .5).astype(np.int64) % 512
    psi = np.clip(np.rint((x[..., 4:6] + 1)*127/2), 0, 127)
    return np.concatenate((phi, psi), axis=-1).astype(np.uint16)


def train_indices(size, seed):
    return np.flatnonzero(np.random.default_rng(seed).random(size) < .70)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--timevae-repo', type=Path, required=True)
    p.add_argument('--split-seed', type=int, default=111)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--pca-dim', type=int, default=32)
    p.add_argument('--latent-dim', type=int, default=8)
    p.add_argument('--smoke-test', action='store_true', help='one class, two epochs; no augmentation NPZ')
    p.add_argument('--dry-run', action='store_true', help='audit split without TensorFlow')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.pca_dim, args.latent_dim) < 1:
        p.error('numeric training parameters must be positive')
    revision = subprocess.check_output(['git', '-C', str(args.timevae_repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM:
        p.error(f'Expected official TimeVAE commit {UPSTREAM}, got {revision}')
    for filename in ('src/vae/timevae.py', 'src/vae/vae_base.py'):
        if subprocess.check_output(['git', '-C', str(args.timevae_repo), 'diff', 'HEAD', '--', filename]):
            p.error(f'Official model has local changes: {filename}')
    if args.output_dir.exists():
        p.error('Output directory exists; use a new directory to preserve previous runs')
    with np.load(args.input, allow_pickle=False) as f:
        x, y = f['x'], f['y']
        metadata = {k: f[k] for k in ('source', 'window_start', 'participant')}
    if x.shape != (len(y), 10, 234, 4) or y.ndim != 1:
        p.error('Expected x [N,10,234,4], y [N]')
    if not np.array_equal(np.unique(y), np.arange(20)):
        p.error('Expected activity labels 0 through 19')
    if any(v.shape != (len(y),) for v in metadata.values()):
        p.error('Metadata must have one value per window')
    for j, top in enumerate((511, 511, 127, 127)):
        a = x[..., j]
        if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > top) or np.any(a != np.floor(a)):
            p.error(f'Invalid BFA angle {j}')
    train = train_indices(len(y), args.split_seed)
    counts = np.bincount(y[train].astype(int), minlength=20)
    if np.any(counts < 2):
        p.error('Every activity needs at least two training windows')
    args.output_dir.mkdir(parents=True)
    report = dict(protocol='official-timevae-classwise-pca-bfa-v1', upstream_commit=revision,
                  input=str(args.input.resolve()), seed=args.seed, split_seed=args.split_seed,
                  train_count=len(train), class_train_counts=counts.tolist(),
                  pca_dim=args.pca_dim, latent_dim=args.latent_dim,
                  epochs=2 if args.smoke_test else args.epochs, batch_size=args.batch_size,
                  model=dict(hidden_layer_sizes=[32,64], trend_poly=0, custom_seas=None,
                             use_residual_conn=True, reconstruction_wt=3.0),
                  smoke_test=args.smoke_test, classes=[],
                  train_indices_sha256=hashlib.sha256(train.astype('<i8').tobytes()).hexdigest())
    def save_report():
        (args.output_dir/'protocol.json').write_text(json.dumps(report, indent=2))
    np.save(args.output_dir/'train_indices.npy', train)
    save_report()
    print('Training windows:', len(train), 'class counts:', counts.tolist(), flush=True)
    print('WARNING: rare classes with <100 train windows:', np.flatnonzero(counts < 100).tolist(), flush=True)
    if args.dry_run:
        return
    import tensorflow as tf
    import joblib
    from sklearn.decomposition import PCA
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    sys.path.insert(0, str(args.timevae_repo.resolve()/'src'))
    from vae.timevae import TimeVAE
    generated = np.empty((len(train), 10, 234, 4), dtype=np.uint16)
    for label in (range(1) if args.smoke_test else range(20)):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + label)
        slots = np.flatnonzero(y[train] == label)
        real = encode(x[train[slots]])
        dim = min(args.pca_dim, real.shape[0]*10, 1404)
        pca = PCA(n_components=dim, svd_solver='randomized', random_state=args.seed+label)
        scores = pca.fit_transform(real.reshape(-1,1404))
        scale = np.maximum(scores.std(axis=0), 1e-6).astype(np.float32)
        inputs = (scores/scale).reshape(-1,10,dim).astype(np.float32)
        folder = args.output_dir/f'class_{label:02d}'
        folder.mkdir()
        joblib.dump(dict(pca=pca, scale=scale), folder/'preprocessing.joblib')
        model = TimeVAE(seq_len=10, feat_dim=dim, latent_dim=args.latent_dim,
                        hidden_layer_sizes=[32,64], reconstruction_wt=3.,
                        batch_size=args.batch_size, trend_poly=0,
                        custom_seas=None, use_residual_conn=True)
        print(f'Class {label}: training={len(slots)} PCA variance={pca.explained_variance_ratio_.sum():.4f}', flush=True)
        history = model.fit(inputs, epochs=report['epochs'], batch_size=args.batch_size,
                            shuffle=True, verbose=2, callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        if not all(np.isfinite(v).all() for v in history.history.values()):
            raise RuntimeError(f'Nonfinite loss in class {label}')
        model.save(str(folder))
        (folder/'history.json').write_text(json.dumps(history.history))
        rng = np.random.default_rng(args.seed+10000+label)
        for start in range(0,len(slots),args.batch_size):
            batch_slots = slots[start:start+args.batch_size]
            z = rng.standard_normal((len(batch_slots),args.latent_dim)).astype(np.float32)
            decoded = model.decoder(z, training=False)
            if isinstance(decoded, (list, tuple)):
                decoded = decoded[0]
            sample = np.asarray(decoded)
            reconstructed = pca.inverse_transform(sample.reshape(-1,dim)*scale)
            generated[batch_slots] = decode(reconstructed.reshape(-1,10,1404))
        report['classes'].append(dict(label=label, samples=len(slots), pca_dim=dim,
                                     pca_variance=float(pca.explained_variance_ratio_.sum())))
        save_report()
        del model, inputs, real, scores, pca
        gc.collect()
    if args.smoke_test:
        print('Smoke test passed (one class, two epochs); no evaluation NPZ created.', flush=True)
        return
    # Source keys are allocation slots for trainer matching, not reconstruction
    # provenance: every row was sampled independently from a class model prior.
    target = args.output_dir/'generated_bfa.npz'
    np.savez_compressed(target, x=generated, y=y[train],
        **{k:v[train] for k,v in metadata.items()}, allocation_real_index=train,
        augmentation_eligible=np.ones(len(train), dtype=bool),
        train_split_seed=np.asarray(args.split_seed),
        augmentation=np.asarray('official-timevae-classwise-pca-prior-v1'))
    print('Saved:', target, 'x:', generated.shape, flush=True)


if __name__ == '__main__':
    main()
