"""Paired frozen-classifier diagnostic: raw vs angle roundtrip vs saved PCA.

Class-specific PCA uses TRUE labels for routing: oracle diagnostic only, NOT
a deployable sensing score. No fitting or classifier retraining is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from train_timevae_bfa import encode, decode, train_indices


def restore_pca(x, preprocessing):
    pca, scale = preprocessing['pca'], preprocessing['scale']
    encoded = encode(x)
    scores = pca.transform(encoded.reshape(-1, 1404))
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid stored PCA scale')
    # Exercise the same standardization/inverse used by generation.
    restored = pca.inverse_transform((scores / scale) * scale)
    return decode(restored.reshape(len(x), 10, 1404))


def choose_baseline(root, real_data, split_seed, model_seed):
    candidates = []
    for path in root.rglob('random_window_metrics.json'):
        try:
            m = json.loads(path.read_text())
            checkpoint = path.parent/'random_window_best.keras'
            if (m.get('augmentation_data') is None and m.get('split') == 'random-window'
                    and m.get('seed') == model_seed and m.get('split_seed') == split_seed
                    and Path(m.get('real_data', '')).resolve() == real_data.resolve()
                    and checkpoint.is_file()):
                candidates.append(path)
        except (ValueError, OSError):
            continue
    if not candidates:
        raise ValueError('No fixed-split baseline found; pass --baseline-dir explicitly')
    # Always remeasure this checkpoint; do not assume its stored metric equals
    # checkpoint performance (EarlyStopping and ModelCheckpoint may differ).
    selected = max(candidates, key=lambda p:p.stat().st_mtime_ns)
    print('Baseline candidates:', [str(p.parent) for p in candidates], flush=True)
    print('Selected latest baseline:', selected.parent, flush=True)
    return selected.parent


def metrics(y, pred):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
    return dict(accuracy=float(accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,labels=np.arange(20),average='macro',zero_division=0)),
                macro_recall=float(recall_score(y,pred,labels=np.arange(20),average='macro',zero_division=0)),
                confusion=confusion_matrix(y,pred,labels=np.arange(20)).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--timevae-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--baseline-dir', type=Path)
    p.add_argument('--results-root', type=Path, default=Path('/home/leehan/results'))
    p.add_argument('--split-seed', type=int, default=111)
    p.add_argument('--model-seed', type=int, default=111)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    if args.batch_size < 1 or args.output_dir.exists():
        p.error('Use positive batch-size and a new output directory')
    baseline = args.baseline_dir or choose_baseline(args.results_root,args.real,args.split_seed,args.model_seed)
    bm = json.loads((baseline/'random_window_metrics.json').read_text())
    if (bm.get('augmentation_data') is not None or bm.get('split') != 'random-window'
            or bm.get('seed') != args.model_seed or bm.get('split_seed') != args.split_seed
            or Path(bm.get('real_data','')).resolve() != args.real.resolve()):
        raise ValueError('Baseline metadata does not match the requested protocol')
    if bm.get('normalization') not in ('none','angle-range'):
        raise ValueError('Unsupported classifier normalization')
    protocol = json.loads((args.timevae_dir/'protocol.json').read_text())
    if protocol['split_seed'] != args.split_seed or protocol.get('smoke_test'):
        raise ValueError('Need full TimeVAE run with matching split seed')
    with np.load(args.real, allow_pickle=False) as f:
        y_all = f['y']
        train = train_indices(len(y_all),args.split_seed)
        test = np.flatnonzero(np.random.default_rng(args.split_seed).random(len(y_all)) >= .85)
        if not np.array_equal(train,np.load(args.timevae_dir/'train_indices.npy',allow_pickle=False)):
            raise ValueError('PCA training indices differ')
        x, y = f['x'][test], y_all[test]
    if len(test) != bm['test_samples'] or x.shape != (len(y),10,234,4):
        raise ValueError('Test count or input shape mismatch')
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
    checkpoint = baseline/'random_window_best.keras'
    model = tf.keras.models.load_model(checkpoint,compile=False)
    predictions = {k:np.empty(len(y),dtype=np.int64) for k in ('raw','angle_roundtrip','pca_oracle')}
    changed = {k:0 for k in ('angle_roundtrip','pca_oracle')}
    variance = {}
    for label in np.unique(y):
        # Load only trusted preprocessors saved by our own TimeVAE run.
        pre = joblib.load(args.timevae_dir/f'class_{label:02d}'/'preprocessing.joblib')
        variance[str(int(label))] = float(pre['pca'].explained_variance_ratio_.sum())
        indexes = np.flatnonzero(y == label)
        for offset in range(0,len(indexes),args.batch_size):
            idx = indexes[offset:offset+args.batch_size]
            raw = x[idx]
            variants = dict(raw=raw,angle_roundtrip=decode(encode(raw)),pca_oracle=restore_pca(raw,pre))
            for name, value in variants.items():
                if name != 'raw':
                    changed[name] += int(np.any(value != raw,axis=(1,2,3)).sum())
                features = value.astype(np.float32)
                if bm['normalization'] == 'angle-range':
                    features /= np.array([511,511,127,127],dtype=np.float32)
                probabilities = np.asarray(model(features,training=False))
                if probabilities.shape != (len(idx),20) or not np.isfinite(probabilities).all():
                    raise ValueError('Invalid classifier output')
                predictions[name][idx] = probabilities.argmax(axis=1)
        print(f'Class {int(label)}: {len(indexes)} test windows; PCA variance={variance[str(int(label))]:.4f}',flush=True)
    result = dict(diagnostic='frozen-baseline-pca-oracle-v1',
                  note='TRUE activity labels route class-specific PCA; not a deployment accuracy.',
                  baseline_dir=str(baseline),checkpoint=str(checkpoint),
                  timevae_dir=str(args.timevae_dir),real=str(args.real),
                  split_seed=args.split_seed,model_seed=args.model_seed,
                  normalization=bm['normalization'],test_samples=len(test),
                  test_indices_sha256=hashlib.sha256(test.astype('<i8').tobytes()).hexdigest(),
                  stored_baseline_accuracy=bm['accuracy'],
                  changed_windows=changed,pca_variance=variance,
                  **{k:metrics(y,v) for k,v in predictions.items()})
    result['pca_minus_raw_pp'] = 100*(result['pca_oracle']['accuracy']-result['raw']['accuracy'])
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output_dir/'predictions.npz',test_indices=test,truth=y,**predictions)
    for name in predictions:
        print(name,json.dumps({k:v for k,v in result[name].items() if k!='confusion'}))
    print('PCA minus raw (pp):',result['pca_minus_raw_pp'])
    print('Saved:',args.output_dir/'result.json')


if __name__ == '__main__':
    main()
