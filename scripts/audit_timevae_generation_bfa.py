"""Separate PCA, posterior-mean reconstruction, and saved prior generation.

Uses oracle class routing for reconstruction. Generated-label agreement is
not downstream sensing accuracy. Never refits PCA, VAE, or classifier.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
from train_timevae_bfa import UPSTREAM, encode, decode, train_indices
from audit_timevae_pca_bfa import metrics, restore_pca


def reconstruct(x, preprocessing, vae):
    pca, scale = preprocessing['pca'], preprocessing['scale']
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid PCA scale')
    scores = pca.transform(encode(x).reshape(-1,1404))
    inputs = (scores/scale).reshape(len(x),10,-1).astype(np.float32)
    # Posterior mean avoids adding sampling noise to this reconstruction test.
    mean, _, _ = vae.encoder(inputs,training=False)
    values = vae.decoder(mean,training=False)
    if isinstance(values,(list,tuple)):
        values = values[0]
    values = np.asarray(values)
    if values.shape != inputs.shape or not np.isfinite(values).all():
        raise ValueError('Invalid VAE reconstruction')
    restored = pca.inverse_transform(values.reshape(-1,inputs.shape[-1])*scale)
    return decode(restored.reshape(len(x),10,1404))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-audit',type=Path,required=True)
    p.add_argument('--timevae-repo',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--augment-seed',type=int,default=111)
    p.add_argument('--augment-ratio',type=float,default=.1)
    args = p.parse_args()
    if args.output_dir.exists() or args.batch_size < 1 or not 0 < args.augment_ratio <= 1:
        p.error('Use a new output directory, positive batch size, and ratio in (0,1]')
    previous = json.loads(args.previous_audit.read_text())
    folder = Path(previous['timevae_dir'])
    protocol = json.loads((folder/'protocol.json').read_text())
    revision = subprocess.check_output(['git','-C',str(args.timevae_repo),'rev-parse','HEAD'],text=True).strip()
    if revision != UPSTREAM or protocol['upstream_commit'] != UPSTREAM:
        raise ValueError('Official TimeVAE revision mismatch')
    for filename in ('src/vae/timevae.py','src/vae/vae_base.py'):
        if subprocess.check_output(['git','-C',str(args.timevae_repo),'diff','HEAD','--',filename]):
            raise ValueError('Official model code was modified')
    if protocol['split_seed'] != previous['split_seed'] or protocol.get('smoke_test'):
        raise ValueError('TimeVAE protocol mismatch')
    with np.load(previous['real'],allow_pickle=False) as f:
        y_all = f['y']
        train = train_indices(len(y_all),previous['split_seed'])
        test = np.flatnonzero(np.random.default_rng(previous['split_seed']).random(len(y_all)) >= .85)
        if not np.array_equal(train,np.load(folder/'train_indices.npy',allow_pickle=False)):
            raise ValueError('Training split mismatch')
        if hashlib.sha256(test.astype('<i8').tobytes()).hexdigest() != previous['test_indices_sha256']:
            raise ValueError('Previous test indices mismatch')
        x, y = f['x'][test], y_all[test]
        train_source, train_start = f['source'][train], f['window_start'][train]
    generated_path = folder/'generated_bfa.npz'
    with np.load(generated_path,allow_pickle=False) as f:
        generated, labels = f['x'], f['y']
        if (not np.array_equal(f['allocation_real_index'],train)
                or not np.array_equal(labels,y_all[train])
                or not np.array_equal(f['source'],train_source)
                or not np.array_equal(f['window_start'],train_start)):
            raise ValueError('Generated rows do not match the original training allocation')
    if generated.shape != (len(train),10,234,4) or x.shape != (len(test),10,234,4):
        raise ValueError('Invalid BFA shapes')
    for values in (x,generated):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > np.array([511,511,127,127])):
            raise ValueError('Invalid BFA ranges')
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
    sys.path.insert(0,str(args.timevae_repo.resolve()/'src'))
    from vae.timevae import TimeVAE
    checkpoint = Path(previous['checkpoint'])
    classifier = tf.keras.models.load_model(checkpoint,compile=False)
    def predict(value):
        value = value.astype(np.float32)
        if previous['normalization'] == 'angle-range':
            value /= np.array([511,511,127,127],dtype=np.float32)
        elif previous['normalization'] != 'none':
            raise ValueError('Unsupported normalization')
        probabilities = np.asarray(classifier(value,training=False))
        if probabilities.shape != (len(value),20) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid classifier output')
        return probabilities.argmax(axis=1)
    predictions = {name:np.empty(len(test),dtype=np.int64) for name in ('raw','pca_oracle','vae_reconstruction_oracle')}
    for label in np.unique(y):
        pre = joblib.load(folder/f'class_{label:02d}'/'preprocessing.joblib')
        vae = TimeVAE.load(str(folder/f'class_{label:02d}'))
        indices = np.flatnonzero(y == label)
        for offset in range(0,len(indices),args.batch_size):
            idx = indices[offset:offset+args.batch_size]
            predictions['raw'][idx] = predict(x[idx])
            predictions['pca_oracle'][idx] = predict(restore_pca(x[idx],pre))
            predictions['vae_reconstruction_oracle'][idx] = predict(reconstruct(x[idx],pre,vae))
        print(f'Reconstructed class {int(label)}: {len(indices)} test windows',flush=True)
        del vae,pre
        gc.collect()
    generated_pred = np.empty(len(labels),dtype=np.int64)
    for offset in range(0,len(labels),args.batch_size):
        generated_pred[offset:offset+args.batch_size] = predict(generated[offset:offset+args.batch_size])
    count = int(np.floor(len(labels)*args.augment_ratio))
    if not count:
        raise ValueError('Empty generated subset')
    subset = (np.random.default_rng(args.augment_seed).choice(np.arange(len(labels)),count,replace=False)
              if args.augment_ratio < 1 else np.arange(len(labels)))
    result = dict(diagnostic='timevae-posterior-mean-vs-prior-v1',previous_audit=str(args.previous_audit),
                  checkpoint=str(checkpoint),timevae_dir=str(folder),generated_path=str(generated_path),
                  note='Reconstruction uses true-label class routing. Prior scores measure assigned-label agreement, not downstream sensing accuracy.',
                  test_samples=len(test),generated_samples=len(labels),subset_samples=len(subset),
                  augmentation_seed=args.augment_seed,augmentation_ratio=args.augment_ratio,
                  **{name:metrics(y,value) for name,value in predictions.items()},
                  prior_generated_all=metrics(labels,generated_pred),
                  prior_generated_selected=metrics(labels[subset],generated_pred[subset]),
                  generated_prediction_counts=np.bincount(generated_pred,minlength=20).tolist(),
                  generated_label_counts=np.bincount(labels,minlength=20).tolist())
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output_dir/'predictions.npz',test_indices=test,truth=y,**predictions,
                        generated_truth=labels,generated_prediction=generated_pred,selected_generated_indices=subset)
    for name in (*predictions,'prior_generated_all','prior_generated_selected'):
        print(name,json.dumps({k:v for k,v in result[name].items() if k!='confusion'}),flush=True)
    print('Saved:',args.output_dir/'result.json')


if __name__ == '__main__':
    main()
