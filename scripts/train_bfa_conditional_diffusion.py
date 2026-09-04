"""Train a compact activity-conditional DDPM directly in the BeamSense BFA domain."""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def encode(x):
    phi = (x[..., :2].astype(np.float32) + .5) * (2 * np.pi / 512)
    psi = x[..., 2:].astype(np.float32) / 127 * 2 - 1
    z = np.concatenate((np.cos(phi), np.sin(phi), psi), axis=-1)
    return np.moveaxis(z, -1, 1)  # N,6,10,234


def decode(z):
    z = np.moveaxis(z, 1, -1)
    phi = np.mod(np.arctan2(z[..., 2:4], z[..., :2]), 2 * np.pi)
    phi = np.rint(phi * (512 / (2 * np.pi)) - .5).astype(np.int64) % 512
    psi = np.clip(np.rint((z[..., 4:6] + 1) * 127 / 2), 0, 127)
    return np.concatenate((phi, psi), axis=-1).astype(np.uint16)


def timestep_embedding(t, dim):
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    freq = torch.exp(torch.arange(half, device=t.device) * -scale)
    emb = t.float()[:, None] * freq[None]
    return torch.cat((emb.sin(), emb.cos()), dim=1)


class Block(nn.Module):
    def __init__(self, width, cond_dim):
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(8, width), nn.GroupNorm(8, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.cond = nn.Linear(cond_dim, width)
        self.act = nn.SiLU()

    def forward(self, x, c):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.cond(c)[:, :, None, None]
        return x + self.conv2(self.act(self.norm2(h)))


class ConditionalDenoiser(nn.Module):
    def __init__(self, width=64, cond_dim=128, classes=20):
        super().__init__()
        self.cond_dim = cond_dim
        self.label = nn.Embedding(classes, cond_dim)
        self.time = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.input = nn.Conv2d(6, width, 3, padding=1)
        self.blocks = nn.ModuleList([Block(width, cond_dim) for _ in range(6)])
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 6, 3, padding=1))

    def forward(self, x, t, y):
        c = self.time(timestep_embedding(t, self.cond_dim)) + self.label(y)
        h = self.input(x)
        for block in self.blocks: h = block(h, c)
        return self.output(h)


def schedule(steps, device):
    beta = torch.linspace(1e-4, .02, steps, device=device)
    alpha = 1 - beta
    abar = torch.cumprod(alpha, 0)
    return beta, alpha, abar


@torch.no_grad()
def sample(model, labels, steps, device):
    beta, alpha, abar = schedule(steps, device)
    x = torch.randn(len(labels), 6, 10, 234, device=device)
    for i in reversed(range(steps)):
        t = torch.full((len(labels),), i, dtype=torch.long, device=device)
        eps = model(x, t, labels)
        x = (x - (1-alpha[i]) / torch.sqrt(1-abar[i]) * eps) / torch.sqrt(alpha[i])
        if i: x += torch.sqrt(beta[i]) * torch.randn_like(x)
    return x.clamp(-1.25, 1.25).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path); p.add_argument('output_dir', type=Path)
    p.add_argument('--split-indices-dir', type=Path)
    p.add_argument('--split-seed', type=int, default=111)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--steps', type=int, default=50)
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=2e-4); p.add_argument('--max-samples', type=int)
    p.add_argument('--temporal-weight', type=float, default=0.5)
    p.add_argument('--unit-circle-weight', type=float, default=0.05)
    p.add_argument('--resume', action='store_true', help='resume from output_dir/checkpoint_latest.pt')
    args = p.parse_args()
    checkpoint_path = args.output_dir / 'checkpoint_latest.pt'
    if args.output_dir.exists() and not args.resume:
        p.error('output directory already exists; pass --resume or use a new directory')
    if args.resume and not checkpoint_path.is_file():
        p.error(f'cannot resume: missing {checkpoint_path}')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(args.input, allow_pickle=False) as f:
        x, y = f['x'], f['y'].astype(np.int64)
        meta = {k:f[k] for k in ('source','window_start','participant')}
    if x.shape[1:] != (10,234,4): p.error(f'expected [N,10,234,4], got {x.shape}')
    if args.split_indices_dir:
        train = np.load(args.split_indices_dir/'train_indices.npy').astype(np.int64)
    else:
        train = np.flatnonzero(np.random.default_rng(args.split_seed).random(len(y)) < .70)
    if args.max_samples:
        rng=np.random.default_rng(args.seed); chosen=[]
        per=max(1,args.max_samples//20)
        for label in range(20):
            c=train[y[train]==label]; chosen.extend(rng.choice(c,min(per,len(c)),replace=False))
        train=np.asarray(chosen,dtype=np.int64)
    ds=TensorDataset(torch.from_numpy(encode(x[train])),torch.from_numpy(y[train]))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    model=ConditionalDenoiser().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    beta,alpha,abar=schedule(args.steps,device)
    args.output_dir.mkdir(parents=True, exist_ok=True); history=[]; start_epoch=0
    if args.resume:
        state=torch.load(checkpoint_path,map_location=device,weights_only=False)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        history=list(state['history']); start_epoch=int(state['epoch'])
        if state['steps'] != args.steps or state['train_indices'].shape != train.shape or not np.array_equal(state['train_indices'],train):
            p.error('resume settings/train indices differ from checkpoint')
        print(f'Resuming after epoch {start_epoch}',flush=True)
    for epoch in range(start_epoch,args.epochs):
        model.train(); losses=[]
        for clean,label in loader:
            clean,label=clean.to(device),label.to(device); t=torch.randint(args.steps,(len(clean),),device=device)
            noise=torch.randn_like(clean); a=abar[t][:,None,None,None]
            noisy=torch.sqrt(a)*clean+torch.sqrt(1-a)*noise
            predicted_noise=model(noisy,t,label)
            diffusion_loss=nn.functional.mse_loss(predicted_noise,noise)
            predicted_clean=(noisy-torch.sqrt(1-a)*predicted_noise)/torch.sqrt(a)
            temporal_loss=nn.functional.smooth_l1_loss(
                predicted_clean[:,:,1:]-predicted_clean[:,:,:-1],
                clean[:,:,1:]-clean[:,:,:-1],
            )
            phi_norm=torch.stack((
                predicted_clean[:,0].square()+predicted_clean[:,2].square(),
                predicted_clean[:,1].square()+predicted_clean[:,3].square(),
            ),dim=1)
            unit_circle_loss=nn.functional.mse_loss(phi_norm,torch.ones_like(phi_norm))
            loss=(diffusion_loss+args.temporal_weight*temporal_loss+
                  args.unit_circle_weight*unit_circle_loss)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(loss.item())
        value=float(np.mean(losses)); history.append(value)
        torch.save(dict(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),
                        history=history,steps=args.steps,train_indices=train),checkpoint_path)
        print({'epoch':epoch+1,'loss':value,'checkpoint':str(checkpoint_path)},flush=True)
    torch.save(model.state_dict(),args.output_dir/'model.pt')
    chunk_dir=args.output_dir/'generation_chunks'; chunk_dir.mkdir(exist_ok=True)
    generated=[]
    for start in range(0,len(train),args.batch_size):
        stop=min(start+args.batch_size,len(train))
        chunk_path=chunk_dir/f'{start:08d}_{stop:08d}.npy'
        if chunk_path.is_file():
            chunk=np.load(chunk_path,allow_pickle=False)
            if chunk.shape != (stop-start,10,234,4):
                raise RuntimeError(f'invalid saved generation chunk: {chunk_path}')
        else:
            labels=torch.from_numpy(y[train[start:stop]]).to(device)
            chunk=decode(sample(model,labels,args.steps,device))
            temporary=chunk_path.with_suffix('.tmp.npy')
            np.save(temporary,chunk,allow_pickle=False); temporary.replace(chunk_path)
            print({'generated':stop,'total':len(train),'chunk':str(chunk_path)},flush=True)
        generated.append(chunk)
    generated=np.concatenate(generated)
    np.savez_compressed(args.output_dir/'generated_bfa.npz',x=generated,y=y[train],
        **{k:v[train] for k,v in meta.items()},allocation_real_index=train,
        augmentation_eligible=np.ones(len(train),bool),train_split_seed=np.asarray(args.split_seed),
        augmentation=np.asarray('bfa-conditional-ddpm-circular-v1'))
    (args.output_dir/'protocol.json').write_text(json.dumps(dict(input=str(args.input),device=str(device),seed=args.seed,
        steps=args.steps,epochs=args.epochs,split_seed=args.split_seed,
        temporal_weight=args.temporal_weight,unit_circle_weight=args.unit_circle_weight,
        split_indices_dir=str(args.split_indices_dir) if args.split_indices_dir else None,
        train_samples=len(train),loss=history),indent=2))
    print('Saved',args.output_dir/'generated_bfa.npz',generated.shape)

if __name__=='__main__': main()
