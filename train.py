import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tdanet_ablated import TDANetAblated
from models.TDANet_best import TDANetBest
from datamodule import MusdbDataModule


def get_device(preferred: str = 'auto'):
    if preferred == 'cpu':
        return torch.device('cpu')
    if preferred == 'npu':
        try:
            import torch_npu  # vendor package for Ascend NPU
            return torch.device('npu')
        except Exception:
            return torch.device('cpu')
    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    # auto: prefer NPU if available, else CUDA, else CPU
    try:
        import torch_npu
        return torch.device('npu')
    except Exception:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')


#def si_sdr_loss(est_source, ref_source, eps=1e-8):
    # shapes: [B, C, T]
#   den = (ref_source ** 2).sum(dim=-1, keepdim=True)
#    proj = num / (den + eps)
#    res = est_source - proj
#    ratio = (proj.pow(2).sum(dim=-1) + eps) / (res.pow(2).sum(dim=-1) + eps)
#    loss = -10 * torch.log10(ratio + eps)
#    return loss.mean()
def si_sdr_loss(est_source, ref_source, eps=1e-8):
    # est_source, ref_source: [B, C, T]
    
    # 1. Calculate the actual energy of each reference stem
    ref_energy = (ref_source ** 2).sum(dim=-1) # [B, C]
    
    # 2. Mask out silent stems (energy less than a tiny threshold)
    valid_mask = (ref_energy > 1e-4).float() 
    
    # 3. Standard SI-SDR projection
    num = (est_source * ref_source).sum(dim=-1, keepdim=True) * ref_source
    den = ref_energy.unsqueeze(-1)
    proj = num / (den + eps)
    res = est_source - proj
    
    ratio = (proj.pow(2).sum(dim=-1) + eps) / (res.pow(2).sum(dim=-1) + eps)
    loss = -10 * torch.log10(ratio + eps) # [B, C]
    
    # 4. Apply the mask and average only over the active stems
    masked_loss = loss * valid_mask
    valid_count = valid_mask.sum()
    
    if valid_count > 0:
        return masked_loss.sum() / valid_count
    else:
        return masked_loss.sum() * 0.0  # Safe fallback if the whole batch is silent

def train(args):
    device = get_device(args.device)
    if args.model == 'orig':
        model = TDANetBest(num_sources=4, sample_rate=args.sr)
    else:
        model = TDANetAblated(num_sources=4, sample_rate=args.sr)
    model.to(device)

    dm_train = MusdbDataModule(args.musdb_root, subset='train', sr=args.sr, batch_size=args.batch_size, segment_length=args.segment_length)
    dm_val = MusdbDataModule(args.musdb_root, subset='valid', sr=args.sr, batch_size=1, segment_length=None)
    train_loader = dm_train.train_dataloader()
    val_loader = dm_val.val_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best = float('inf')
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for i, (mixture, sources) in enumerate(train_loader):
            mixture = mixture.to(device)
            sources = sources.to(device)
            est = model(mixture)
            loss = si_sdr_loss(est, sources)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()
            n += 1

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            val_chunk_samples = int(args.segment_length * args.sr)
            
            # Start 30 seconds into the track (480,000 samples at 16kHz)
            # This guarantees we skip the silent intros and hit the actual music
            start = 480000 
            end = start + val_chunk_samples
            
            for mixture, sources in val_loader:
                
                mixture = mixture[:, start:end].to(device)
                sources = sources[:, :, start:end].to(device)
                
        #with torch.no_grad():
        #    # Calculate the exact sample length based on args
        #    val_chunk_samples = int(args.segment_length * args.sr)
        #    
        #    for mixture, sources in val_loader:
        #        # FIX 2: Slice the validation tensors to prevent NPU/Transformer memory explosion
        #        mixture = mixture[:, :val_chunk_samples].to(device)
        #        sources = sources[:, :, :val_chunk_samples].to(device)
                
                est = model(mixture)
                val_loss += si_sdr_loss(est, sources).item()
        val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(val_loss)

        print(f"Epoch {epoch} train_loss={running/max(1,n):.4f} val_loss={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best.pth'))
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print('Early stopping')
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--musdb-root', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='./out')
    parser.add_argument('--model', choices=['orig', 'ablated'], default='ablated')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'npu'], default='auto')
    parser.add_argument('--segment-length', type=float, default=3.0, help='training segment length in seconds')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stop', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()


def run_training(**kwargs):
    """Convenience wrapper to call training with keyword args.

    Example:
        run_training(musdb_root='path', out_dir='./out', model='orig', epochs=10)
    """
    from argparse import Namespace

    defaults = dict(
        musdb_root=None,
        out_dir='./out',
        model='ablated',
        device='auto',
        segment_length=3.0,
        sr=16000,
        batch_size=1,
        epochs=100,
        early_stop=30,
    )
    defaults.update(kwargs)
    if defaults['musdb_root'] is None:
        raise ValueError('musdb_root must be provided')

    args = Namespace(**defaults)
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
