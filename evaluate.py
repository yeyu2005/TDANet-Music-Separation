import argparse
import os
import torch
import numpy as np
import csv
from datamodule import MusdbDataModule
from tdanet_ablated import TDANetAblated
from models.TDANet_best import TDANetBest


def get_device(preferred: str = 'auto'):
    if preferred == 'cpu':
        return torch.device('cpu')
    if preferred == 'npu':
        try:
            import torch_npu
            return torch.device('npu')
        except Exception:
            return torch.device('cpu')
    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import torch_npu
        return torch.device('npu')
    except Exception:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')


def sliding_inference(model, mixture, device, chunk_size, overlap_samples=0):
    """Sliding-window inference with optional overlap-add reconstruction.

    model: torch Module
    mixture: tensor [B, T]
    chunk_size: int (samples)
    overlap_samples: int (samples) amount of overlap between consecutive chunks
    Returns: estimated tensor [B, C, T]
    """
    model.to(device)
    model.eval()
    import torch.nn.functional as F

    B, L = mixture.shape
    hop = chunk_size - overlap_samples if overlap_samples < chunk_size else chunk_size

    # Prepare output buffers
    with torch.no_grad():
        # determine number of sources by doing a single forward on zeros (lazy)
        dummy = torch.zeros(1, min(chunk_size, L)).to(device)
        out_dummy = model(dummy)
        C = out_dummy.shape[1]

    device_cpu = torch.device('cpu')
    output = torch.zeros(B, C, L, dtype=torch.float32, device=device_cpu)
    norm = torch.zeros(B, 1, L, dtype=torch.float32, device=device_cpu)

    # Precompute window for overlap-add
    if overlap_samples > 0 and chunk_size > 1:
        window = torch.hann_window(chunk_size, periodic=False, dtype=torch.float32, device=device_cpu)
    else:
        window = torch.ones(chunk_size, dtype=torch.float32, device=device_cpu)

    with torch.no_grad():
        for start in range(0, L, hop):
            end = start + chunk_size
            chunk = mixture[:, start:end].to(device)
            pad_len = 0
            if chunk.shape[-1] < chunk_size:
                pad_len = chunk_size - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad_len))

            est_chunk = model(chunk)  # [B, C, chunk_size]
            if pad_len > 0:
                est_chunk = est_chunk[:, :, :-pad_len]

            cur_len = est_chunk.shape[-1]
            # select window for this chunk (may be shorter at the end)
            w = window[:cur_len].to(device_cpu)
            w = w.view(1, 1, -1)

            # move est_chunk to CPU and apply window
            est_cpu = est_chunk.cpu()
            est_cpu = est_cpu * w

            # add to output and accumulate norm
            out_start = start
            out_end = start + cur_len
            output[:, :, out_start:out_end] += est_cpu
            norm[:, :, out_start:out_end] += w

    # FIX: avoid division by zero safely using broadcasting
    norm_safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    output = output / norm_safe

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--musdb-root', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--model', choices=['orig', 'ablated'], default='ablated')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'npu'], default='auto')
    parser.add_argument('--chunk-size', type=int, default=48000, help='inference chunk size in samples')
    parser.add_argument('--overlap', type=float, default=0.0, help='overlap in seconds for chunked inference (overlap-add)')
    parser.add_argument('--out-csv', type=str, default=None, help='path to write per-track metrics CSV')
    args = parser.parse_args()
    
    run_evaluation(args)


def run_evaluation(args):
    """Run evaluation given an args-like object with attributes:
    musdb_root, model_path, model, device, chunk_size, overlap, out_csv
    """
    device = get_device(args.device)
    if args.model == 'orig':
        model = TDANetBest(num_sources=4, sample_rate=16000, enc_kernel_size=4)
    else:
        model = TDANetAblated(num_sources=4, sample_rate=16000, enc_kernel_size=4)
        
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    # Use the `test` subset by default for evaluation
    dm = MusdbDataModule(args.musdb_root, subset='test', sr=16000)
    val_loader = dm.val_dataloader()

    # compute overlap in samples
    sr = dm.sr
    overlap_samples = int(args.overlap * sr)

    results = []
    src_names = ['vocals', 'drums', 'bass', 'other']

    for batch_idx, (mixture, sources) in enumerate(val_loader):
        # mixture shape: [B, T], sources: [B, C, T]
        est = sliding_inference(model, mixture, device, args.chunk_size, overlap_samples)

        B = est.shape[0]
        for i in range(B):
            est_i = est[i].cpu().numpy()
            ref_i = sources[i].cpu().numpy()

            # ensure same shape
            T = min(est_i.shape[-1], ref_i.shape[-1])
            est_i = est_i[..., :T]
            ref_i = ref_i[..., :T]

            # compute SI-SDR per source
            def si_sdr(est, ref, eps=1e-8):
                est_zm = est - est.mean()
                ref_zm = ref - ref.mean()
                denom = np.sum(ref_zm ** 2) + eps
                proj = (np.sum(est_zm * ref_zm) / denom) * ref_zm
                e_noise = est_zm - proj
                ratio = (np.sum(proj ** 2) + eps) / (np.sum(e_noise ** 2) + eps)
                return 10 * np.log10(ratio)

            num_src = est_i.shape[0]
            si_sdrs = []
            for s in range(num_src):
                si_sdrs.append(float(si_sdr(est_i[s], ref_i[s])))

            # FIX: Compute SDR/SIR/SAR using mir_eval (Natively supports Mono arrays)
            sdrs = [float('nan')] * num_src
            sirs = [float('nan')] * num_src
            sars = [float('nan')] * num_src
            isrs = [float('nan')] * num_src
            
            try:
                import mir_eval
                # compute_permutation=False forces it to strictly compare vocals->vocals, drums->drums
                _sdrs, _sirs, _sars, _ = mir_eval.separation.bss_eval_sources(ref_i, est_i, compute_permutation=False)
                
                sdrs = _sdrs.tolist()
                sirs = _sirs.tolist()
                sars = _sars.tolist()
            except ImportError:
                print('Error: mir_eval is not installed. Please run: !pip install mir_eval')
            except Exception as e:
                print(f'mir_eval math error on track segment: {e}')

            # collect row
            row = {'track': f'batch{batch_idx}_idx{i}'}
            for idx, name in enumerate(src_names[:num_src]):
                row[f'{name}_si-sdr'] = si_sdrs[idx]
                row[f'{name}_sdr'] = float(sdrs[idx])
                row[f'{name}_sir'] = float(sirs[idx])
                row[f'{name}_isr'] = float(isrs[idx])
                row[f'{name}_sar'] = float(sars[idx])

            # averages
            row['si-sdr_avg'] = float(np.mean(si_sdrs))
            row['sdr_avg'] = float(np.nanmean(sdrs))
            row['sir_avg'] = float(np.nanmean(sirs))
            row['isr_avg'] = float(np.nanmean(isrs))
            row['sar_avg'] = float(np.nanmean(sars))

            results.append(row)

    # write CSV
    if args.out_csv is None:
        out_dir = os.path.dirname(args.model_path)
        args.out_csv = os.path.join(out_dir, f'eval_{os.path.basename(args.model_path)}.csv')

    if len(results) > 0:
        keys = list(results[0].keys())
        with open(args.out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print('Wrote metrics to', args.out_csv)


if __name__ == '__main__':
    main()