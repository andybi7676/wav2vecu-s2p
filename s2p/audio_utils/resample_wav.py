import os
import os.path as osp
import librosa as lb
import tqdm
import soundfile as sf
import multiprocessing as mp

manifest_root = "/work/c/LibriTTS/manifest"
splits = ["train"]
out_dir = '/work/c/LibriTTS/sr_16k'

os.makedirs(out_dir, exist_ok=True)
tgt_sr = 16_000
for split in splits:
    os.makedirs(osp.join(out_dir, split), exist_ok=True)
    tsv_fp = osp.join(manifest_root, split+".tsv")
    print(f"Processing split: {split}")
    with open(tsv_fp, 'r') as fr, open(osp.join(out_dir, f"rs_16k_{split}_unsuccessed.log"), 'w') as log_fw:
        lines = fr.readlines()
        audio_root = lines[0].strip()
        # cols = lines[0].strip()
        def resample(line):
            fname = line.strip().split('\t')[0].split('.')[0]
            fp = osp.join(audio_root, f"{fname}.wav")
            new_dir = osp.join(out_dir, split, '/'.join(fname.split('/')[:-1]))
            os.makedirs(new_dir, exist_ok=True)
            new_fp = osp.join(out_dir, split, f"{fname}.wav")
            if not osp.exists(new_fp):
                try:
                    wav, sr = lb.load(fp)
                    new_wav = lb.resample(wav, orig_sr=sr, target_sr=tgt_sr)
                    sf.write(new_fp, new_wav, tgt_sr)
                    return 1
                except:
                    return new_fp
        pool = mp.Pool(16)
        results = list(tqdm.tqdm(pool.imap(resample, lines[1:]), total=len(lines[1:])))
        for res in results:
            if res != 1:
                print(res, file=log_fw)
            


