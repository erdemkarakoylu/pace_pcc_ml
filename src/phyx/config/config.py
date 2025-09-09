"""Band clippi to match specified bandset in config/bandsets.yaml.
Pre-clipped feature file can be generated using
python scripts/clip_rrs_by_config.py \
  --in data_raw/df_rrs.pqt \
  --out data_pace_340_720/df_rrs.pqt \
  --name pace_10nm_340_720

"""

import argparse, yaml, pandas as pd
from pathlib import Path

def select_columns(df, spec):
    if spec["kind"] == "whitelist":
        wl = range(spec["start_nm"], spec["stop_nm"] + 1, spec["step_nm"])
        keep = [f"{spec['prefix']}{w}" for w in wl]
        # keep any non-Rrs ancillary columns too
        anc = [c for c in df.columns if not c.startswith(spec["prefix"])]
        return df[keep + anc]
    raise ValueError("Unsupported spec.kind")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/bandsets.yaml")
    ap.add_argument("--name", default="pace_10nm_340_720")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.cfg))[args.name]
    df = pd.read_parquet(args.inp)
    df2 = select_columns(df, spec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(args.out, index=False)
    print(f"wrote {args.out} with {df2.shape[1]} columns")
