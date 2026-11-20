# QA findings: OASIM Rrs and NOBM surface outputs (1998–2021 run)

_Session: data QA/QC · Completed 2026-11-19 · Units updated 2026-11-20 · Notebook: `notebooks/oasim2021/02_inspect_nobm.ipynb` (EC2 clone)_

Scope: an inventory of all files, plus detailed checks on **January 1998** and the **January→February 1998** boundary. A full QA of all 288 months was deliberately not run separately: it will be built into the ingestion pipeline as automatic checks (see the list at the end).

---

## 1. Data location and inventory

| Set | S3 prefix | Files | Size | Per-day size |
|---|---|---|---|---|
| OASIM Rrs | `s3://mio-water-entero-data/shared/pcc/oasim/` | 288 (`rrsYYYYMM.nc`, 1998-01 … 2021-12) | 1.18 TB | 135,053,568 B (= 501 × 288 × 234 × 4) |
| NOBM surface | `s3://mio-water-entero-data/shared/pcc/nobm/` | 288 (`nobmYYYYMM.nc`, 1998-01 … 2021-12) | 70.9 GB | ≈ 8.1 MB |

- Every month is present in both sets (outer merge on YYYYMM: 288 both, 0 one-sided).
- **Real calendar:** leap-year Februaries (2000, 2004, … 2020) have 29 days in both sets; day counts agree for every month.
- **Same run** as the previous (December 2021) training set used for the preprint (Susanne, 2026-11-18).
- **Daily resolution throughout:** each monthly file holds one global simulation per day (Ian). This includes `pp`/`tpp`, so they are genuine daily values, not monthly means.
- Derived outputs go to `s3://mio-water-entero-data/shared/pcc/erdemk/` (write access confirmed, OK'd by Ian).

## 2. File structure

**Rrs (`rrsYYYYMM.nc`)**
- Dimensions: `date` (28–31) × `lon` 288 × `lat` 234 × `wavelength` 501.
- One variable: `rrs` (date, lon, lat, wavelength), float32.
- Coordinates:
  - `lon`: 0–358.75 in 1.25° steps (0–360 convention).
  - `lat`: −84.0 to 70.74 in ≈0.664° steps, south to north.
  - `wavelength`: 250–750 nm, 1 nm, int32.
- **No units or attributes** (variable or global). Fill value NaN.
- Storage is uncompressed and contiguous (not chunked), so one pixel's full spectrum is contiguous (≈2 KB).
- **Read strategy:** read whole days (≈135 MB) and subsample in memory. Reading scattered pixels over S3 is very slow.

**NOBM (`nobmYYYYMM.nc`)**
- Dimensions: `date` × `lon` 288 × `lat` 234 × `component` 6.
- **`lon`/`lat` are index numbers (0–287, 0–233), not degrees.** Use the Rrs coordinates (see §3).
- `component` = `dia, chl, cya, coc, din, pha`.
- 20 variables, float32, uncompressed, contiguous, fill NaN, no attributes:
  - (date, lon, lat): `tot, dtc, pic, cdc, t, s, alk, dic, doc, fco, h, irn, pco, tpp, rnh, rno, sil, zoo`
  - (date, lon, lat, component): `phy` (the six groups), `pp` (primary production per group)
- Array order is (lon, lat), not (lat, lon), in both sets. Transpose for maps.

## 3. Grid alignment
- The land/ocean masks of NOBM `t`, NOBM `tot` and Rrs(443) are **identical** on 1998-01-01: 26,695 of 67,392 cells are missing in each, with **0 mismatched cells**. Flipping latitude gives 27,298 mismatches.
- **So NOBM cell (i, j) is Rrs cell (i, j), in the same latitude order.** Copy the Rrs `lon`/`lat` onto NOBM.
- Ocean cells per day ≈ 40,697. The missing ≈40% is land, plus areas outside the domain (NOBM covers only open ocean deeper than 200 m, −84° to 72°).

## 4. Values (January 1998, ocean cells)
No signs of decoding errors: no values near 1e38, no garbage negatives, nothing constant.

| Variable | min | p01 | median | p99 | max | Notes |
|---|---|---|---|---|---|---|
| t | −0.99 | 0.66 | 18.98 | 29.16 | 29.75 | °C plausible |
| s | 32.19 | 32.66 | 34.55 | 36.81 | 37.22 | psu plausible |
| h | 5 | 5 | 49.1 | 200 | 200 | **bounded at 5 and 200 m** (question to Cecile) |
| tot | 2.4e-14 | 0.016 | 0.13 | 0.82 | 4.66 | mg chl m⁻³ |
| phy[dia] | 1.8e-33 | 1.6e-08 | 0.0080 | 0.387 | 1.58 | |
| phy[chl] | 0 | 2.2e-27 | 0.0028 | 0.244 | 1.16 | |
| phy[cya] | 2.3e-33 | 2.1e-32 | 0.0031 | 0.091 | 0.46 | |
| phy[coc] | 0 | 7.2e-18 | 0.00016 | 0.248 | 1.17 | |
| phy[din] | 2.9e-33 | 9.0e-33 | 1.1e-18 | **0.0031** | 0.55 | **almost absent; signal in <1% of cells** |
| phy[pha] | 6.1e-37 | 1.0e-34 | 7.6e-19 | 0.782 | 4.66 | Phaeocystis bloom equals the `tot` maximum |
| pco | **2.1e-08** | 208 | 337 | 563 | 1471 | near-zero minimum in a few cells (question to Cecile) |
| fco | −2.99 | −1.65 | −0.06 | 1.34 | 8.49 | both signs expected (flux) |
| tpp | 0 | 2.97 | 291 | 1163 | 4055 | ~4,500 zeros (Arctic polar night) |
| alk / dic | 2108 / 1814 | — | 2360 / 2074 | — | 2430 / 2267 | µM, plausible |

Other variables (`dtc, pic, cdc, doc, irn, rnh, rno, sil, zoo`, `pp[*]`) are in plausible ranges; see the notebook.

- **`tot` equals the sum of `phy` over the six components exactly** (max abs diff 0.0). `tot` is derived, not stored independently.
- **Floor values (~1e-33)** appear in every group, as a model numerical floor. They are harmless with the current objective (squared error on raw values) but would dominate any log transform. A floor or cutoff is to be decided. Susanne thinks the tiny non-zero values are probably zeros filled with small values (a model artefact), related to the `pco` minimum; she is checking.

## 5. Continuity (Jan 31 → Feb 1, 1998)
Median absolute change over ocean cells:

| Variable | Typical daily change (within January) | Jan 31 → Feb 1 |
|---|---|---|
| t | 0.0208 | 0.0171 |
| tot | 0.00726 | 0.00766 |
| rrs(443) | 0.00029 (Jan 30→31) | 0.00027 |

No jump at the file boundary.

## 6. Rrs or nLw?
- Rrs(443) ≈ 0.001–0.0125 on 1998-01-01, the typical magnitude of Rrs in sr⁻¹.
- Spectral shapes are as expected: the clear gyre peaks in the blue, the peak drops and shifts to green as chlorophyll rises, and values are ≈0 above ~650 nm.
- **Conclusion: the files contain Rrs (sr⁻¹), not nLw.**

## 7. Units
No units are stored in the files. Sources: Cecile's model-code list (authoritative, consistent with the value ranges), plus follow-ups from Susanne and Ian (2026-11-20).

| Variable | Units | Source |
|---|---|---|
| rno (nitrate), rnh (ammonium), sil (silica) | µM | Cecile, code list |
| irn (iron) | nM | Cecile, code list |
| phy: dia, chl, cya, coc | mg chl m⁻³ | Cecile, code list |
| phy: din, pha | mg chl m⁻³ | Susanne ("if I remember correctly"; six groups, no change from the earlier subset). Near-certain. |
| tot | mg chl m⁻³ (sum of the six groups) | derived |
| zoo (herbivores) | mg chl m⁻³ (chl equivalent) | Cecile, code list |
| dtc (detrital carbon) | µgC l⁻¹ | Cecile, code list (N/C detritus) |
| pic | µgC l⁻¹ | Cecile |
| dic | µM | Cecile, code list |
| pco | µatm | Cecile, code list |
| pp, tpp | mgC m⁻² d⁻¹, **daily values** | Cecile; Ian (daily simulations in monthly files) |
| rrs | sr⁻¹ | this QA (§6) |

Note: Cecile's second list (with question marks) conflicts with the code list for `dtc`, `doc`, `irn`, `sil` and `pp`. Where they conflict, the code list matches the value ranges, so the code list is used.

**Still open:**
- `doc`: µM (code list; the values favour it) or µgC l⁻¹?
- `cdc`: what is it? "m⁻¹" is implausible given values of 0.5–70.
- `fco`, `alk`: units.
- Confirm `t`, `s`, `h` are °C, psu and m.

## 8. Checks to build into the ingestion pipeline
Apply to every file or day as it's read; log the results and fail or flag as appropriate:
1. YYYYMM pairing of Rrs and NOBM; day count matches the calendar and agrees between the sets.
2. Coordinates and shape identical to the reference (288 × 234; Rrs lat/lon; 501 wavelengths 250–750).
3. Land/ocean mask identical between Rrs and NOBM (and stable from day to day, or log the changes).
4. Value-range limits per variable, from the table above with margins. Flag values near 1e38, NaNs inside the ocean mask, negative concentrations.
5. `tot` = sum of `phy` (tolerance about 1e-6 relative).
6. **`tpp` = sum of the six `pp` components** (not yet checked).
7. Continuity at month boundaries: the step change compared with the within-month median daily change (flag if more than ~5×).
8. Rrs sanity: magnitude range and spectral shape (e.g. Rrs above 700 nm ≈ 0).