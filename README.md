# Satellites FOV Pass Finder

Finds satellite passes through a camera or custom field-of-view (FOV) using Skyfield and TLE data.

The script supports:
- Per-day TLE auto-selection (closest TLE file for each UTC day segment)
- Multiprocessing across time segments
- Single time-range mode or CSV batch mode
- Optional filtering to calibration satellites from satellites.json

## Project Files

- [CheckForSatellites_FOV.py](CheckForSatellites_FOV.py) — main script
- [satellites.json](satellites.json) — optional calibration satellite list used by --calsats

## Requirements

Python 3.9+

Install dependencies:

pip install numpy pandas skyfield

## TLE Files

If using --tle-auto, place TLE files in a folder named TLEs at the project root.

Expected filename pattern:

TLE_YYYYMMDD_hhmmss_*.txt

Example:

TLE_20260223_120000_catalog.txt

The script selects the closest TLE file for each day segment in the requested UTC time range.

## Time Input Formats

Accepted UTC formats:
- CSV style: YYYYMMDD:HH:MM:SS.ssssss
- CLI style: YYYYMMDD_hhmmss

## Usage

### 1) Single range with camera preset + auto TLE

python CheckForSatellites_FOV.py --tle-auto --cam_id 02G --start-time 20260223_000000 --end-time 20260225_235959 --workers 4

### 2) Single range with custom FOV and custom observer location

python CheckForSatellites_FOV.py --tle-file TLEs/TLE_20260223_120000_catalog.txt --fov 43 355 53 5 --latitude 43.19 --longitude -81.31 --start-time 20260223_000000 --end-time 20260223_235959

### 3) Batch mode from CSV with per-day auto TLE and multiprocessing

python CheckForSatellites_FOV.py --tle-auto --time-file ranges.csv --cam_id 01F --workers 6

### 4) Restrict search to calibration satellites

python CheckForSatellites_FOV.py --tle-auto --time-file ranges.csv --cam_id 01F --calsats

## CSV Input (for --time-file)

Required columns:
- beg_utc
- end_utc

Optional column:
- filename (used as a label in output rows)

Example:

filename,beg_utc,end_utc
capture_a,20260223:00:00:00.000000,20260223:02:00:00.000000
capture_b,20260224:10:30:00.000000,20260225:01:00:00.000000

## Camera Presets

- 01F: Tavistock, FOV [75, 288, 85, 298]
- 01G: Tavistock, FOV [42, 322, 52, 332]
- 02F: Elginfield, FOV [67, 59, 77, 69]
- 02G: Elginfield, FOV [43, 355, 53, 5]

## Output

Default output file:

calsat_matches.csv

If --cam_id is provided and output file is not overridden, output name becomes:

calsats_matched_<CAM_ID>.csv

Output columns include:
- filename
- satellite_name
- satellite_norad_id
- time_enters
- time_leaves
- azimuth_enters
- altitude_enters
- azimuth_leaves
- altitude_leaves

## Notes

- Use --workers 1 to disable multiprocessing.
- With --tle-auto, the script prints a per-segment summary showing which TLE file was selected.
- If no passes are found, no CSV rows are written.
