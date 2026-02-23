# Satellites FOV Pass Finder

Finds satellite passes through a camera or custom field-of-view (FOV) using Skyfield and TLE data.

The script supports:
- Per-day TLE auto-selection (closest TLE file for each UTC day segment)
- Multiprocessing across time segments
- Single time-range mode or CSV batch mode
- Automatic time-file generation from Linux system .vid files
- Optional filtering to calibration satellites from satellites.json

## Project Files

- [CheckForSatellites_FOV.py](CheckForSatellites_FOV.py) — main script
- [CheckForSatellites_FOV_py26.py](CheckForSatellites_FOV_py26.py) — Python 2.6-compatible variant
- [build_time_file_from_system.py](build_time_file_from_system.py) — standalone time_file generator (Python 2.6+ and Python 3)
- [satellites.json](satellites.json) — optional calibration satellite list used by --calsats
- [requirements.txt](requirements.txt) — Python dependency list
- [install_dependencies.py](install_dependencies.py) — helper installer (installs from requirements.txt)

## Requirements

Python 3.9+

Install dependencies (recommended):

python install_dependencies.py

Alternative:

python -m pip install -r requirements.txt

The installer script upgrades pip and then installs from requirements.txt.

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

Choose exactly one time input source:
- --start-time + --end-time
- --time-file
- --time-from-system

## Usage

### 1) Single range with camera preset + auto TLE

python CheckForSatellites_FOV.py --tle-auto --cam_id 02G --start-time 20260223_000000 --end-time 20260225_235959 --workers 4

### 2) Single range with custom FOV and custom observer location

python CheckForSatellites_FOV.py --tle-file TLEs/TLE_20260223_120000_catalog.txt --fov 43 355 53 5 --latitude 43.19 --longitude -81.31 --start-time 20260223_000000 --end-time 20260223_235959

### 3) Batch mode from CSV with per-day auto TLE and multiprocessing

python CheckForSatellites_FOV.py --tle-auto --time-file ranges.csv --cam_id 01F --workers 6

### 4) Restrict search to calibration satellites

python CheckForSatellites_FOV.py --tle-auto --time-file ranges.csv --cam_id 01F --calsats

### 5) Build time-file from existing Linux system files

python CheckForSatellites_FOV.py --tle-auto --cam_id 01F --time-from-system --workers 4

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

## System Time Extraction (--time-from-system)

When you provide --time-from-system:
- The script requires Linux.
- It checks for /dump.vid.
- It scans /dump.vid/<cam_id>/ recursively for .vid files.
- Each .vid file contributes one row to a generated CSV:
	- beg_utc = file creation timestamp (or st_ctime fallback when creation time is unavailable)
	- end_utc = beg_utc + 10 minutes - 1 second
- Generated file name: time_file_from_system_<cam_id>.csv

This option is mutually exclusive with:
- --time-file
- --start-time / --end-time

And it requires:
- --cam_id

## Standalone Time-File Builder

You can pre-generate a `time_file` without running the full pass finder:

python build_time_file_from_system.py --cam-id 01F

Optional arguments:
- `--dump-dir` (default: `/dump.vid`)
- `--output` (default: `time_file_from_system_<cam-id>.csv`)

Example:

python build_time_file_from_system.py --cam-id 01F --output ranges_from_system.csv

Then run the main script with that file:

python CheckForSatellites_FOV.py --tle-auto --cam_id 01F --time-file ranges_from_system.csv

Compatibility:
- This utility script is designed to run on both Python 2.6 and Python 3.
- It is Linux-only because it scans `/dump.vid/<cam_id>/`.

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
- --time-from-system is Linux-only and requires /dump.vid/<cam_id>/.

## Python 2.6 Script

For legacy environments, use:

python CheckForSatellites_FOV_py26.py [options]

Compatibility notes:
- Uses optparse and multiprocessing.Pool (instead of argparse and concurrent.futures).
- Maintains the same core workflow: per-day segments, optional --time-from-system, closest-TLE per segment, and CSV output.
- Uses naive UTC datetime handling to avoid Python 3 timezone APIs.
- Dependency support on Python 2.6 may vary by platform (especially skyfield/pandas/numpy).
