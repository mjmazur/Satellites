# Python script to track satellite positions through a specified field-of-view.

# Notes on execution model:
# 1) Build one task per UTC day segment in each requested time range.
# 2) Optionally resolve the closest TLE for each segment.
# 3) Process segments sequentially or with multiprocessing.
# 4) Merge results and write a single CSV.

import argparse
import json
import os
import glob
import platform
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
from datetime import datetime, timedelta, timezone


def generate_time_file_from_system(cam_id):
    """
    Builds a CSV time-file from existing .vid files for a camera on Linux.

    Logic:
    - Require /dump.vid to exist.
    - Read .vid files from /dump.vid/<cam_id>/ (recursive).
    - File creation time is treated as begin time.
    - Each file represents 10 minutes of data.
    """
    if platform.system() != 'Linux':
        raise OSError("Error: --time-from-system is only supported on Linux.")

    dump_vid_dir = '/dump.vid'
    if not os.path.isdir(dump_vid_dir):
        raise FileNotFoundError("Error: Required directory '/dump.vid' was not found.")

    cam_dir = os.path.join(dump_vid_dir, cam_id)
    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"Error: Camera directory '{cam_dir}' was not found.")

    vid_files = []
    for root, _, files in os.walk(cam_dir):
        for file_name in files:
            if file_name.lower().endswith('.vid'):
                vid_files.append(os.path.join(root, file_name))

    if not vid_files:
        raise FileNotFoundError(f"Error: No .vid files found under '{cam_dir}'.")

    rows = []
    warned_about_linux_ctime = False
    for vid_path in vid_files:
        stat_info = os.stat(vid_path)

        # Linux generally does not expose true file creation time.
        # Use st_birthtime when available; otherwise fallback to st_ctime.
        if hasattr(stat_info, 'st_birthtime'):
            begin_epoch = stat_info.st_birthtime
        else:
            begin_epoch = stat_info.st_ctime
            if not warned_about_linux_ctime:
                print("Warning: Linux does not always expose true creation time; using st_ctime as file begin time.")
                warned_about_linux_ctime = True

        begin_utc = datetime.fromtimestamp(begin_epoch, tz=timezone.utc)
        end_utc = begin_utc + timedelta(minutes=10) - timedelta(seconds=1)

        rows.append({
            'filename': os.path.basename(vid_path),
            'beg_utc': begin_utc.strftime('%Y%m%d:%H:%M:%S.%f'),
            'end_utc': end_utc.strftime('%Y%m%d:%H:%M:%S.%f'),
            '_sort_key': begin_utc
        })

    rows.sort(key=lambda row: row['_sort_key'])
    for row in rows:
        row.pop('_sort_key', None)

    output_time_file = f"time_file_from_system_{cam_id}.csv"
    pd.DataFrame(rows).to_csv(output_time_file, index=False)
    print(f"Created time-file from system data: {output_time_file} ({len(rows)} entries)")

    return output_time_file

def parse_time_utc(time_value):
    """
    Parses a UTC time value into a timezone-aware datetime in UTC.
    Accepts datetime objects or strings in either CSV or CLI formats.
    """
    # Accept datetime objects directly and normalize to UTC.
    if isinstance(time_value, datetime):
        if time_value.tzinfo is None:
            return time_value.replace(tzinfo=timezone.utc)
        return time_value.astimezone(timezone.utc)

    # Accept both time formats used by this script:
    # - CSV style: YYYYMMDD:HH:MM:SS.ssssss
    # - CLI style: YYYYMMDD_hhmmss
    for time_format in ('%Y%m%d:%H:%M:%S.%f', '%Y%m%d_%H%M%S'):
        try:
            return datetime.strptime(str(time_value), time_format).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Error: Invalid time format for '{time_value}'.")


def split_time_range_by_day(start_utc, end_utc):
    """
    Splits a UTC datetime range into day-bounded segments.
    """
    # Guard against invalid ranges.
    if end_utc < start_utc:
        return []

    # Produce inclusive day-bounded segments:
    # [segment_start, segment_end] where each segment stays within one UTC date.
    segments = []
    current_start = start_utc
    while current_start <= end_utc:
        next_day_midnight = datetime(
            current_start.year,
            current_start.month,
            current_start.day,
            tzinfo=timezone.utc
        ) + timedelta(days=1)
        current_end = min(end_utc, next_day_midnight - timedelta(seconds=1))
        segments.append((current_start, current_end))
        current_start = next_day_midnight

    return segments


def find_closest_tle_file(start_time):
    """
    Finds the TLE file in the 'TLEs/' directory that is closest in date to the start_time.
    TLE filenames are expected in the format 'TLE_YYYYMMDD_hhmmss_*.txt'.
    """
    # TLE lookup directory and expected filename pattern.
    tle_dir = "TLEs"
    if not os.path.isdir(tle_dir):
        raise FileNotFoundError(f"Error: The TLE directory '{tle_dir}' was not found.")

    tle_pattern = os.path.join(tle_dir, "TLE_*.txt")
    tle_files = glob.glob(tle_pattern)

    if not tle_files:
        raise FileNotFoundError(f"Error: No TLE files found in '{tle_dir}' matching the pattern 'TLE_*.txt'.")

    # Compare naive datetimes because TLE filename timestamps are timezone-agnostic.
    start_time_dt = parse_time_utc(start_time).replace(tzinfo=None)

    closest_file = None
    min_time_diff = timedelta.max

    # Scan all candidate files and keep the one with minimum absolute time difference.
    for tle_file in tle_files:
        filename = os.path.basename(tle_file)
        try:
            # Extract the timestamp part of the filename, e.g., '20230101_120000'
            parts = filename.split('_')
            timestamp_str = f"{parts[1]}_{parts[2]}"
            tle_time_dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            # Calculate the absolute difference
            time_diff = abs(start_time_dt - tle_time_dt)
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_file = tle_file
        except (IndexError, ValueError):
            print(f"Warning: Could not parse timestamp from filename '{filename}'. Skipping.")
            continue
            
    if closest_file is None:
        raise FileNotFoundError("Error: Could not determine the closest TLE file.")

    return closest_file


def create_observer(observer_config):
    """
    Creates a Skyfield observer from a serializable config dictionary.
    """
    # Keep observer construction in one place so task payloads stay serializable.
    return wgs84.latlon(
        observer_config['lat'],
        observer_config['lon'],
        elevation_m=observer_config['elev']
    )


def find_satellite_passes(observer, tle_file, start_utc, end_utc, fov, calsats_file, filename):
    """
    Finds satellite passes and appends them to a results list.
    """
    # Unpack the field-of-view
    alt_min, az_min, alt_max, az_max = fov

    # Load all satellites from a single selected TLE file for this segment.
    ts = load.timescale()
    all_sats_by_norad = {sat.model.satnum: sat for sat in load.tle_file(tle_file)}
    print(f"Loaded {len(all_sats_by_norad)} total satellites from {tle_file}.")

    # Determine the target satellite set:
    # - with --calsats: only NORAD IDs listed in satellites.json
    # - otherwise: every satellite in the TLE
    sats_to_check = []
    cal_sat_norads = set()
    if calsats_file:
        # If calsats flag is used, only check satellites from the JSON file
        try:
            with open(calsats_file, 'r') as f:
                cal_sats_info = json.load(f)
            
            for cal_sat in cal_sats_info:
                norad_id_str = cal_sat.get('norad')
                if norad_id_str:
                    try:
                        norad_id = int(norad_id_str)
                        cal_sat_norads.add(norad_id)
                        if norad_id in all_sats_by_norad:
                            sats_to_check.append(all_sats_by_norad[norad_id])
                        else:
                            print(f"Warning: Calibration satellite '{cal_sat.get('name')}' (NORAD {norad_id}) not found in TLE file.")
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid NORAD ID '{norad_id_str}' in {calsats_file}. It must be an integer.")
            
            print(f"Checking {len(sats_to_check)} calibration satellites specified in {calsats_file}.")
        
        except FileNotFoundError:
            print(f"Error: Calibration satellite file not found at '{calsats_file}'. Exiting.")
            return
        except (json.JSONDecodeError, KeyError):
            print(f"Error: Could not parse calibration satellite file '{calsats_file}'. Exiting.")
            return
    else:
        # Otherwise, check all satellites from the TLE file
        sats_to_check = list(all_sats_by_norad.values())



    results = []

    # Build one timestamp per second in the segment and evaluate all satellites on that grid.
    time_points = []
    current_time = start_utc
    while current_time <= end_utc:
        time_points.append(current_time)
        current_time += timedelta(seconds=1)

    if not time_points:
        print("Error: The provided time range is invalid or too short.")
        return

    timestamps = ts.from_datetimes(time_points)
    
    print(f"\nSearching for passes between {start_utc.isoformat()} and {end_utc.isoformat()}...")
    
    # Track whether at least one pass is detected in this segment.
    passes_found = False
    
    # Iterate through the selected satellites to find their passes
    for sat in sats_to_check:
        # Convert satellite trajectory to topocentric frame for this observer.
        difference = sat - observer
        topocentric = difference.at(timestamps)
        
        alt, az, _ = topocentric.altaz()
        
        # Handle the azimuth wraparound case
        if az_min <= az_max:
            az_mask = (az.degrees >= az_min) & (az.degrees <= az_max)
        else:
            # This handles ranges like 355 to 5 degrees
            az_mask = (az.degrees >= az_min) | (az.degrees <= az_max)

        in_fov_mask = (
            (alt.degrees >= alt_min) & (alt.degrees <= alt_max) &
            az_mask
        )
        
        # Detect transitions False->True (entry) and True->False (exit).
        change_indices = np.where(np.diff(in_fov_mask))[0]
        
        is_currently_in_view = in_fov_mask[0]
        pass_start_index = 0 if is_currently_in_view else None

        # Traverse transitions to create pass records.
        for idx in change_indices:
            if is_currently_in_view:
                # Satellite is exiting the FOV
                entry_idx = pass_start_index
                exit_idx = idx
                results.append({
                    'filename': filename,
                    'satellite_name': sat.name,
                    'satellite_norad_id': sat.model.satnum,
                    'time_enters': timestamps[entry_idx].utc_iso(),
                    'time_leaves': timestamps[exit_idx].utc_iso(),
                    'azimuth_enters': f"{az.degrees[entry_idx]:.2f}",
                    'altitude_enters': f"{alt.degrees[entry_idx]:.2f}",
                    'azimuth_leaves': f"{az.degrees[exit_idx]:.2f}",
                    'altitude_leaves': f"{alt.degrees[exit_idx]:.2f}"
                })
                passes_found = True
                is_currently_in_view = False
                pass_start_index = None
            else:
                # Satellite is entering the FOV
                is_currently_in_view = True
                pass_start_index = idx + 1

        # Handle a pass that is still ongoing at the end of the time window
        if is_currently_in_view:
            entry_idx = pass_start_index
            results.append({
                'filename': filename,
                'satellite_name': sat.name,
                'satellite_norad_id': sat.model.satnum,
                'time_enters': timestamps[entry_idx].utc_iso(),
                'time_leaves': 'Still in view',
                'azimuth_enters': f"{az.degrees[entry_idx]:.2f}",
                'altitude_enters': f"{alt.degrees[entry_idx]:.2f}",
                'azimuth_leaves': 'N/A',
                'altitude_leaves': 'N/A'
            })
            passes_found = True

    if not passes_found:
        print(f"No satellite passes found for '{filename}'.")

    return results


def process_time_segment(task):
    """
    Worker function for multiprocessing: processes one day-bounded segment.
    """
    # Reconstruct observer inside worker process from serializable task payload.
    observer = create_observer(task['observer_config'])
    tle_file_to_use = task.get('resolved_tle_file', task['tle_file'])

    # Fallback resolution path if task was not pre-resolved in the parent process.
    if task['tle_auto'] and tle_file_to_use is None:
        tle_file_to_use = find_closest_tle_file(task['start_utc'])

    return find_satellite_passes(
        observer,
        tle_file_to_use,
        task['start_utc'],
        task['end_utc'],
        task['fov'],
        task['calsats_file'],
        task['filename']
    )


def main():
    """
    Parses command-line arguments and runs the satellite tracking function.
    """
    parser = argparse.ArgumentParser(
        description="Find satellite passes through a specified field-of-view.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--tle-file',
        default=None,
        help="Path to the TLE file containing satellite data.\n"
             "You can download TLE files from CelesTrak: https://celestrak.org/"
    )
    parser.add_argument(
        '--tle-auto',
        action='store_true',
        help="Automatically select the TLE file from the 'TLEs/' directory."
    )
    parser.add_argument(
        '--time-file',
        default=None,
        help="Path to a CSV file containing time ranges. Overrides start/end times."
    )
    parser.add_argument(
        '--time-from-system',
        action='store_true',
        help="Build a time-file from .vid files in /dump.vid/<cam_id>/ (Linux only)."
    )
    parser.add_argument(
        '--start-time',
        default=None,
        help="Start time in YYYYMMDD_hhmmss format."
    )
    parser.add_argument(
        '--end-time',
        default=None,
        help="End time in YYYYMMDD_hhmmss format."
    )
    parser.add_argument(
        '--cam_id',
        choices=['01F', '01G', '02F', '02G'],
        default=None,
        help="Camera ID for predefined site and FOV. Overrides --site and --fov."
    )
    parser.add_argument(
        '--fov',
        nargs=4,
        type=float,
        metavar=('ALT_MIN', 'AZ_MIN', 'ALT_MAX', 'AZ_MAX'),
        help="Field-of-view in degrees. Required if --cam_id is not used."
    )
    parser.add_argument(
        '--site',
        choices=['tavistock', 'elginfield'],
        default=None,
        help="Name of the observation site. Used if --cam_id is not specified."
    )
    parser.add_argument(
        '--latitude',
        type=float,
        default=None,
        help="Custom observer latitude in degrees. Overrides --site."
    )
    parser.add_argument(
        '--longitude',
        type=float,
        default=None,
        help="Custom observer longitude in degrees. Overrides --site."
    )
    parser.add_argument(
        '--calsats',
        action='store_true',
        help="Enable this flag to check for calibration satellites from 'satellites.json'."
    )
    parser.add_argument(
        '--output-file',
        default='calsat_matches.csv',
        help="Path to the output CSV file (default: calsat_matches.csv)."
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of worker processes to use (default: CPU count - 1). Use 1 to disable multiprocessing."
    )

    args = parser.parse_args()

    # --- Input Validation ---
    # Validate mutually exclusive and required argument combinations early.
    if not args.tle_file and not args.tle_auto:
        parser.error("You must specify either --tle-file or --tle-auto.")

    if args.tle_file and args.tle_auto:
        parser.error("You cannot use both --tle-file and --tle-auto.")

    if args.time_from_system and (args.time_file or args.start_time or args.end_time):
        parser.error("--time-from-system cannot be used with --time-file, --start-time, or --end-time.")

    if not args.time_from_system and not args.time_file and not (args.start_time and args.end_time):
        parser.error("You must specify one of: --time-from-system, --time-file, or both --start-time and --end-time.")
    
    if args.time_file and (args.start_time or args.end_time):
        parser.error("You cannot use --time-file with --start-time or --end-time.")

    if (args.start_time and not args.end_time) or (not args.start_time and args.end_time):
        parser.error("--start-time and --end-time must be used together.")
    
    if args.cam_id and (args.site or args.fov or args.latitude or args.longitude):
        parser.error("--cam_id cannot be used with --site, --fov, --latitude, or --longitude.")
        
    if not args.cam_id and not args.fov:
        parser.error("You must specify either --cam_id or --fov.")

    if args.time_from_system and not args.cam_id:
        parser.error("--time-from-system requires --cam_id to identify /dump.vid/<cam_id>/.")

    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    # --- Site and Camera Configuration ---
    # Camera presets map to a site and fixed FOV.
    SITES = {
        'elginfield': {'lat': 43.192909, 'lon': -81.315655, 'elev': 327},
        'tavistock': {'lat': 43.264027, 'lon': -80.772143, 'elev': 330}
    }
    CAMERAS = {
        '01F': {'site': 'tavistock', 'fov': [75, 288, 85, 298]},
        '01G': {'site': 'tavistock', 'fov': [42, 322, 52, 332]},
        '02F': {'site': 'elginfield', 'fov': [67, 59, 77, 69]},
        '02G': {'site': 'elginfield', 'fov': [43, 355, 53, 5]}
    }

    if args.cam_id:
        cam_info = CAMERAS[args.cam_id]
        site_info = SITES[cam_info['site']]
        fov = cam_info['fov']
        observer_config = {'lat': site_info['lat'], 'lon': site_info['lon'], 'elev': site_info['elev']}
        observer = create_observer(observer_config)
        print(f"Using Camera ID '{args.cam_id}': Site='{cam_info['site']}', FOV={fov}")
    else:
        fov = args.fov
        if args.latitude is not None and args.longitude is not None:
            observer_config = {'lat': args.latitude, 'lon': args.longitude, 'elev': 0}
            observer = create_observer(observer_config)
            print(f"Using custom observer location: Latitude={args.latitude}°, Longitude={args.longitude}°")
        else:
            site = args.site if args.site else 'elginfield'
            site_info = SITES[site]
            observer_config = {'lat': site_info['lat'], 'lon': site_info['lon'], 'elev': site_info['elev']}
            observer = create_observer(observer_config)
            print(f"Using site: '{site}' at Latitude={site_info['lat']}°, Longitude={site_info['lon']}°")


    # If --calsats is enabled, limit processing to NORAD IDs listed in satellites.json.
    calsats_file = 'satellites.json' if args.calsats else None
    tasks = []

    if args.time_from_system:
        try:
            args.time_file = generate_time_file_from_system(args.cam_id)
        except (OSError, FileNotFoundError) as e:
            print(str(e))
            return

    # Build processing tasks (one per day-bounded segment).
    # This is done before execution so the runner can choose sequential or parallel mode.
    if args.time_file:
        try:
            time_df = pd.read_csv(args.time_file)
            if 'beg_utc' not in time_df.columns or 'end_utc' not in time_df.columns:
                print("Error: The time file must contain 'beg_utc' and 'end_utc' columns.")
                return
        except FileNotFoundError:
            print(f"Error: Time file not found at '{args.time_file}'.")
            return
        except Exception as e:
            print(f"Error reading time file: {e}")
            return

        # Process each time range from the CSV file
        for row_num, (index, row) in enumerate(time_df.iterrows(), start=1):
            start_time_str = row['beg_utc']
            end_time_str = row['end_utc']
            filename = row.get('filename', f"Row {row_num}")

            print(f"\n{'='*60}")
            print(f"Processing Time Range for '{filename}'")
            print(f"Start: {start_time_str}, End: {end_time_str}")
            print(f"{'='*60}\n")

            try:
                start_utc = parse_time_utc(start_time_str)
                end_utc = parse_time_utc(end_time_str)
            except ValueError as e:
                print(str(e))
                continue

            if end_utc < start_utc:
                print(f"Error: End time is before start time for '{filename}'. Skipping.")
                continue

            # Split each row into day-bounded segments so each day can use its own closest TLE.
            for segment_start, segment_end in split_time_range_by_day(start_utc, end_utc):
                tasks.append({
                    'observer_config': observer_config,
                    'tle_file': args.tle_file,
                    'tle_auto': args.tle_auto,
                    'start_utc': segment_start,
                    'end_utc': segment_end,
                    'fov': fov,
                    'calsats_file': calsats_file,
                    'filename': filename
                })
    else:
        # Process the single time range from command-line arguments
        try:
            start_utc = parse_time_utc(args.start_time)
            end_utc = parse_time_utc(args.end_time)
        except ValueError as e:
            print(str(e))
            return

        if end_utc < start_utc:
            print("Error: End time is before start time.")
            return

        # Split single CLI range into day-bounded segments.
        for segment_start, segment_end in split_time_range_by_day(start_utc, end_utc):
            tasks.append({
                'observer_config': observer_config,
                'tle_file': args.tle_file,
                'tle_auto': args.tle_auto,
                'start_utc': segment_start,
                'end_utc': segment_end,
                'fov': fov,
                'calsats_file': calsats_file,
                'filename': 'command_line_input'
            })

    results = []
    if not tasks:
        print("No valid time segments to process.")
        return

    if args.tle_auto:
        # Pre-resolve and print one TLE selection line per segment.
        # This gives an explicit run summary and avoids repeated selection in workers.
        print(f"Selecting closest TLE file per day-segment for {len(tasks)} segment(s).")
        valid_tasks = []
        for i, task in enumerate(tasks, start=1):
            try:
                resolved_tle = find_closest_tle_file(task['start_utc'])
                task['resolved_tle_file'] = resolved_tle
                print(
                    f"  [{i}/{len(tasks)}] {task['filename']} | "
                    f"{task['start_utc'].isoformat()} -> {task['end_utc'].isoformat()} | "
                    f"TLE: {resolved_tle}"
                )
                valid_tasks.append(task)
            except (FileNotFoundError, ValueError) as e:
                print(str(e))
                print(f"Skipping segment for '{task['filename']}' due to TLE selection error.")
        tasks = valid_tasks

        if not tasks:
            print("No valid time segments to process after TLE selection.")
            return

    # Run tasks in parallel when beneficial; otherwise keep a single-process path.
    worker_count = min(args.workers, len(tasks))
    if worker_count > 1:
        print(f"Processing {len(tasks)} segment(s) using {worker_count} worker processes.")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Submit all segments first, then collect as they complete.
            futures = [executor.submit(process_time_segment, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except (FileNotFoundError, ValueError) as e:
                    print(str(e))
    else:
        print(f"Processing {len(tasks)} segment(s) sequentially.")
        # Debug-friendly fallback mode and useful for constrained systems.
        for task in tasks:
            try:
                results.extend(process_time_segment(task))
            except (FileNotFoundError, ValueError) as e:
                print(str(e))

    # Determine the output filename
    output_filename = args.output_file
    if args.cam_id and args.output_file == 'calsat_matches.csv':
        output_filename = f"calsats_matched_{args.cam_id}.csv"

    # Save the results to a CSV file
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_filename, index=False)
        print(f"\nSaved {len(results)} matches to '{output_filename}'.")
    else:
        print("\nNo satellite passes were found to save.")

if __name__ == "__main__":
    main()
