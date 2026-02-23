# Python script to track satellite positions through a specified field-of-view.

import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
from datetime import datetime, timedelta, timezone

def find_closest_tle_file(start_time_str):
    """
    Finds the TLE file in the 'TLEs/' directory that is closest in date to the start_time.
    TLE filenames are expected in the format 'TLE_YYYYMMDD_hhmmss_*.txt'.
    """
    tle_dir = "TLEs"
    if not os.path.isdir(tle_dir):
        raise FileNotFoundError(f"Error: The TLE directory '{tle_dir}' was not found.")

    tle_pattern = os.path.join(tle_dir, "TLE_*.txt")
    tle_files = glob.glob(tle_pattern)

    if not tle_files:
        raise FileNotFoundError(f"Error: No TLE files found in '{tle_dir}' matching the pattern 'TLE_*.txt'.")

    # Parse the start time, trying multiple formats
    try:
        # Try parsing the CSV format first
        start_time_dt = datetime.strptime(start_time_str, '%Y%m%d:%H:%M:%S.%f')
    except ValueError:
        # Fallback to the command-line format
        try:
            start_time_dt = datetime.strptime(start_time_str, '%Y%m%d_%H%M%S')
        except ValueError:
            raise ValueError(f"Error: Invalid time format for '{start_time_str}'.")

    closest_file = None
    min_time_diff = timedelta.max

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

def find_satellite_passes(observer, tle_file, start_time_str, end_time_str, fov, calsats_file, results, filename):
    """
    Finds satellite passes and appends them to a results list.
    """
    # Unpack the field-of-view
    alt_min, az_min, alt_max, az_max = fov

    # Load the Skyfield timescale and all satellites from the TLE file into a dictionary
    ts = load.timescale()
    all_sats_by_norad = {sat.model.satnum: sat for sat in load.tle_file(tle_file)}
    print(f"Loaded {len(all_sats_by_norad)} total satellites from {tle_file}.")

    # Determine which satellites to check, and get the set of cal sat NORADs for labeling
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



    # Parse the time range from the new format
    try:
        # Try parsing the CSV format first
        start_utc = datetime.strptime(start_time_str, '%Y%m%d:%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        end_utc = datetime.strptime(end_time_str, '%Y%m%d:%H:%M:%S.%f').replace(tzinfo=timezone.utc)
    except ValueError:
        # Fallback to the command-line format
        try:
            start_utc = datetime.strptime(start_time_str, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
            end_utc = datetime.strptime(end_time_str, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Error: Invalid time format for '{start_time_str}' or '{end_time_str}'.")
            return

    start_time = ts.utc(start_utc)
    end_time = ts.utc(end_utc)
    
    time_points = []
    current_time = start_utc
    while current_time <= end_utc:
        time_points.append(current_time)
        current_time += timedelta(seconds=1)

    if not time_points:
        print("Error: The provided time range is invalid or too short.")
        return

    timestamps = ts.from_datetimes(time_points)
    
    print(f"\nSearching for passes between {start_time_str} and {end_time_str}...")
    
    passes_found = False
    
    # Iterate through the selected satellites to find their passes
    for sat in sats_to_check:
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
        
        # Find the indices where the satellite enters or exits the FOV
        change_indices = np.where(np.diff(in_fov_mask))[0]
        
        is_currently_in_view = in_fov_mask[0]
        pass_start_index = 0 if is_currently_in_view else None

        # Loop through the state changes to identify complete passes
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

    args = parser.parse_args()

    # --- Input Validation ---
    if not args.tle_file and not args.tle_auto:
        parser.error("You must specify either --tle-file or --tle-auto.")

    if args.tle_file and args.tle_auto:
        parser.error("You cannot use both --tle-file and --tle-auto.")

    if not args.time_file and not (args.start_time and args.end_time):
        parser.error("You must specify either --time-file or both --start-time and --end-time.")
    
    if args.time_file and (args.start_time or args.end_time):
        parser.error("You cannot use --time-file with --start-time or --end-time.")

    if (args.start_time and not args.end_time) or (not args.start_time and args.end_time):
        parser.error("--start-time and --end-time must be used together.")
    
    if args.cam_id and (args.site or args.fov or args.latitude or args.longitude):
        parser.error("--cam_id cannot be used with --site, --fov, --latitude, or --longitude.")
        
    if not args.cam_id and not args.fov:
        parser.error("You must specify either --cam_id or --fov.")

    # --- Site and Camera Configuration ---
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
        observer = wgs84.latlon(site_info['lat'], site_info['lon'], elevation_m=site_info['elev'])
        print(f"Using Camera ID '{args.cam_id}': Site='{cam_info['site']}', FOV={fov}")
    else:
        fov = args.fov
        if args.latitude is not None and args.longitude is not None:
            observer = wgs84.latlon(args.latitude, args.longitude)
            print(f"Using custom observer location: Latitude={args.latitude}°, Longitude={args.longitude}°")
        else:
            site = args.site if args.site else 'elginfield'
            site_info = SITES[site]
            observer = wgs84.latlon(site_info['lat'], site_info['lon'], elevation_m=site_info['elev'])
            print(f"Using site: '{site}' at Latitude={site_info['lat']}°, Longitude={site_info['lon']}°")


    calsats_file = 'satellites.json' if args.calsats else None
    results = []

    # Load the time ranges from the CSV file
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
        for index, row in time_df.iterrows():
            start_time_str = row['beg_utc']
            end_time_str = row['end_utc']
            filename = row.get('filename', f"Row {index + 1}")

            print(f"\n{'='*60}")
            print(f"Processing Time Range for '{filename}'")
            print(f"Start: {start_time_str}, End: {end_time_str}")
            print(f"{'='*60}\n")

            tle_file_to_use = args.tle_file
            if args.tle_auto:
                try:
                    tle_file_to_use = find_closest_tle_file(start_time_str)
                    print(f"Automatically selected TLE file: {tle_file_to_use}")
                except (FileNotFoundError, ValueError) as e:
                    print(str(e))
                    continue

            find_satellite_passes(
                observer,
                tle_file_to_use,
                start_time_str,
                end_time_str,
                fov,
                calsats_file,
                results,
                filename
            )
    else:
        # Process the single time range from command-line arguments
        tle_file_to_use = args.tle_file
        if args.tle_auto:
            try:
                tle_file_to_use = find_closest_tle_file(args.start_time)
                print(f"Automatically selected TLE file: {tle_file_to_use}")
            except (FileNotFoundError, ValueError) as e:
                print(str(e))
                return

        find_satellite_passes(
            observer,
            tle_file_to_use,
            args.start_time,
            args.end_time,
            fov,
            calsats_file,
            results,
            "command_line_input"
        )

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
