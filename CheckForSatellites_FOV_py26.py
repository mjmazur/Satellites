# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Python 2.6-compatible variant of CheckForSatellites_FOV.py.

Notes:
- Uses optparse instead of argparse.
- Uses multiprocessing.Pool instead of concurrent.futures.
- Uses naive UTC datetimes (no datetime.timezone dependency).
- Preserves core behavior: per-day segmentation, closest-TLE selection, optional
  system time-file generation from /dump.vid/<cam_id>/, and CSV output.
"""

import csv
import glob
import json
import os
import platform
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from optparse import OptionParser

import numpy as np
from skyfield.api import load, wgs84


def parse_time_utc(time_value):
    """
    Parse a UTC time value into a naive UTC datetime.
    Accepts datetime or string in either supported format.
    """
    if isinstance(time_value, datetime):
        return time_value

    value_str = str(time_value)
    formats = ['%Y%m%d:%H:%M:%S.%f', '%Y%m%d_%H%M%S']
    for time_format in formats:
        try:
            return datetime.strptime(value_str, time_format)
        except ValueError:
            pass

    raise ValueError("Error: Invalid time format for '%s'." % value_str)


def split_time_range_by_day(start_utc, end_utc):
    """Split a UTC datetime range into inclusive day-bounded segments."""
    if end_utc < start_utc:
        return []

    segments = []
    current_start = start_utc
    while current_start <= end_utc:
        next_day_midnight = datetime(
            current_start.year,
            current_start.month,
            current_start.day
        ) + timedelta(days=1)
        current_end = min(end_utc, next_day_midnight - timedelta(seconds=1))
        segments.append((current_start, current_end))
        current_start = next_day_midnight

    return segments


def find_closest_tle_file(start_time):
    """
    Find the TLE file in 'TLEs/' closest to start_time.
    Expected pattern: TLE_YYYYMMDD_hhmmss_*.txt
    """
    tle_dir = 'TLEs'
    if not os.path.isdir(tle_dir):
        raise IOError("Error: The TLE directory '%s' was not found." % tle_dir)

    tle_pattern = os.path.join(tle_dir, 'TLE_*.txt')
    tle_files = glob.glob(tle_pattern)
    if not tle_files:
        raise IOError("Error: No TLE files found in '%s' matching 'TLE_*.txt'." % tle_dir)

    start_time_dt = parse_time_utc(start_time)

    closest_file = None
    min_time_diff = timedelta.max

    for tle_file in tle_files:
        filename = os.path.basename(tle_file)
        try:
            parts = filename.split('_')
            timestamp_str = '%s_%s' % (parts[1], parts[2])
            tle_time_dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            time_diff = abs(start_time_dt - tle_time_dt)

            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_file = tle_file
        except (IndexError, ValueError):
            print("Warning: Could not parse timestamp from filename '%s'. Skipping." % filename)

    if closest_file is None:
        raise IOError('Error: Could not determine the closest TLE file.')

    return closest_file


def create_observer(observer_config):
    """Create a Skyfield observer from a serializable config dict."""
    return wgs84.latlon(
        observer_config['lat'],
        observer_config['lon'],
        elevation_m=observer_config['elev']
    )


def generate_time_file_from_system(cam_id):
    """
    Build a CSV time-file from .vid files in /dump.vid/<cam_id>/ on Linux.

    Each .vid contributes one range:
      begin = file creation time (or st_ctime fallback)
      end   = begin + 10 minutes - 1 second
    """
    if platform.system() != 'Linux':
        raise OSError('Error: --time-from-system is only supported on Linux.')

    dump_vid_dir = '/dump.vid'
    if not os.path.isdir(dump_vid_dir):
        raise IOError("Error: Required directory '/dump.vid' was not found.")

    cam_dir = os.path.join(dump_vid_dir, cam_id)
    if not os.path.isdir(cam_dir):
        raise IOError("Error: Camera directory '%s' was not found." % cam_dir)

    vid_files = []
    for root, _, files in os.walk(cam_dir):
        for file_name in files:
            if file_name.lower().endswith('.vid'):
                vid_files.append(os.path.join(root, file_name))

    if not vid_files:
        raise IOError("Error: No .vid files found under '%s'." % cam_dir)

    rows = []
    warned_about_linux_ctime = False

    for vid_path in vid_files:
        stat_info = os.stat(vid_path)

        if hasattr(stat_info, 'st_birthtime'):
            begin_epoch = stat_info.st_birthtime
        else:
            begin_epoch = stat_info.st_ctime
            if not warned_about_linux_ctime:
                print('Warning: Linux may not expose true creation time; using st_ctime as begin time.')
                warned_about_linux_ctime = True

        begin_utc = datetime.utcfromtimestamp(begin_epoch)
        end_utc = begin_utc + timedelta(minutes=10) - timedelta(seconds=1)

        rows.append({
            'filename': os.path.basename(vid_path),
            'beg_utc': begin_utc.strftime('%Y%m%d:%H:%M:%S.%f'),
            'end_utc': end_utc.strftime('%Y%m%d:%H:%M:%S.%f'),
            '_sort_key': begin_utc
        })

    rows.sort(key=lambda row: row['_sort_key'])
    for row in rows:
        if '_sort_key' in row:
            del row['_sort_key']

    output_time_file = 'time_file_from_system_%s.csv' % cam_id
    write_time_file(output_time_file, rows)
    print('Created time-file from system data: %s (%d entries)' % (output_time_file, len(rows)))

    return output_time_file


def read_time_file(time_file):
    """Read a CSV time-file into a list of dict rows."""
    if not os.path.isfile(time_file):
        raise IOError("Error: Time file not found at '%s'." % time_file)

    with open_csv_for_read(time_file) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return []

    if 'beg_utc' not in rows[0] or 'end_utc' not in rows[0]:
        raise ValueError("Error: The time file must contain 'beg_utc' and 'end_utc' columns.")

    return rows


def write_time_file(path, rows):
    """Write time rows (filename, beg_utc, end_utc) to CSV."""
    fieldnames = ['filename', 'beg_utc', 'end_utc']
    with open_csv_for_write(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'filename': row.get('filename', ''),
                'beg_utc': row.get('beg_utc', ''),
                'end_utc': row.get('end_utc', '')
            })


def open_csv_for_read(path):
    """Open a CSV file for reading in a Python 2/3 compatible way."""
    if sys.version_info[0] >= 3:
        return open(path, 'r', newline='', encoding='utf-8')
    return open(path, 'rb')


def open_csv_for_write(path):
    """Open a CSV file for writing in a Python 2/3 compatible way."""
    if sys.version_info[0] >= 3:
        return open(path, 'w', newline='', encoding='utf-8')
    return open(path, 'wb')


def find_satellite_passes(observer, tle_file, start_utc, end_utc, fov, calsats_file, filename):
    """Find satellite passes for a single segment and return rows."""
    alt_min, az_min, alt_max, az_max = fov

    ts = load.timescale()
    all_sats_by_norad = {}
    for sat in load.tle_file(tle_file):
        all_sats_by_norad[sat.model.satnum] = sat

    print('Loaded %d total satellites from %s.' % (len(all_sats_by_norad), tle_file))

    sats_to_check = []
    if calsats_file:
        try:
            with open(calsats_file, 'rb') as handle:
                cal_sats_info = json.load(handle)

            for cal_sat in cal_sats_info:
                norad_id_str = cal_sat.get('norad')
                if not norad_id_str:
                    continue
                try:
                    norad_id = int(norad_id_str)
                    if norad_id in all_sats_by_norad:
                        sats_to_check.append(all_sats_by_norad[norad_id])
                    else:
                        print("Warning: Calibration satellite '%s' (NORAD %d) not found in TLE file." % (
                            cal_sat.get('name'), norad_id
                        ))
                except (ValueError, TypeError):
                    print("Warning: Invalid NORAD ID '%s' in %s. It must be an integer." % (
                        norad_id_str, calsats_file
                    ))

            print('Checking %d calibration satellites specified in %s.' % (len(sats_to_check), calsats_file))

        except IOError:
            print("Error: Calibration satellite file not found at '%s'. Exiting segment." % calsats_file)
            return []
        except (ValueError, KeyError):
            print("Error: Could not parse calibration satellite file '%s'. Exiting segment." % calsats_file)
            return []
    else:
        sats_to_check = list(all_sats_by_norad.values())

    time_points = []
    current_time = start_utc
    while current_time <= end_utc:
        time_points.append(current_time)
        current_time += timedelta(seconds=1)

    if not time_points:
        print('Error: The provided time range is invalid or too short.')
        return []

    timestamps = ts.from_datetimes(time_points)

    print('\nSearching for passes between %s and %s...' % (
        start_utc.isoformat(), end_utc.isoformat()
    ))

    passes_found = False
    results = []

    for sat in sats_to_check:
        difference = sat - observer
        topocentric = difference.at(timestamps)

        alt, az, _ = topocentric.altaz()

        if az_min <= az_max:
            az_mask = (az.degrees >= az_min) & (az.degrees <= az_max)
        else:
            az_mask = (az.degrees >= az_min) | (az.degrees <= az_max)

        in_fov_mask = (
            (alt.degrees >= alt_min) & (alt.degrees <= alt_max) &
            az_mask
        )

        change_indices = np.where(np.diff(in_fov_mask))[0]

        is_currently_in_view = bool(in_fov_mask[0])
        pass_start_index = 0 if is_currently_in_view else None

        for idx in change_indices:
            if is_currently_in_view:
                entry_idx = pass_start_index
                exit_idx = idx
                results.append({
                    'filename': filename,
                    'satellite_name': sat.name,
                    'satellite_norad_id': sat.model.satnum,
                    'time_enters': timestamps[entry_idx].utc_iso(),
                    'time_leaves': timestamps[exit_idx].utc_iso(),
                    'azimuth_enters': '%.2f' % az.degrees[entry_idx],
                    'altitude_enters': '%.2f' % alt.degrees[entry_idx],
                    'azimuth_leaves': '%.2f' % az.degrees[exit_idx],
                    'altitude_leaves': '%.2f' % alt.degrees[exit_idx]
                })
                passes_found = True
                is_currently_in_view = False
                pass_start_index = None
            else:
                is_currently_in_view = True
                pass_start_index = idx + 1

        if is_currently_in_view:
            entry_idx = pass_start_index
            results.append({
                'filename': filename,
                'satellite_name': sat.name,
                'satellite_norad_id': sat.model.satnum,
                'time_enters': timestamps[entry_idx].utc_iso(),
                'time_leaves': 'Still in view',
                'azimuth_enters': '%.2f' % az.degrees[entry_idx],
                'altitude_enters': '%.2f' % alt.degrees[entry_idx],
                'azimuth_leaves': 'N/A',
                'altitude_leaves': 'N/A'
            })
            passes_found = True

    if not passes_found:
        print("No satellite passes found for '%s'." % filename)

    return results


def process_time_segment(task):
    """Worker function for multiprocessing."""
    observer = create_observer(task['observer_config'])
    tle_file_to_use = task.get('resolved_tle_file', task.get('tle_file'))

    if task.get('tle_auto') and tle_file_to_use is None:
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


def parse_fov_option(fov_option):
    """Parse --fov option into [alt_min, az_min, alt_max, az_max]."""
    if fov_option is None:
        return None

    if isinstance(fov_option, tuple) or isinstance(fov_option, list):
        values = fov_option
    else:
        values = str(fov_option).replace(',', ' ').split()

    if len(values) != 4:
        raise ValueError('--fov requires four numeric values.')

    try:
        return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
    except ValueError:
        raise ValueError('--fov values must be numeric.')


def build_parser():
    """Create and return the option parser."""
    parser = OptionParser(usage='python %prog [options]')

    parser.add_option('--tle-file', dest='tle_file', default=None,
                      help='Path to the TLE file containing satellite data.')
    parser.add_option('--tle-auto', dest='tle_auto', action='store_true', default=False,
                      help="Automatically select the TLE file from the 'TLEs/' directory.")

    parser.add_option('--time-file', dest='time_file', default=None,
                      help='Path to CSV containing beg_utc/end_utc columns.')
    parser.add_option('--time-from-system', dest='time_from_system', action='store_true', default=False,
                      help='Build a time-file from /dump.vid/<cam_id>/ .vid files (Linux only).')
    parser.add_option('--start-time', dest='start_time', default=None,
                      help='Start time in YYYYMMDD_hhmmss format.')
    parser.add_option('--end-time', dest='end_time', default=None,
                      help='End time in YYYYMMDD_hhmmss format.')

    parser.add_option('--cam_id', dest='cam_id', default=None,
                      help='Camera ID: 01F, 01G, 02F, or 02G')
    parser.add_option('--fov', dest='fov', nargs=4, type='float', default=None,
                      help='Field-of-view ALT_MIN AZ_MIN ALT_MAX AZ_MAX')

    parser.add_option('--site', dest='site', default=None,
                      help='Observation site: tavistock or elginfield')
    parser.add_option('--latitude', dest='latitude', type='float', default=None,
                      help='Custom observer latitude in degrees.')
    parser.add_option('--longitude', dest='longitude', type='float', default=None,
                      help='Custom observer longitude in degrees.')

    parser.add_option('--calsats', dest='calsats', action='store_true', default=False,
                      help="Check only calibration satellites from satellites.json.")
    parser.add_option('--output-file', dest='output_file', default='calsat_matches.csv',
                      help='Path to output CSV (default: calsat_matches.csv).')
    parser.add_option('--workers', dest='workers', type='int', default=max(1, cpu_count() - 1),
                      help='Number of worker processes (default: CPU count - 1).')

    return parser


def validate_options(options, parser):
    """Validate CLI options and enforce compatibility rules."""
    valid_cam_ids = ['01F', '01G', '02F', '02G']
    valid_sites = ['tavistock', 'elginfield']

    if not options.tle_file and not options.tle_auto:
        parser.error('You must specify either --tle-file or --tle-auto.')

    if options.tle_file and options.tle_auto:
        parser.error('You cannot use both --tle-file and --tle-auto.')

    if options.time_from_system and (options.time_file or options.start_time or options.end_time):
        parser.error('--time-from-system cannot be used with --time-file, --start-time, or --end-time.')

    if (not options.time_from_system and
            not options.time_file and
            not (options.start_time and options.end_time)):
        parser.error('You must specify one of: --time-from-system, --time-file, or both --start-time and --end-time.')

    if options.time_file and (options.start_time or options.end_time):
        parser.error('You cannot use --time-file with --start-time or --end-time.')

    if (options.start_time and not options.end_time) or (options.end_time and not options.start_time):
        parser.error('--start-time and --end-time must be used together.')

    if options.cam_id and (options.site or options.fov or options.latitude is not None or options.longitude is not None):
        parser.error('--cam_id cannot be used with --site, --fov, --latitude, or --longitude.')

    if options.cam_id is None and options.fov is None:
        parser.error('You must specify either --cam_id or --fov.')

    if options.cam_id and options.cam_id not in valid_cam_ids:
        parser.error('--cam_id must be one of: %s' % ', '.join(valid_cam_ids))

    if options.site and options.site not in valid_sites:
        parser.error('--site must be one of: %s' % ', '.join(valid_sites))

    if options.time_from_system and not options.cam_id:
        parser.error('--time-from-system requires --cam_id to identify /dump.vid/<cam_id>/.')

    if options.workers < 1:
        parser.error('--workers must be at least 1.')


def write_results_csv(results, output_filename):
    """Write final results rows to CSV."""
    fieldnames = [
        'filename',
        'satellite_name',
        'satellite_norad_id',
        'time_enters',
        'time_leaves',
        'azimuth_enters',
        'altitude_enters',
        'azimuth_leaves',
        'altitude_leaves'
    ]

    with open_csv_for_write(output_filename) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    """Entry point."""
    parser = build_parser()
    options, _ = parser.parse_args()
    validate_options(options, parser)

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

    if options.cam_id:
        cam_info = CAMERAS[options.cam_id]
        site_info = SITES[cam_info['site']]
        fov = cam_info['fov']
        observer_config = {'lat': site_info['lat'], 'lon': site_info['lon'], 'elev': site_info['elev']}
        print("Using Camera ID '%s': Site='%s', FOV=%s" % (options.cam_id, cam_info['site'], fov))
    else:
        fov = parse_fov_option(options.fov)
        if options.latitude is not None and options.longitude is not None:
            observer_config = {'lat': options.latitude, 'lon': options.longitude, 'elev': 0}
            print('Using custom observer location: Latitude=%s, Longitude=%s' % (
                options.latitude, options.longitude
            ))
        else:
            site = options.site if options.site else 'elginfield'
            site_info = SITES[site]
            observer_config = {'lat': site_info['lat'], 'lon': site_info['lon'], 'elev': site_info['elev']}
            print("Using site: '%s' at Latitude=%s, Longitude=%s" % (
                site, site_info['lat'], site_info['lon']
            ))

    calsats_file = 'satellites.json' if options.calsats else None

    if options.time_from_system:
        try:
            options.time_file = generate_time_file_from_system(options.cam_id)
        except (OSError, IOError) as exc:
            print(str(exc))
            return

    tasks = []

    if options.time_file:
        try:
            time_rows = read_time_file(options.time_file)
        except (IOError, ValueError) as exc:
            print(str(exc))
            return

        for row_num, row in enumerate(time_rows):
            start_time_str = row.get('beg_utc')
            end_time_str = row.get('end_utc')
            filename = row.get('filename', 'Row %d' % (row_num + 1))

            print('\n' + '=' * 60)
            print("Processing Time Range for '%s'" % filename)
            print('Start: %s, End: %s' % (start_time_str, end_time_str))
            print('=' * 60 + '\n')

            try:
                start_utc = parse_time_utc(start_time_str)
                end_utc = parse_time_utc(end_time_str)
            except ValueError as exc:
                print(str(exc))
                continue

            if end_utc < start_utc:
                print("Error: End time is before start time for '%s'. Skipping." % filename)
                continue

            for segment_start, segment_end in split_time_range_by_day(start_utc, end_utc):
                tasks.append({
                    'observer_config': observer_config,
                    'tle_file': options.tle_file,
                    'tle_auto': options.tle_auto,
                    'start_utc': segment_start,
                    'end_utc': segment_end,
                    'fov': fov,
                    'calsats_file': calsats_file,
                    'filename': filename
                })
    else:
        try:
            start_utc = parse_time_utc(options.start_time)
            end_utc = parse_time_utc(options.end_time)
        except ValueError as exc:
            print(str(exc))
            return

        if end_utc < start_utc:
            print('Error: End time is before start time.')
            return

        for segment_start, segment_end in split_time_range_by_day(start_utc, end_utc):
            tasks.append({
                'observer_config': observer_config,
                'tle_file': options.tle_file,
                'tle_auto': options.tle_auto,
                'start_utc': segment_start,
                'end_utc': segment_end,
                'fov': fov,
                'calsats_file': calsats_file,
                'filename': 'command_line_input'
            })

    if not tasks:
        print('No valid time segments to process.')
        return

    if options.tle_auto:
        print('Selecting closest TLE file per day-segment for %d segment(s).' % len(tasks))
        valid_tasks = []
        for i, task in enumerate(tasks):
            try:
                resolved_tle = find_closest_tle_file(task['start_utc'])
                task['resolved_tle_file'] = resolved_tle
                print('  [%d/%d] %s | %s -> %s | TLE: %s' % (
                    i + 1,
                    len(tasks),
                    task['filename'],
                    task['start_utc'].isoformat(),
                    task['end_utc'].isoformat(),
                    resolved_tle
                ))
                valid_tasks.append(task)
            except (IOError, ValueError) as exc:
                print(str(exc))
                print("Skipping segment for '%s' due to TLE selection error." % task['filename'])
        tasks = valid_tasks

        if not tasks:
            print('No valid time segments to process after TLE selection.')
            return

    results = []
    worker_count = min(options.workers, len(tasks))

    if worker_count > 1:
        print('Processing %d segment(s) using %d worker processes.' % (len(tasks), worker_count))
        pool = Pool(processes=worker_count)
        try:
            segment_results = pool.map(process_time_segment, tasks)
            for row_set in segment_results:
                results.extend(row_set)
        except Exception as exc:
            print('Error during multiprocessing execution: %s' % str(exc))
        finally:
            pool.close()
            pool.join()
    else:
        print('Processing %d segment(s) sequentially.' % len(tasks))
        for task in tasks:
            try:
                results.extend(process_time_segment(task))
            except (IOError, ValueError) as exc:
                print(str(exc))

    output_filename = options.output_file
    if options.cam_id and options.output_file == 'calsat_matches.csv':
        output_filename = 'calsats_matched_%s.csv' % options.cam_id

    if results:
        write_results_csv(results, output_filename)
        print("\nSaved %d matches to '%s'." % (len(results), output_filename))
    else:
        print('\nNo satellite passes were found to save.')


if __name__ == '__main__':
    main()
