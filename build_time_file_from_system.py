# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Build a time_file CSV from .vid files in /dump.vid/<cam_id>/.

Compatible with Python 2.6+ and Python 3.

Usage:
    python build_time_file_from_system.py --cam-id 01F
    python build_time_file_from_system.py --cam-id 01F --dump-dir /dump.vid --output custom.csv
"""

import csv
import os
import platform
import sys
from datetime import datetime, timedelta
from optparse import OptionParser


def open_csv_for_write(path):
    """Open CSV file for writing with Python 2/3 compatibility."""
    if sys.version_info[0] >= 3:
        return open(path, 'w', newline='', encoding='utf-8')
    return open(path, 'wb')


def build_time_rows(cam_id, dump_vid_dir):
    """
    Build row dictionaries from .vid files for one camera.

    Each row contains:
      - filename
      - beg_utc (YYYYMMDD:HH:MM:SS.%f)
      - end_utc (beg_utc + 10 minutes - 1 second)
    """
    if platform.system() != 'Linux':
        raise OSError('Error: This script is only supported on Linux.')

    if not os.path.isdir(dump_vid_dir):
        raise IOError("Error: Required directory '%s' was not found." % dump_vid_dir)

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

        # Linux often lacks true creation time. Use st_birthtime when available,
        # otherwise fallback to st_ctime.
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

    return rows


def write_time_file(output_path, rows):
    """Write rows to CSV with expected columns."""
    fieldnames = ['filename', 'beg_utc', 'end_utc']

    with open_csv_for_write(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'filename': row.get('filename', ''),
                'beg_utc': row.get('beg_utc', ''),
                'end_utc': row.get('end_utc', '')
            })


def parse_args():
    """Parse command-line options (Python 2.6-compatible)."""
    parser = OptionParser(usage='python %prog --cam-id 01F [options]')
    parser.add_option('--cam-id', dest='cam_id', default=None,
                      help='Camera ID directory under dump dir (e.g., 01F, 01G, 02F, 02G).')
    parser.add_option('--dump-dir', dest='dump_dir', default='/dump.vid',
                      help='Base dump directory (default: /dump.vid).')
    parser.add_option('--output', dest='output', default=None,
                      help='Output CSV path (default: time_file_from_system_<cam-id>.csv).')

    options, _ = parser.parse_args()

    if not options.cam_id:
        parser.error('--cam-id is required.')

    return options


def main():
    options = parse_args()
    output_path = options.output
    if not output_path:
        output_path = 'time_file_from_system_%s.csv' % options.cam_id

    try:
        rows = build_time_rows(options.cam_id, options.dump_dir)
        write_time_file(output_path, rows)
    except (OSError, IOError) as exc:
        print(str(exc))
        return 1

    print('Created time-file: %s (%d entries)' % (output_path, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
