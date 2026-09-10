"""Log loading for the torque tuning tool.

Two input paths:
  * CSV export from VCM Scanner (reliable, downsampled).
  * HPL binary (full frame decoding via reverse-engineered format).

HPL files are decoded directly — no VCM Scanner CSV export needed. The
format uses section groups (SS\\x00\\x00 + "SYNC"/"sync" tag) containing
SC (channel table), MET (metadata), and CDG (compressed data) sections.
CDG blocks are raw DEFLATE compressed and contain timestamped sample
records with per-channel values.

The loader returns a LogData object with named columns exposed as numpy
arrays for the channels the correction logic cares about:
  - rpm
  - load          (calculated load / airmass, g/cyl or normalized)
  - brake_torque  (Engine Brake Torque, N m)
  - desired_torque (Desired Brake Torque, N m)
  - indicated_torque (Engine Indicated Torque Reference, N m - echoes table)
  - throttle, accel_pedal, map, etc. (best-effort, used for steady-state filter)
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# Canonical channel names we look for. HPT channel names vary by vehicle /
# config, so each has a list of fuzzy aliases.
CHANNEL_ALIASES = {
    "rpm": ["Engine RPM", "RPM", "Engine Speed", "Engine Speed (RPM)"],
    "load": ["Air Load", "Calculated Load", "Calculated Load Value", "Load",
             "Air Mass", "Average Air Mass", "Calculated Air Mass",
             "Relative Throttle Position"],
    "brake_torque": ["Engine Brake Torque", "Brake Torque", "Engine Torque",
                     "Actual Torque", "Engine Brake Torque Reference"],
    "desired_torque": ["Desired Brake Torque", "Desired Torque",
                       "Driver Demand Torque", "Requested Torque"],
    "indicated_torque": ["Engine Indicated Torque", "Engine Indicated Torque Reference",
                         "Indicated Torque"],
    "throttle": ["Throttle Angle", "Throttle Position", "Throttle Desired Angle"],
    "accel_pedal": ["Accelerator Pedal Position", "Accel Pedal Position",
                    "Accelerator Pedal"],
    "map": ["Calculated Manifold Absolute Pressure", "Manifold Absolute Pressure",
            "MAP"],
    "spark": ["Timing Advance", "Spark Advance", "Ignition Timing Advance"],
    "commanded_torque": ["Torque Source", "Commanded Torque"],
}


@dataclass
class LogData:
    """Holds parsed log channels."""
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    column_names: list[str] = field(default_factory=list)
    sample_rate_hz: float = 0.0
    source_path: str = ""
    # mapped point weight channels: {mp_label: array}, e.g. {"0": [...], "OP": [...]}
    mp_weights: dict[str, np.ndarray] = field(default_factory=dict)
    # per-sample active mapped point label (e.g. "0", "OP", "14")
    active_mp: Optional[np.ndarray] = None

    def get(self, canonical: str) -> Optional[np.ndarray]:
        """Return the array for a canonical channel name, or None."""
        return self.columns.get(canonical)

    def has(self, canonical: str) -> bool:
        """Return True if the canonical channel is present."""
        return canonical in self.columns

    def available_canonical(self) -> list[str]:
        """Return canonical channel names that are present in this log."""
        return [k for k in CHANNEL_ALIASES if k in self.columns]

    def available_mapped_points(self) -> list[str]:
        """Sorted list of mapped point labels present in the log."""
        if not self.mp_weights:
            return []
        return sorted(self.mp_weights.keys(), key=_mp_sort_key)

    def mapped_point_mask(self, mp_label: str) -> np.ndarray:
        """Boolean mask: True where the given mapped point is the active one."""
        if self.active_mp is None:
            return np.ones(0, dtype=bool)
        return self.active_mp == mp_label


# ---- CSV loader -----------------------------------------------------------

def load_csv(path: str | Path) -> LogData:
    """Load a VCM Scanner CSV export.

    Handles two layouts:
      1. HP Tuners native CSV (banner + [Channel Information] header +
         [Channel Data] section, first column = Offset in seconds).
      2. Plain CSV (first row = header, rest = data).
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        text = f.read()
    # Detect HPT native format by the banner / section markers.
    if "HP Tuners CSV Log File" in text[:200] or "[Channel Information]" in text:
        return _load_hpt_csv(text, str(path))
    return _load_plain_csv(text, str(path))


def _load_hpt_csv(text: str, src: str) -> LogData:
    lines = text.splitlines()
    # find [Channel Information] then the header line (2 lines below: IDs, names)
    try:
        ci = lines.index("[Channel Information]")
    except ValueError as exc:
        raise ValueError("HPT CSV missing [Channel Information] section") from exc
    # line ci+1 = numeric IDs, line ci+2 = channel names, line ci+3 = units
    header = [c.strip() for c in lines[ci + 2].split(",")]
    # find [Channel Data]
    try:
        cd = lines.index("[Channel Data]")
    except ValueError as exc:
        raise ValueError("HPT CSV missing [Channel Data] section") from exc
    data_lines = lines[cd + 1:]
    return _build_log(header, data_lines, src, time_column="Offset")


def _load_plain_csv(text: str, src: str) -> LogData:
    rows = [r for r in text.splitlines() if r.strip()]
    if not rows:
        raise ValueError("CSV is empty")
    header = [c.strip() for c in rows[0].split(",")]
    data_lines = rows[1:]
    return _build_log(header, data_lines, src)


def _build_log(header: list[str], data_lines: list[str], src: str,
               time_column: str | None = None) -> LogData:
    raw: dict[str, list[float]] = {h: [] for h in header}
    for ln in data_lines:
        if not ln.strip():
            continue
        parts = ln.split(",")
        for i, h in enumerate(header):
            val = parts[i].strip() if i < len(parts) else ""
            try:
                raw[h].append(float(val))
            except ValueError:
                raw[h].append(np.nan)
    raw_arr = {h: np.array(v, dtype=float) for h, v in raw.items()}
    log = LogData(column_names=header, source_path=src)
    # map to canonical names
    for canon, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            if alias in raw_arr:
                log.columns[canon] = raw_arr[alias]
                break
        else:
            # fuzzy: case-insensitive contains
            for h in header:
                hl = h.lower()
                if any(a.lower() in hl for a in aliases):
                    log.columns[canon] = raw_arr[h]
                    break
    # sample rate from the time column if present
    if time_column and time_column in raw_arr:
        t = raw_arr[time_column]
        good = t[~np.isnan(t)]
        if len(good) > 1:
            dt = np.diff(good)
            dt = dt[(dt > 0) & (dt < 10)]
            if len(dt):
                log.sample_rate_hz = float(1.0 / np.median(dt))
    # parse mapped point weight channels and compute active MP per sample
    _parse_mapped_points(log, raw_arr)
    return log


def _mp_sort_key(label: str):
    """Sort key: numeric MPs first (0,1,2,...14), then 'OP'."""
    if label == "OP":
        return (1, 0)
    return (0, int(label))


def _parse_mapped_points(log: LogData, raw_arr: dict[str, np.ndarray]) -> None:
    """Find 'Mapped Point X Weight' columns, store them, and compute the
    per-sample active mapped point (highest weight wins)."""
    mp_weights: dict[str, np.ndarray] = {}
    for h in log.column_names:
        hl = h.lower().strip()
        if not (hl.startswith("mapped point") and "weight" in hl):
            continue
        # extract the MP label: "Mapped Point 0 Weight" -> "0", "OP" -> "OP"
        # strip "Mapped Point " prefix and " Weight" suffix
        middle = h[len("Mapped Point "):].rsplit(" Weight", 1)[0].strip()
        if middle in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                       "10", "11", "12", "13", "14", "OP"):
            mp_weights[middle] = raw_arr[h]
    if not mp_weights:
        return
    log.mp_weights = mp_weights
    # stack weights and find argmax per sample
    labels = sorted(mp_weights.keys(), key=_mp_sort_key)
    n = len(next(iter(mp_weights.values())))
    stacked = np.zeros((n, len(labels)))
    for j, lbl in enumerate(labels):
        arr = mp_weights[lbl]
        stacked[:len(arr), j] = arr[:n] if len(arr) >= n else arr
    # handle NaN as -inf so they never win
    stacked = np.where(np.isnan(stacked), -1.0, stacked)
    active_idx = np.argmax(stacked, axis=1)
    label_arr = np.array(labels, dtype=object)
    log.active_mp = label_arr[active_idx]


# ---- HPL loader (full frame decoding) --------------------------------------
#
# The HPL format was reverse-engineered from Johnson Tuning's CSV Log Viewer
# (https://www.johnsontuning.com/shop/p/csv-log-viewer), which opens .hpl files
# natively. The format is:
#
#   File header:  b"HPT " + 2 bytes padding (6 bytes total before first group)
#   Section groups:  b"SS\x00\x00" + 4-byte tag ("SYNC" = current, "sync" = legacy)
#     Group header:  14 bytes for "SYNC", 13 for "sync"
#   Sections within a group (3-byte tag):
#     "SC\x00"  = channel table (stream channels)
#     "MET"     = metadata
#     "CDG"     = compressed data group (raw DEFLATE)
#
# SC channel table:
#   offset 19: channel count (uint16 LE)
#   offset 21: channel entries, each:
#     streamId(2) + channelId(2) + unitId(2) + scale(float64) + offset(float64)
#     + nameLen(1) + name + unitLen(1) + unit
#   valueType = unitId >> 8  (2=u8, 4=u16, 5=i32, 9=f32, 10=f64, 11=enum)
#
# CDG section:
#   offset 4: compressed_size (uint32 LE)
#   offset 8: uncompressed_size (uint32 LE)
#   offset 12: raw DEFLATE compressed data
#
# Decompressed CDG: first 2 bytes = record count (uint16 LE), then records:
#   Each record: timestamp(3 bytes uint24 LE) + channel_count(1 byte)
#   Then for each channel: streamId(1 byte)
#     If streamId & 0x80: skip (channel not present this frame)
#     Else: read value based on valueType
#   Value = raw * scale + offset  (if scale == 1: raw + offset)

# Value type constants
_VT_U8 = 2
_VT_U16 = 4
_VT_I32 = 5
_VT_F32 = 9
_VT_F64 = 10
_VT_ENUM = 11

# Section group header sizes
_GROUP_HEADER_CURRENT = 14
_GROUP_HEADER_LEGACY = 13

# SC section offsets
_SC_CHAN_COUNT_OFF = 19
_SC_HEADER_SIZE = 21
_SC_ENTRY_FIXED = 22  # streamId(2)+channelId(2)+unitId(2)+scale(8)+offset(8)

# CDG section offsets
_CDG_SIZE_OFF = 4
_CDG_HEADER_SIZE = 12


@dataclass
class HplChannel:
    """A channel definition from the HPL SC (stream channel) section."""
    stream_id: int
    channel_id: int
    unit_id: int
    scale: float
    offset: float
    name: str
    unit: str
    value_type: int
    is_enum: bool


@dataclass
class HplParam:
    """Legacy parameter-definition record (kept for backward compatibility)."""
    name: str
    unit: str
    pid: int
    unk: int
    scale: float
    offset: float
    def_off: int


def parse_hpl_params(path: str | Path) -> list[HplParam]:
    """Parse the parameter-definition section of an HPL file.

    This is the legacy flat-list format found in the first SC section.
    Kept for backward compatibility with code that only needs param names.
    """
    channels = _parse_hpl_channels(path)
    params: list[HplParam] = []
    for ch in channels:
        params.append(HplParam(
            name=ch.name, unit=ch.unit, pid=ch.channel_id,
            unk=ch.unit_id, scale=ch.scale, offset=ch.offset,
            def_off=0,
        ))
    return params


def _read_str(data: bytes, off: int, length: int) -> str:
    return data[off:off + length].decode("utf-8", "replace")


def _parse_sc_section(data: bytes, start: int) -> tuple[list[HplChannel], int]:
    """Parse an SC (stream channel) section starting at *start*.

    Returns (channels, end_offset).
    """
    count = struct.unpack_from("<H", data, start + _SC_CHAN_COUNT_OFF)[0]
    if count > 127:
        raise ValueError(f"HPL channel table declares {count} channels (>127)")
    channels: list[HplChannel] = []
    p = start + _SC_HEADER_SIZE
    for _ in range(count):
        stream_id = struct.unpack_from("<H", data, p)[0]
        channel_id = struct.unpack_from("<H", data, p + 2)[0]
        unit_id = struct.unpack_from("<H", data, p + 4)[0]
        scale = struct.unpack_from("<d", data, p + 6)[0]
        offset = struct.unpack_from("<d", data, p + 14)[0]
        name_len = data[p + _SC_ENTRY_FIXED]
        name_off = p + _SC_ENTRY_FIXED + 1
        name = _read_str(data, name_off, name_len)
        unit_len = data[name_off + name_len]
        unit_off = name_off + name_len + 1
        unit = _read_str(data, unit_off, unit_len)
        value_type = unit_id >> 8
        channels.append(HplChannel(
            stream_id=stream_id, channel_id=channel_id, unit_id=unit_id,
            scale=scale, offset=offset, name=name, unit=unit,
            value_type=value_type, is_enum=(value_type == _VT_ENUM),
        ))
        p = unit_off + unit_len
    return channels, p


def _parse_cdg_section(data: bytes, start: int) -> bytes:
    """Parse and decompress a CDG section starting at *start*."""
    comp_size = struct.unpack_from("<I", data, start + _CDG_SIZE_OFF)[0]
    uncomp_size = struct.unpack_from("<I", data, start + 8)[0]
    comp = data[start + _CDG_HEADER_SIZE:start + _CDG_HEADER_SIZE + comp_size]
    raw = zlib.decompress(comp, -15)
    if len(raw) != uncomp_size:
        raise ValueError(f"HPL CDG decompressed {len(raw)} bytes, expected {uncomp_size}")
    return raw


def _detect_group_type(data: bytes, off: int) -> str | None:
    """Check if *off* starts a section group (SS\\x00\\x00 + tag).

    Returns "current", "legacy", or None.
    """
    if off + 8 > len(data):
        return None
    if data[off:off + 2] != b"SS" or data[off + 2] != 0 or data[off + 3] != 0:
        return None
    tag = data[off + 4:off + 8]
    if tag == b"SYNC":
        return "current"
    if tag == b"sync":
        return "legacy"
    return None


def _read_tag3(data: bytes, off: int) -> str:
    """Read a 3-character section tag."""
    return data[off:off + 3].decode("ascii", "replace")


def _parse_hpl_channels(path: str | Path) -> list[HplChannel]:
    """Parse channel definitions from an HPL file's SC sections."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"HPT ":
        raise ValueError("not an HPL file (bad magic)")
    channels: list[HplChannel] | None = None
    off = 6  # first section group starts at offset 6
    while off < len(data):
        gtype = _detect_group_type(data, off)
        if gtype is None:
            break
        header_size = _GROUP_HEADER_CURRENT if gtype == "current" else _GROUP_HEADER_LEGACY
        sec = off + header_size
        # Parse sections within this group
        while sec < len(data) and _detect_group_type(data, sec) is None:
            tag = _read_tag3(data, sec)
            if tag.startswith("SC") and data[sec + 2] == 0:
                channels, sec = _parse_sc_section(data, sec)
            elif tag == "MET":
                # Skip metadata section
                comp_size = struct.unpack_from("<I", data, sec + 11)[0]
                sec = sec + 19 + comp_size
            elif tag == "CDG":
                # Skip CDG section (we only want channels here)
                comp_size = struct.unpack_from("<I", data, sec + _CDG_SIZE_OFF)[0]
                sec = sec + _CDG_HEADER_SIZE + comp_size
            else:
                break
        off = sec
    if channels is None:
        raise ValueError("HPL file has no channel table (SC section)")
    return channels


def _decode_cdg_records(frame: bytes, channels: list[HplChannel],
                        stream_to_idx: dict[int, int]) -> tuple[list[int], dict[int, list]]:
    """Decode one decompressed CDG block.

    Returns (tick_list, {channel_index: value_list}).
    """
    declared = struct.unpack_from("<H", frame, 0)[0]
    body = frame[2:]
    ticks: list[int] = []
    values: dict[int, list] = {i: [] for i in range(len(channels))}
    w = 0
    for _rec in range(declared) if declared > 0 else iter(int, 0):
        if w + 4 > len(body):
            break
        tick = body[w] | (body[w + 1] << 8) | (body[w + 2] << 16)
        n_chans = body[w + 3]
        w += 4
        ticks.append(tick)
        for _ in range(n_chans):
            if w >= len(body):
                break
            sid_byte = body[w]
            w += 1
            idx = stream_to_idx.get(sid_byte & 0x7F)
            if idx is None:
                # Unknown stream — skip its value
                continue
            if sid_byte & 0x80:
                # Channel not present this frame
                values[idx].append(None)
                continue
            ch = channels[idx]
            if ch.is_enum:
                slen = body[w]
                sval = body[w + 1:w + 1 + slen].decode("utf-8", "replace")
                w += 1 + slen
                values[idx].append(sval)
            else:
                vt = ch.value_type
                if vt == _VT_U8:
                    raw = body[w]
                    w += 1
                elif vt == _VT_U16:
                    raw = struct.unpack_from("<H", body, w)[0]
                    w += 2
                elif vt == _VT_I32:
                    raw = struct.unpack_from("<i", body, w)[0]
                    w += 4
                elif vt == _VT_F32:
                    raw = struct.unpack_from("<f", body, w)[0]
                    w += 4
                elif vt == _VT_F64:
                    raw = struct.unpack_from("<d", body, w)[0]
                    w += 8
                else:
                    raise ValueError(f"Unknown value type 0x{vt:x} for channel {ch.name}")
                if ch.scale == 1.0:
                    val = raw + ch.offset
                else:
                    val = raw / ch.scale + ch.offset
                values[idx].append(val)
    return ticks, values


def load_hpl(path: str | Path) -> LogData:
    """Load an HPL file with full frame-data decoding.

    Parses channel definitions from SC sections, decompresses CDG data blocks
    (raw DEFLATE), and decodes all sample frames into channel arrays.
    Returns a LogData with decoded columns ready for correction.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"HPT ":
        raise ValueError("not an HPL file (bad magic)")

    # Phase 1: parse all SC sections and CDG blocks
    channels: list[HplChannel] | None = None
    cdg_blocks: list[bytes] = []
    off = 6
    while off < len(data):
        gtype = _detect_group_type(data, off)
        if gtype is None:
            break
        header_size = _GROUP_HEADER_CURRENT if gtype == "current" else _GROUP_HEADER_LEGACY
        sec = off + header_size
        while sec < len(data) and _detect_group_type(data, sec) is None:
            tag = _read_tag3(data, sec)
            if tag.startswith("SC") and data[sec + 2] == 0:
                channels, sec = _parse_sc_section(data, sec)
            elif tag == "MET":
                comp_size = struct.unpack_from("<I", data, sec + 11)[0]
                sec = sec + 19 + comp_size
            elif tag == "CDG":
                raw = _parse_cdg_section(data, sec)
                cdg_blocks.append(raw)
                comp_size = struct.unpack_from("<I", data, sec + _CDG_SIZE_OFF)[0]
                sec = sec + _CDG_HEADER_SIZE + comp_size
            else:
                break
        off = sec

    if channels is None:
        raise ValueError("HPL file has no channel table (SC section)")
    if not cdg_blocks:
        raise ValueError("HPL file has no compressed data (CDG) sections")

    # Phase 2: build stream-id → channel-index map
    stream_to_idx: dict[int, int] = {}
    for i, ch in enumerate(channels):
        stream_to_idx[ch.stream_id] = i

    # Phase 3: decode all CDG blocks
    all_ticks: list[int] = []
    all_values: dict[int, list] = {i: [] for i in range(len(channels))}
    for block in cdg_blocks:
        ticks, vals = _decode_cdg_records(block, channels, stream_to_idx)
        all_ticks.extend(ticks)
        for idx, vlist in vals.items():
            all_values[idx].extend(vlist)

    if len(all_ticks) < 2:
        raise ValueError("HPL file decoded fewer than 2 samples")

    # Phase 4: build unique sorted tick index and align channels
    unique_ticks = sorted(set(all_ticks))
    tick_to_idx = {t: i for i, t in enumerate(unique_ticks)}
    n = len(unique_ticks)

    # Build time axis in seconds
    time_arr = np.array([t / 1000.0 for t in unique_ticks], dtype=float)

    # Build column arrays with forward-fill for missing values
    col_names = ["Time (s)"] + [ch.name for ch in channels]
    raw_arrays: dict[str, np.ndarray] = {}

    for idx, ch in enumerate(channels):
        vals = all_values[idx]
        arr = np.full(n, np.nan, dtype=float)
        # Track which entries are strings (enums)
        is_string_col = any(isinstance(v, str) for v in vals[:100])
        if is_string_col:
            # For enum/string channels, store as object array
            str_arr = np.full(n, None, dtype=object)
            for i_tick, v in zip(all_ticks, vals):
                if v is not None:
                    str_arr[tick_to_idx[i_tick]] = v
            # Forward-fill strings
            last = None
            for i in range(n):
                if str_arr[i] is not None:
                    last = str_arr[i]
                else:
                    str_arr[i] = last
            raw_arrays[ch.name] = str_arr
        else:
            for i_tick, v in zip(all_ticks, vals):
                if v is not None and not isinstance(v, str):
                    arr[tick_to_idx[i_tick]] = v
            # Forward-fill numeric NaNs
            for i in range(1, n):
                if np.isnan(arr[i]):
                    arr[i] = arr[i - 1]
            raw_arrays[ch.name] = arr

    # Phase 5: build LogData with canonical channel mapping
    log = LogData(column_names=col_names, source_path=str(path))
    log.columns["Time (s)"] = time_arr

    # Map raw channels to canonical names using the alias system
    for canon, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            if alias in raw_arrays:
                arr = raw_arrays[alias]
                if arr.dtype == object:
                    continue  # skip enum channels for numeric canonicals
                log.columns[canon] = arr
                break
        else:
            # fuzzy: case-insensitive contains
            for name, arr in raw_arrays.items():
                if arr.dtype == object:
                    continue
                nl = name.lower()
                if any(a.lower() in nl for a in aliases):
                    log.columns[canon] = arr
                    break

    # Store all raw arrays too (for channels not in canonical set)
    for name, arr in raw_arrays.items():
        if name not in log.columns and arr.dtype != object:
            log.columns[name] = arr

    # Sample rate from time axis
    if len(time_arr) > 1:
        dt = np.diff(time_arr)
        dt = dt[(dt > 0) & (dt < 10)]
        if len(dt):
            log.sample_rate_hz = float(1.0 / np.median(dt))

    # Parse mapped-point weight channels
    _parse_mapped_points(log, raw_arrays)
    return log


# ---- WOT max extraction ----------------------------------------------------

@dataclass
class WotMaxCurve:
    """Engine max torque vs RPM, extracted from a WOT log pull.

    rpm_bins:  RPM breakpoints (sorted, unique)
    torque:    max brake torque observed at each RPM bin (N m)
    n_samples: number of WOT samples in each bin (confidence indicator)
    source:    description of where the data came from
    """
    rpm_bins: np.ndarray
    torque: np.ndarray
    n_samples: np.ndarray
    source: str = ""

    def torque_at_rpm(self, rpm: float | np.ndarray) -> float | np.ndarray:
        """Interpolate max torque at a given RPM (clamped to axis range)."""
        return np.interp(rpm, self.rpm_bins, self.torque,
                         left=self.torque[0], right=self.torque[-1])


def extract_wot_max(
    log: LogData,
    wot_pedal_threshold: float = 80.0,
    min_rpm: float = 1000.0,
    rpm_bin_size: float = 250.0,
    min_samples_per_bin: int = 2,
) -> WotMaxCurve:
    """Extract the engine's max torque curve from a WOT log pull.

    Filters log samples for WOT conditions (high pedal + high RPM),
    bins them by RPM, and takes the max brake torque in each bin.

    Parameters:
      wot_pedal_threshold: accelerator pedal % above which counts as WOT.
        HPTuners logs pedal as 0-100%. Stock Coyote pedal reaches ~99% at
        WOT, but reported values can be truncated, so 80% is a safe default
        that captures the WOT plateau without including part-throttle.
        If the log has no pedal channel, falls back to throttle position
        (typically ~88% at WOT on Coyote).
      min_rpm: ignore samples below this RPM (idle, revving in neutral).
      rpm_bin_size: width of each RPM bin in N m (smaller = more resolution
        but needs more data).
      min_samples_per_bin: bins with fewer samples are dropped (unreliable).

    Returns a WotMaxCurve with the max torque at each RPM bin.

    Requires either brake_torque or (load + an indicated torque table
    to convert load to torque). If brake_torque is available, uses it
    directly. Otherwise, if load is available, uses load as a proxy
    (higher load = more torque) and the caller can convert via the
    indicated table.
    """
    rpm = log.get("rpm")
    if rpm is None:
        raise ValueError("Log has no RPM channel")

    # Determine WOT mask: use accel_pedal, fall back to throttle
    pedal = log.get("accel_pedal")
    throttle = log.get("throttle")
    if pedal is not None:
        wot_mask = pedal >= wot_pedal_threshold
    elif throttle is not None:
        wot_mask = throttle >= wot_pedal_threshold
    else:
        raise ValueError(
            "Log has no accelerator pedal or throttle channel for WOT filter")

    # Filter: WOT + above min RPM + valid (non-NaN) RPM
    valid = wot_mask & (rpm >= min_rpm) & ~np.isnan(rpm)

    # Get torque channel
    torque = log.get("brake_torque")
    load = log.get("load")
    if torque is not None:
        torque_vals = torque
        val_label = "brake torque"
    elif load is not None:
        torque_vals = load
        val_label = "load (g/cyl)"
    else:
        raise ValueError(
            "Log has no brake torque or load channel for max extraction")

    valid = valid & ~np.isnan(torque_vals)

    rpm_filt = rpm[valid]
    tq_filt = torque_vals[valid]

    if len(rpm_filt) < 10:
        raise ValueError(
            f"Only {len(rpm_filt)} WOT samples found "
            f"(need at least 10 for a reliable max curve)")

    # Bin by RPM and take max in each bin
    rpm_min = np.floor(rpm_filt.min() / rpm_bin_size) * rpm_bin_size
    rpm_max = np.ceil(rpm_filt.max() / rpm_bin_size) * rpm_bin_size
    bin_edges = np.arange(rpm_min, rpm_max + rpm_bin_size, rpm_bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    n_bins = len(bin_centers)
    max_torque = np.full(n_bins, np.nan)
    n_samples = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (rpm_filt >= lo) & (rpm_filt < hi)
        n = int(np.sum(mask))
        n_samples[i] = n
        if n >= min_samples_per_bin:
            max_torque[i] = float(np.max(tq_filt[mask]))

    # Drop bins with insufficient data
    good = ~np.isnan(max_torque)
    if np.sum(good) < 3:
        raise ValueError(
            f"Only {int(np.sum(good))} RPM bins have enough WOT samples "
            f"(need at least 3)")

    return WotMaxCurve(
        rpm_bins=bin_centers[good],
        torque=max_torque[good],
        n_samples=n_samples[good],
        source=f"WOT log ({val_label}, {int(np.sum(valid))} samples, "
               f"{wot_pedal_threshold:.0f}% pedal threshold)",
    )
