function rec = mea_read_brw(filepath, varargin)
% MEA_READ_BRW  Read metadata and optionally signal data from a BW4 .brw file.
%
% SYNTAX
%   rec = mea_read_brw(filepath)
%   rec = mea_read_brw(filepath, 'Channels', idx, 'Start', s, 'End', e)
%
% REQUIRED INPUT
%   filepath   char — path to the .brw / .h5 / .hdf5 file
%
% OPTIONAL NAME-VALUE PAIRS
%   'Channels'   integer vector of flat channel indices (0-based) to load.
%                If omitted, no signal data is loaded.
%   'Start'      first sample index (0-based, inclusive).  Default = 0.
%   'End'        last  sample index (0-based, exclusive).  Default = NRecFrames.
%
% OUTPUT  rec — struct with fields:
%   filepath        char
%   n_channels      double   number of active electrodes
%   n_frames        double   total time samples in the file
%   sampling_rate   double   Hz  (NaN if unreadable)
%   rows            n_channels×1 double  electrode row indices
%   cols            n_channels×1 double  electrode col indices
%   layout          char     '1D' or '2D'
%   traces          n_samples × n_req_channels double  (only if Channels given)
%   channel_indices 1×n_req_channels double            (only if Channels given)
%
% NOTES
%   Channel indexing is 0-based to match the Python mea_io convention.
%   Internally MATLAB uses 1-based indexing; the conversion is done here.

% ── HDF5 path constants ──────────────────────────────────────────────────────
RAW_PATH    = '/3BData/Raw';
CHS_PATH    = '/3BRecInfo/3BMeaStreams/Raw/Chs';
SR_PATH     = '/3BRecInfo/3BRecVars/SamplingRate';
NF_PATH     = '/3BRecInfo/3BRecVars/NRecFrames';

% ── Parse optional arguments ─────────────────────────────────────────────────
p = inputParser();
p.addParameter('Channels',  [],  @(x) isnumeric(x));
p.addParameter('Start',      0,  @(x) isnumeric(x) && isscalar(x));
p.addParameter('End',       [],  @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
p.parse(varargin{:});

req_channels = p.Results.Channels(:)';   % row vec, 0-based
start_idx    = p.Results.Start;          % 0-based
end_idx      = p.Results.End;            % 0-based exclusive ([] = all)

% ── Read metadata ─────────────────────────────────────────────────────────────
info = h5info(filepath, RAW_PATH);
dset_size = info.Dataspace.Size;   % MATLAB returns in Fortran order

chs_data = h5read(filepath, CHS_PATH);   % struct with Row, Col fields
rows = double(chs_data.Row);
cols = double(chs_data.Col);
n_channels = numel(rows);

% Sampling rate
sampling_rate = NaN;
try
    sampling_rate = double(h5read(filepath, SR_PATH));
    sampling_rate = sampling_rate(1);
catch
end

% Determine layout and n_frames
if numel(dset_size) == 2
    % 2D dataset: MATLAB h5read returns [n_channels × n_frames] (Fortran order)
    n_frames = dset_size(2);
    layout   = '2D';
elseif numel(dset_size) == 1
    n_total  = dset_size(1);
    if mod(n_total, n_channels) ~= 0
        error('mea_read_brw:corrupt', ...
            'Flat dataset length %d not divisible by channel count %d.', ...
            n_total, n_channels);
    end
    n_frames = n_total / n_channels;
    layout   = '1D';
else
    error('mea_read_brw:unsupported', ...
        'Unexpected dataset dimensionality: %dD', numel(dset_size));
end

% ── Assemble metadata struct ──────────────────────────────────────────────────
rec.filepath      = filepath;
rec.n_channels    = n_channels;
rec.n_frames      = n_frames;
rec.sampling_rate = sampling_rate;
rec.rows          = rows;
rec.cols          = cols;
rec.layout        = layout;

% ── Optionally load signal data ───────────────────────────────────────────────
if isempty(req_channels)
    return
end

% Resolve end index
if isempty(end_idx)
    end_idx = n_frames;
else
    end_idx = min(end_idx, n_frames);
end

n_samples = end_idx - start_idx;
if n_samples <= 0
    error('mea_read_brw:badRange', 'start (%d) must be less than end (%d).', ...
        start_idx, end_idx);
end

% Validate channel indices (0-based)
bad = req_channels(req_channels < 0 | req_channels >= n_channels);
if ~isempty(bad)
    error('mea_read_brw:badChannel', ...
        'Channel index/indices [%s] out of range (0–%d).', ...
        num2str(bad), n_channels - 1);
end

% Convert to 1-based for MATLAB
ch1 = req_channels + 1;   % 1-based channel indices

if strcmp(layout, '2D')
    % h5read block: start/count in Fortran (column-major) order
    % Dataset shape in MATLAB: [n_channels, n_frames]
    % We want rows ch1 and columns start_idx+1 : end_idx
    block = h5read(filepath, RAW_PATH, ...
        [1,            start_idx + 1], ...
        [n_channels,   n_samples]);
    % block is [n_channels × n_samples]; select requested channels
    traces = double(block(ch1, :))';   % → [n_samples × n_req_channels]

else  % 1D flat
    flat_start = start_idx * n_channels + 1;   % 1-based
    flat_count = n_samples * n_channels;
    flat = double(h5read(filepath, RAW_PATH, flat_start, flat_count));
    block = reshape(flat, n_channels, n_samples);
    traces = block(ch1, :)';                   % [n_samples × n_req_channels]
end

rec.traces          = traces;
rec.channel_indices = req_channels;

end