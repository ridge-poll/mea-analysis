function mea_surface_plot()
% MEA_SURFACE_PLOT  3-D spatial visualization of HD-MEA electrode activity.
%
% Display modes (set DISPLAY_MODE below):
%   'rms'       — static surface; Z = RMS amplitude over the loaded window
%   'snapshot'  — static surface; Z = amplitude at a single time point
%   'browse'    — static surface + a time slider you drag to scrub through
%                 the recording. An optional Play button will auto-advance
%                 the slider, but scrubbing (not playback) is the main way
%                 to use this mode. ('animate' is accepted as a legacy alias.)
%
% DEPENDENCIES
%   mea_read_brw.m — same directory or on the MATLAB path.
%   MATLAB R2019b+.
%   Signal Processing Toolbox — required for the 60 Hz notch filter
%   (iirnotch, filtfilt). Set APPLY_NOTCH_FILTER = false below if you
%   don't have it; everything else needs no toolboxes.
%
% USAGE
%   mea_surface_plot

% ════════════════════════════════════════════════════════════════════════════
%% CONFIG ────────────────────────────────────────────────────────────────────

START_SAMPLE = 0;
END_SAMPLE   = 80000;       % [] = whole file

DISPLAY_MODE = 'browse';   % 'rms' | 'snapshot' | 'browse'
SNAPSHOT_IDX = 1;          % 1-based (snapshot mode only)

PLAYBACK_STEP_SAMPLES = 10; % samples advanced per frame when Play is running
PLAYBACK_FPS          = 30; % only affects the optional Play button

MESH_RES        = 4;
CMAP            = 'turbo';
TRANSPARENCY    = 0.85;
SHOW_ELECTRODES = true;

OUTLIER_MAD_THRESH = 5;
OUTLIER_NEIGHBOR_K = 6;

APPLY_NOTCH_FILTER = true;  % remove mains hum before anything else runs
NOTCH_FREQ_HZ      = 60;    % use 50 outside the US/other 60 Hz-mains regions
NOTCH_Q            = 30;    % higher Q = narrower notch (less signal removed
                             % around 60 Hz, but less effective if hum drifts)

AMPLITUDE_ADJ = 1; % 1 for full z

% ════════════════════════════════════════════════════════════════════════════
%% FILE PICKER ────────────────────────────────────────────────────────────────

[fname, fpath] = uigetfile( ...
    {'*.brw;*.h5;*.hdf5', 'MEA recordings (*.brw, *.h5, *.hdf5)'; ...
     '*.*', 'All files (*.*)'}, 'Open MEA recording');

if isequal(fname, 0)
    disp('No file selected — aborting.');
    return
end
FILE_PATH = fullfile(fpath, fname);

% ════════════════════════════════════════════════════════════════════════════
%% LOAD + CLEAN DATA ──────────────────────────────────────────────────────────

fprintf('Loading %s ...\n', FILE_PATH);
meta   = mea_read_brw(FILE_PATH);
all_ch = 0 : meta.n_channels - 1;

rec = mea_read_brw(FILE_PATH, 'Channels', all_ch, ...
                               'Start',    START_SAMPLE, ...
                               'End',      END_SAMPLE);

fprintf('  %d channels  |  %d samples  |  layout: %s\n', ...
    rec.n_channels, size(rec.traces, 1), rec.layout);

if APPLY_NOTCH_FILTER
    fprintf('  Applying %d Hz notch filter (Q=%d) ...\n', NOTCH_FREQ_HZ, NOTCH_Q);
    rec.traces = apply_notch_filter(rec.traces, rec.sampling_rate, NOTCH_FREQ_HZ, NOTCH_Q);
end

rec.traces = reject_outliers(rec, OUTLIER_MAD_THRESH, OUTLIER_NEIGHBOR_K);

% ════════════════════════════════════════════════════════════════════════════
%% SPATIAL MESH ───────────────────────────────────────────────────────────────

ex = rec.cols;
ey = rec.rows;
x_min = min(ex);  x_max = max(ex);
y_min = min(ey);  y_max = max(ey);

nx = max(round((x_max - x_min) * MESH_RES) + 1, 2);
ny = max(round((y_max - y_min) * MESH_RES) + 1, 2);
[XI, YI] = meshgrid(linspace(x_min, x_max, nx), linspace(y_min, y_max, ny));

F = scatteredInterpolant(ex, ey, zeros(size(ex)), 'natural', 'none');

% ════════════════════════════════════════════════════════════════════════════
%% FIGURE ─────────────────────────────────────────────────────────────────────

fig = figure('Name', 'MEA Surface Plot', 'Color', [0.12 0.12 0.14], ...
             'NumberTitle', 'off', 'Position', [100 80 1020 720]);

ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.05 0.20 0.88 0.76], ...
          'Color', [0.10 0.10 0.12], ...
          'XColor', [0.75 0.75 0.75], 'YColor', [0.75 0.75 0.75], 'ZColor', [0.75 0.75 0.75], ...
          'GridColor', [0.40 0.40 0.40], 'GridAlpha', 0.3, 'BoxStyle', 'full', 'FontSize', 10);
hold(ax, 'on');
grid(ax, 'on');
colormap(ax, CMAP);

xlabel(ax, 'Column (X)', 'Color', [0.85 0.85 0.85]);
ylabel(ax, 'Row (Y)',    'Color', [0.85 0.85 0.85]);
zlabel(ax, 'Amplitude',  'Color', [0.85 0.85 0.85]);

cb = colorbar(ax);
cb.Color = [0.8 0.8 0.8];
cb.Label.String = 'Amplitude';
cb.Label.Color  = [0.8 0.8 0.8];

light(ax, 'Position', [x_max*2, y_max*2, 100], 'Style', 'infinite');
lighting(ax, 'gouraud');

sp = {'FaceAlpha', TRANSPARENCY, 'EdgeColor', 'none', 'FaceLighting', 'gouraud', ...
      'AmbientStrength', 0.35, 'DiffuseStrength', 0.75, 'SpecularStrength', 0.45};

% ════════════════════════════════════════════════════════════════════════════
%% RENDER ─────────────────────────────────────────────────────────────════════

mode = lower(DISPLAY_MODE);
if strcmp(mode, 'animate')
    mode = 'browse';   % legacy alias
end

if strcmp(mode, 'rms')

    z_vals = electrode_rms(rec.traces);
    ZI     = interp_surface(F, XI, YI, z_vals);
    surf(ax, XI, YI, ZI, sp{:});
    if SHOW_ELECTRODES
        scatter3(ax, ex, ey, z_vals, 28, z_vals, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
    end
    title(ax, sprintf('MEA RMS  |  %d ch  |  samples %d-%d', ...
        rec.n_channels, START_SAMPLE, START_SAMPLE + size(rec.traces,1)), ...
        'Color', [0.9 0.9 0.9], 'FontSize', 11);
    view(ax, -35, 30);  axis(ax, 'tight');  zlim_pad(ax);

elseif strcmp(mode, 'snapshot')

    z_vals = electrode_snapshot(rec.traces, SNAPSHOT_IDX);
    ZI     = interp_surface(F, XI, YI, z_vals);
    surf(ax, XI, YI, ZI, sp{:});
    if SHOW_ELECTRODES
        scatter3(ax, ex, ey, z_vals, 28, z_vals, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
    end
    t_s = (START_SAMPLE + SNAPSHOT_IDX - 1) / rec.sampling_rate;
    title(ax, sprintf('MEA snapshot  |  t = %.4f s  (sample %d)', ...
        t_s, START_SAMPLE + SNAPSHOT_IDX - 1), 'Color', [0.9 0.9 0.9], 'FontSize', 11);
    view(ax, -35, 30);  axis(ax, 'tight');  zlim_pad(ax);

elseif strcmp(mode, 'browse')

    n_samples = size(rec.traces, 1);

    % Global z-limits from cleaned, downsampled data
    ds           = max(1, floor(n_samples / 200));
    all_z        = rec.traces(1:ds:end, :);
    z_global_min = min(all_z(:));
    z_global_max = AMPLITUDE_ADJ*max(all_z(:));
    if z_global_min == z_global_max;  z_global_max = z_global_min + 1;  end
    clim(ax, [z_global_min, z_global_max]);

    % Initial surface (drawn once at sample 1, then only redrawn on scrub/play)
    z0     = electrode_snapshot(rec.traces, 1);
    ZI     = interp_surface(F, XI, YI, z0);
    h_surf = surf(ax, XI, YI, ZI, sp{:});
    if SHOW_ELECTRODES
        h_dots = scatter3(ax, ex, ey, z0, 28, z0, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.6);
    else
        h_dots = [];
    end
    view(ax, -35, 30);
    zlim(ax, [z_global_min, z_global_max]);
    axis(ax, 'manual');
    t_title = title(ax, '', 'Color', [0.9 0.9 0.9], 'FontSize', 11);

    % Controls panel — the slider is the primary control; Play is optional
    ctrl = uipanel('Parent', fig, 'Units', 'normalized', 'Position', [0.0 0.0 1.0 0.18], ...
                   'BackgroundColor', [0.10 0.10 0.12], 'BorderType', 'none');

    btn_play = uicontrol('Parent', ctrl, 'Style', 'pushbutton', 'String', 'Play', ...
                          'Units', 'normalized', 'Position', [0.01 0.25 0.08 0.50], ...
                          'BackgroundColor', [0.30 0.30 0.35], 'ForegroundColor', [0.95 0.95 0.95], ...
                          'FontSize', 11, 'FontWeight', 'bold');

    lbl_time = uicontrol('Parent', ctrl, 'Style', 'text', 'String', '0.000 s', ...
                         'Units', 'normalized', 'Position', [0.10 0.25 0.09 0.45], ...
                         'BackgroundColor', [0.10 0.10 0.12], 'ForegroundColor', [0.80 0.80 0.80], ...
                         'FontSize', 10, 'HorizontalAlignment', 'right');

    slider = uicontrol('Parent', ctrl, 'Style', 'slider', ...
                       'Units', 'normalized', 'Position', [0.20 0.35 0.73 0.30], ...
                       'Min', 1, 'Max', n_samples, 'Value', 1, ...
                       'SliderStep', [PLAYBACK_STEP_SAMPLES/n_samples, min(1, 100*PLAYBACK_STEP_SAMPLES/n_samples)], ...
                       'BackgroundColor', [0.28 0.28 0.33]);

    t_end_s = (START_SAMPLE + n_samples - 1) / rec.sampling_rate;
    uicontrol('Parent', ctrl, 'Style', 'text', ...
              'String', sprintf('%.2f s', t_end_s), ...
              'Units', 'normalized', 'Position', [0.94 0.25 0.05 0.45], ...
              'BackgroundColor', [0.10 0.10 0.12], 'ForegroundColor', [0.55 0.55 0.55], ...
              'FontSize', 9);

    % Mutable state passed via UserData on the figure
    state.t_idx     = 1;
    state.playing   = false;   % starts paused — scrubbing is the default interaction
    state.n_samples = n_samples;
    state.step      = PLAYBACK_STEP_SAMPLES;
    state.sr        = rec.sampling_rate;
    state.start     = START_SAMPLE;
    state.traces    = rec.traces;
    state.F         = F;
    state.XI        = XI;
    state.YI        = YI;
    state.ex        = ex;
    state.ey        = ey;
    state.h_surf    = h_surf;
    state.h_dots    = h_dots;
    state.t_title   = t_title;
    state.lbl_time  = lbl_time;
    state.slider    = slider;
    state.btn_play  = btn_play;
    fig.UserData    = state;

    % Timer only exists/ticks while actually playing — it is started and
    % stopped by the Play/Pause button rather than left running in the
    % background and merely skipping its own redraw. That was the main
    % cause of "pause feels slow": a timer that keeps firing every frame
    % still queues events ahead of your click even when it isn't drawing
    % anything, so Pause had to wait its turn. Now there's simply nothing
    % running once you pause, so it's instant.
    tmr = timer('ExecutionMode', 'fixedRate', ...
                'Period',        max(0.01, 1/PLAYBACK_FPS), ...
                'TimerFcn',      @(~,~) cb_step_frame(fig));
    fig.DeleteFcn = @(~,~) safe_stop_timer(tmr);
    state.tmr = tmr;
    fig.UserData = state;

    % Wire up callbacks now that state is stored
    btn_play.Callback = @(~,~) cb_toggle_play(fig);
    slider.Callback   = @(~,~) cb_slider_moved(fig);   % fires on release
    % Also update live while dragging, not just on release. This is the
    % fix for "scrolling doesn't work": a plain uicontrol slider Callback
    % only fires once you let go of the mouse. ContinuousValueChange fires
    % continuously while dragging, in normal (non-uifigure) figures.
    try
        addlistener(slider, 'ContinuousValueChange', @(~,~) cb_slider_moved(fig));
    catch
        % Falls back to release-only updates if unsupported in this
        % MATLAB/figure configuration.
    end

    fprintf('Drag the slider to scrub through time. Click Play to auto-advance.\n');

else
    error('mea_surface_plot:badMode', ...
        'Unknown DISPLAY_MODE "%s". Use "rms", "snapshot", or "browse".', DISPLAY_MODE);
end

% ════════════════════════════════════════════════════════════════════════════
%% BROWSE / PLAYBACK CALLBACKS (plain local functions — no nesting) ═══════════

function cb_step_frame(fig)
    if ~ishandle(fig);  return;  end
    s = fig.UserData;
    if ~s.playing;  return;  end
    draw_frame(fig, s.t_idx);
    s.t_idx = s.t_idx + s.step;
    if s.t_idx > s.n_samples;  s.t_idx = 1;  end
    s.slider.Value = s.t_idx;
    fig.UserData = s;
end

function cb_slider_moved(fig)
    if ~ishandle(fig);  return;  end
    s = fig.UserData;
    idx = max(1, min(round(s.slider.Value), s.n_samples));
    if idx == s.t_idx
        return   % no-op if the drag didn't cross a new sample
    end
    s.t_idx = idx;
    fig.UserData = s;
    draw_frame(fig, idx);
end

function cb_toggle_play(fig)
    if ~ishandle(fig);  return;  end
    s = fig.UserData;
    s.playing = ~s.playing;
    if s.playing
        s.btn_play.String          = 'Pause';
        s.btn_play.BackgroundColor = [0.20 0.45 0.20];
        start(s.tmr);
    else
        s.btn_play.String          = 'Play';
        s.btn_play.BackgroundColor = [0.30 0.30 0.35];
        stop(s.tmr);   % nothing left running -> Pause is instant
    end
    fig.UserData = s;
end

function draw_frame(fig, idx)
    if ~ishandle(fig);  return;  end
    s      = fig.UserData;
    z_vals = electrode_snapshot(s.traces, idx);
    ZI     = interp_surface(s.F, s.XI, s.YI, z_vals);
    set(s.h_surf, 'ZData', ZI, 'CData', ZI);
    if ~isempty(s.h_dots) && ishandle(s.h_dots)
        set(s.h_dots, 'ZData', z_vals, 'CData', z_vals);
    end
    t_s = (s.start + idx - 1) / s.sr;
    set(s.t_title, 'String', sprintf('MEA  |  t = %.4f s  (sample %d / %d)', ...
        t_s, s.start + idx - 1, s.start + s.n_samples - 1));
    s.lbl_time.String = sprintf('%.3f s', t_s);
    drawnow;   % plain drawnow: scrubbing is user-driven, so update immediately
               % rather than deferring to the next 'limitrate' opportunity
end

% ════════════════════════════════════════════════════════════════════════════
%% SIGNAL / INTERPOLATION HELPERS ─────────────────────────────────────────────

function traces = apply_notch_filter(traces, fs, f0, Q)
    % Signal Processing Toolbox notch filter. filtfilt applies it
    % zero-phase with proper edge padding, so start/end samples don't
    % ring the way a plain forward-backward filter() pass can.
    % traces is [n_samples x n_channels]; filtfilt filters each column.
    if f0 <= 0 || f0 >= fs/2
        warning('mea_surface_plot:badNotchFreq', ...
            'Notch frequency %g Hz is outside the valid range for fs=%g Hz — skipping.', f0, fs);
        return
    end
    w0 = f0 / (fs/2);   % normalized frequency (0 to 1, 1 = Nyquist)
    bw = w0 / Q;        % normalized notch bandwidth
    [b, a] = iirnotch(w0, bw);
    traces = filtfilt(b, a, traces);
end

function z = electrode_rms(traces)
    z = sqrt(mean(traces .^ 2, 1))';
end

function z = electrode_snapshot(traces, t_idx)
    t_idx = max(1, min(t_idx, size(traces, 1)));
    z = traces(t_idx, :)';
end

function ZI = interp_surface(F, XI, YI, z_vals)
    F.Values = z_vals;
    ZI = F(XI, YI);
    ZI(isnan(ZI)) = 0;
end

function zlim_pad(ax)
    zl = zlim(ax);
    span = max(zl(2) - zl(1), 1);
    zlim(ax, [zl(1) - 0.1*span, zl(2) + 0.1*span]);
end

function safe_stop_timer(tmr)
    try
        if isvalid(tmr);  stop(tmr);  delete(tmr);  end
    catch
    end
end

function cleaned = reject_outliers(rec, mad_thresh, n_neighbors)
    cleaned = rec.traces;
    ch_var  = var(cleaned, 0, 1);
    med_var = median(ch_var);
    mad_var = median(abs(ch_var - med_var));
    if mad_var == 0
        fprintf('  No outlier electrodes detected.\n');  return
    end
    outlier_mask = ch_var > (med_var + mad_thresh * mad_var);
    n_out = sum(outlier_mask);
    if n_out == 0
        fprintf('  No outlier electrodes detected.\n');  return
    end
    fprintf('  Outlier electrodes detected (%d): channels %s\n', ...
        n_out, num2str(find(outlier_mask) - 1));
    fprintf('  Replacing with IDW of %d nearest spatial neighbours ...\n', n_neighbors);
    ex_all = rec.cols;
    ey_all = rec.rows;
    for bad_i = find(outlier_mask)
        dx   = ex_all - ex_all(bad_i);
        dy   = ey_all - ey_all(bad_i);
        dist = sqrt(dx.^2 + dy.^2);
        dist(bad_i)        = Inf;
        dist(outlier_mask) = Inf;
        [sorted_d, sorted_i] = sort(dist);
        k    = min(n_neighbors, sum(~isinf(sorted_d)));
        nbrs = sorted_i(1:k);
        w    = 1 ./ sorted_d(1:k);  w = w / sum(w);
        cleaned(:, bad_i) = cleaned(:, nbrs) * w;
    end
end

end  % mea_surface_plot