obs = obslua

-- Settings
enabled = true
python_exe = "python"                  -- можно полный путь: C:\\Python311\\python.exe
python_script = "C:\\Users\\user\\Documents\\Projects\\CodeStorage\\Scripts\\obspremiere_embed_markers.py"
ffprobe_exe = "ffprobe"
exiftool_exe = "exiftool"
call_delay_ms = 2000                   -- задержка после stop (файл может быть занят)
write_sidecar = true                   -- писать video.mp4.xmp (удобно для отладки)
max_retries = 8
retry_delay_ms = 1500

-- State
markers = {}
point_count = 0
range_count = 0
open_range_start_ms = nil

rec_start_ns = nil
paused_accum_ns = 0
pause_start_ns = nil

pending_cmd = nil
pending_log = nil
pending_try = 0

hotkey_point_id = nil
hotkey_range_toggle_id = nil

local function logi(s) obs.script_log(obs.LOG_INFO, "[premiere-markers] " .. s) end
local function logw(s) obs.script_log(obs.LOG_WARNING, "[premiere-markers] " .. s) end

local function recording_active()
  return obs.obs_frontend_recording_active()
end

local function get_rec_time_ms()
  if not recording_active() then return nil end
  if rec_start_ns == nil then
    rec_start_ns = obs.os_gettime_ns()
    paused_accum_ns = 0
    pause_start_ns = nil
  end

  local now = obs.os_gettime_ns()
  local paused_now_ns = paused_accum_ns
  if pause_start_ns ~= nil then
    paused_now_ns = paused_now_ns + (now - pause_start_ns)
  end

  local elapsed_ns = (now - rec_start_ns) - paused_now_ns
  if elapsed_ns < 0 then elapsed_ns = 0 end
  return math.floor(elapsed_ns / 1000000 + 0.5)
end

local function json_escape(s)
  s = tostring(s or "")
  s = s:gsub("\\", "\\\\")
  s = s:gsub("\"", "\\\"")
  s = s:gsub("\r", "\\r")
  s = s:gsub("\n", "\\n")
  s = s:gsub("\t", "\\t")
  return s
end

local function write_json(path, video_path)
  local f, err = io.open(path, "w")
  if not f then
    logw("Cannot write JSON: " .. tostring(err))
    return false
  end

  f:write("{\n")
  f:write('  "video": "' .. json_escape(video_path) .. '",\n')
  f:write('  "markers": [\n')

  for i, m in ipairs(markers) do
    f:write("    {\n")
    f:write('      "start_ms": ' .. tostring(m.start_ms) .. ",\n")
    f:write('      "duration_ms": ' .. tostring(m.duration_ms or 0) .. ",\n")
    f:write('      "name": "' .. json_escape(m.name or "") .. '",\n')
    f:write('      "comment": "' .. json_escape(m.comment or "") .. '",\n')
    f:write('      "track_type": "' .. json_escape(m.track_type or "Comment") .. '"\n')
    f:write("    }" .. (i < #markers and "," or "") .. "\n")
  end

  f:write("  ]\n")
  f:write("}\n")
  f:close()
  return true
end

local function add_point_marker(t_ms)
  point_count = point_count + 1
  local name = string.format("P%03d", point_count)
  table.insert(markers, { start_ms = t_ms, duration_ms = 0, name = name, comment = "point", track_type = "Comment" })
  logi("Point " .. name .. " @ " .. tostring(t_ms) .. " ms")
end

local function add_range_marker(s_ms, e_ms)
  if e_ms < s_ms then local tmp = s_ms; s_ms = e_ms; e_ms = tmp end
  range_count = range_count + 1
  local name = string.format("R%03d", range_count)
  table.insert(markers, { start_ms = s_ms, duration_ms = (e_ms - s_ms), name = name, comment = "range", track_type = "Comment" })
  logi("Range " .. name .. " " .. tostring(s_ms) .. ".." .. tostring(e_ms) .. " ms")
end

-- Robust Windows cmd quoting pattern:
-- cmd.exe /C ""arg1" "arg2" ... "
local function cmd_quote(arg)
  arg = tostring(arg or "")
  arg = arg:gsub('"', '""')
  return '"' .. arg .. '"'
end

local function build_cmd(video_path, json_path, log_path)
  local sidecar_flag = write_sidecar and " --write-sidecar" or ""
  local cmd =
    'cmd.exe /C ""' ..
    cmd_quote(python_exe) .. " " ..
    cmd_quote(python_script) ..
    " --video " .. cmd_quote(video_path) ..
    " --markers " .. cmd_quote(json_path) ..
    " --ffprobe " .. cmd_quote(ffprobe_exe) ..
    " --exiftool " .. cmd_quote(exiftool_exe) ..
    sidecar_flag ..
    " 1>>" .. cmd_quote(log_path) .. " 2>>&1" ..
    ' "'
  return cmd
end

function call_python_retry()
  obs.timer_remove(call_python_retry)

  if not enabled or pending_cmd == nil then return end

  pending_try = pending_try + 1
  logi("Run embedder (try " .. tostring(pending_try) .. "): " .. pending_cmd)

  local rc = os.execute(pending_cmd)

  if rc == 0 then
    logi("Embedder OK. Log: " .. pending_log)
    pending_cmd = nil
    pending_log = nil
    pending_try = 0
    return
  end

  if pending_try < max_retries then
    logw("Embedder failed (rc=" .. tostring(rc) .. "). Retry in " .. tostring(retry_delay_ms) .. " ms. Log: " .. pending_log)
    obs.timer_add(call_python_retry, retry_delay_ms)
  else
    logw("Embedder failed; giving up. Log: " .. pending_log)
    pending_cmd = nil
    pending_log = nil
    pending_try = 0
  end
end

-- Hotkeys
function on_hotkey_point(pressed)
  if not pressed or not enabled then return end
  local t = get_rec_time_ms()
  if t == nil then return end
  add_point_marker(t)
end

function on_hotkey_range_toggle(pressed)
  if not pressed or not enabled then return end
  local t = get_rec_time_ms()
  if t == nil then return end

  if open_range_start_ms == nil then
    open_range_start_ms = t
    logi("Range START @ " .. tostring(t) .. " ms")
  else
    add_range_marker(open_range_start_ms, t)
    open_range_start_ms = nil
  end
end

-- OBS events
function on_frontend_event(event)
  if not enabled then return end

  if event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED then
    markers = {}
    point_count = 0
    range_count = 0
    open_range_start_ms = nil

    rec_start_ns = obs.os_gettime_ns()
    paused_accum_ns = 0
    pause_start_ns = nil

    logi("Recording started: reset markers.")
  elseif event == obs.OBS_FRONTEND_EVENT_RECORDING_PAUSED then
    if pause_start_ns == nil then
      pause_start_ns = obs.os_gettime_ns()
      logi("Recording paused.")
    end
  elseif event == obs.OBS_FRONTEND_EVENT_RECORDING_UNPAUSED then
    if pause_start_ns ~= nil then
      paused_accum_ns = paused_accum_ns + (obs.os_gettime_ns() - pause_start_ns)
      pause_start_ns = nil
      logi("Recording unpaused.")
    end
  elseif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED then
    -- close open range at end
    local final_ms = get_rec_time_ms()
    if open_range_start_ms ~= nil and final_ms ~= nil then
      add_range_marker(open_range_start_ms, final_ms)
      open_range_start_ms = nil
    end

    local video_path = obs.obs_frontend_get_last_recording()
    if video_path == nil or video_path == "" then
      logw("No last recording path.")
      return
    end

    local json_path = video_path .. ".markers.json"
    if not write_json(json_path, video_path) then return end

    local log_path = video_path .. ".markers_embed.log"
    local cmd = build_cmd(video_path, json_path, log_path)

    pending_cmd = cmd
    pending_log = log_path
    pending_try = 0

    logi("JSON written: " .. json_path .. " markers=" .. tostring(#markers))
    obs.timer_add(call_python_retry, call_delay_ms)

    rec_start_ns = nil
    paused_accum_ns = 0
    pause_start_ns = nil
  end
end

-- Script UI
function script_description()
  return "OBS hotkeys -> JSON -> Python embeds Premiere-style XMP markers into MP4 using ffprobe+exiftool.\n" ..
         "Creates point markers and range markers (toggle)."
end

function script_properties()
  local props = obs.obs_properties_create()
  obs.obs_properties_add_bool(props, "enabled", "Enabled")
  obs.obs_properties_add_text(props, "python_exe", "python.exe path", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_path(props, "python_script", "premiere_embed_markers.py", obs.OBS_PATH_FILE, "*.py", nil)
  obs.obs_properties_add_text(props, "ffprobe_exe", "ffprobe.exe path", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_text(props, "exiftool_exe", "exiftool.exe path", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_int(props, "call_delay_ms", "Delay after stop (ms)", 0, 20000, 100)
  obs.obs_properties_add_bool(props, "write_sidecar", "Also write sidecar video.mp4.xmp")
  obs.obs_properties_add_int(props, "max_retries", "Max retries", 1, 30, 1)
  obs.obs_properties_add_int(props, "retry_delay_ms", "Retry delay (ms)", 200, 10000, 100)
  return props
end

function script_update(settings)
  enabled = obs.obs_data_get_bool(settings, "enabled")
  python_exe = obs.obs_data_get_string(settings, "python_exe")
  python_script = obs.obs_data_get_string(settings, "python_script")
  ffprobe_exe = obs.obs_data_get_string(settings, "ffprobe_exe")
  exiftool_exe = obs.obs_data_get_string(settings, "exiftool_exe")
  call_delay_ms = obs.obs_data_get_int(settings, "call_delay_ms")
  write_sidecar = obs.obs_data_get_bool(settings, "write_sidecar")
  max_retries = obs.obs_data_get_int(settings, "max_retries")
  retry_delay_ms = obs.obs_data_get_int(settings, "retry_delay_ms")
end

function script_load(settings)
  obs.obs_frontend_add_event_callback(on_frontend_event)

  hotkey_point_id = obs.obs_hotkey_register_frontend(
    "premiere_markers_point",
    "Premiere markers: Point marker",
    on_hotkey_point
  )
  hotkey_range_toggle_id = obs.obs_hotkey_register_frontend(
    "premiere_markers_range_toggle",
    "Premiere markers: Range marker (toggle start/end)",
    on_hotkey_range_toggle
  )

  local hk_point = obs.obs_data_get_array(settings, "premiere_markers_point")
  obs.obs_hotkey_load(hotkey_point_id, hk_point)
  obs.obs_data_array_release(hk_point)

  local hk_range = obs.obs_data_get_array(settings, "premiere_markers_range_toggle")
  obs.obs_hotkey_load(hotkey_range_toggle_id, hk_range)
  obs.obs_data_array_release(hk_range)

  logi("Loaded. Set hotkeys in Settings -> Hotkeys.")
end

function script_save(settings)
  local hk_point = obs.obs_hotkey_save(hotkey_point_id)
  obs.obs_data_set_array(settings, "premiere_markers_point", hk_point)
  obs.obs_data_array_release(hk_point)

  local hk_range = obs.obs_hotkey_save(hotkey_range_toggle_id)
  obs.obs_data_set_array(settings, "premiere_markers_range_toggle", hk_range)
  obs.obs_data_array_release(hk_range)
end