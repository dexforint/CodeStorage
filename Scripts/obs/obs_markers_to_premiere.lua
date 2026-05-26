obs = obslua

-- ---------- settings ----------
python_exe = "python"      -- можно полный путь: C:\\Python311\\python.exe
python_script = "C:/Users/user/Documents/Projects/CodeStorage/Scripts/obs/embedder.py"         -- путь к embedder.py (см. ниже)
call_delay_ms = 1500       -- подождать после остановки записи (чтобы OBS точно отпустил файл)
enabled = true

-- ---------- state ----------
markers = {}
point_count = 0
range_count = 0
open_range_start_ms = nil

rec_start_ns = nil
paused_accum_ns = 0
pause_start_ns = nil

pending_video_path = nil
pending_json_path = nil
pending_tries = 0

hotkey_point_id = nil
hotkey_range_toggle_id = nil

local function logi(s) obs.script_log(obs.LOG_INFO, "[markers] " .. s) end
local function logw(s) obs.script_log(obs.LOG_WARNING, "[markers] " .. s) end

local function quote(s)
  if s == nil then return '""' end
  s = tostring(s)
  s = s:gsub('"', '\\"')
  return '"' .. s .. '"'
end

local function recording_active()
  return obs.obs_frontend_recording_active()
end

local function now_ms()
  return math.floor(obs.os_gettime_ns() / 1000000 + 0.5)
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

local function add_point_marker(t_ms)
  point_count = point_count + 1
  local name = string.format("P%03d", point_count)
  table.insert(markers, {
    start_ms = t_ms,
    duration_ms = 0,
    name = name,
    comment = "point",
    track_type = "Comment",
  })
  logi("Point " .. name .. " @ " .. tostring(t_ms) .. " ms")
end

local function add_range_marker(s_ms, e_ms)
  if e_ms < s_ms then local tmp = s_ms; s_ms = e_ms; e_ms = tmp end
  range_count = range_count + 1
  local name = string.format("R%03d", range_count)
  table.insert(markers, {
    start_ms = s_ms,
    duration_ms = (e_ms - s_ms),
    name = name,
    comment = "range",
    track_type = "Comment",
  })
  logi("Range " .. name .. " " .. tostring(s_ms) .. ".." .. tostring(e_ms) .. " ms")
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

-- one-shot timer callback
function do_call_python()
  obs.timer_remove(do_call_python)

  if not enabled then return end
  if pending_video_path == nil or pending_json_path == nil then return end
  if python_script == nil or python_script == "" then
    logw("python_script is not set in script settings.")
    return
  end

  pending_tries = pending_tries + 1

  local cmd =
    quote(python_exe) .. " " ..
    quote(python_script) .. " " ..
    "--video " .. quote(pending_video_path) .. " " ..
    "--markers " .. quote(pending_json_path)

  logi("Running: " .. cmd)
  local ok = os.execute(cmd)

  -- если не получилось (например файл ещё занят) — попробуем ещё пару раз
  if ok == nil or ok == false then
    if pending_tries < 5 then
      logw("Python call failed; retry in 1s (try " .. tostring(pending_tries) .. ")")
      obs.timer_add(do_call_python, 1000)
    else
      logw("Python call failed; giving up.")
    end
  else
    logi("Python done.")
  end
end

-- ---------- hotkeys ----------
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

-- ---------- OBS events ----------
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

    logi("Recording started; reset markers.")
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
    -- закрыть открытый range в конец (если остался)
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
    local ok = write_json(json_path, video_path)
    if not ok then return end

    pending_video_path = video_path
    pending_json_path = json_path
    pending_tries = 0

    logi("Wrote JSON: " .. json_path .. " (markers=" .. tostring(#markers) .. ")")
    obs.timer_add(do_call_python, call_delay_ms)

    rec_start_ns = nil
    paused_accum_ns = 0
    pause_start_ns = nil
  end
end

-- ---------- UI ----------
function script_description()
  return "Capture OBS hotkeys as markers, save JSON, then call Python to embed Premiere markers into MP4 via exiftool."
end

function script_properties()
  local props = obs.obs_properties_create()
  obs.obs_properties_add_bool(props, "enabled", "Enabled")
  obs.obs_properties_add_text(props, "python_exe", "Python executable (python.exe)", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_path(props, "python_script", "Python embedder script (embedder.py)", obs.OBS_PATH_FILE, "*.py", nil)
  obs.obs_properties_add_int(props, "call_delay_ms", "Delay after stop (ms)", 0, 10000, 100)
  return props
end

function script_update(settings)
  enabled = obs.obs_data_get_bool(settings, "enabled")
  python_exe = obs.obs_data_get_string(settings, "python_exe")
  python_script = obs.obs_data_get_string(settings, "python_script")
  call_delay_ms = obs.obs_data_get_int(settings, "call_delay_ms")
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

  logi("Loaded. Assign hotkeys in Settings -> Hotkeys.")
end

function script_save(settings)
  local hk_point = obs.obs_hotkey_save(hotkey_point_id)
  obs.obs_data_set_array(settings, "premiere_markers_point", hk_point)
  obs.obs_data_array_release(hk_point)

  local hk_range = obs.obs_hotkey_save(hotkey_range_toggle_id)
  obs.obs_data_set_array(settings, "premiere_markers_range_toggle", hk_range)
  obs.obs_data_array_release(hk_range)
end