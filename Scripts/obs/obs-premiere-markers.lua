obs = obslua

-- ====== defaults / settings ======
local DEFAULT_FPS_TEXT = "60"      -- или "60000/1001" для 59.94
local DEFAULT_TRACK_NAME = "Markers"
local DEFAULT_POINT_PREFIX = "P"
local DEFAULT_RANGE_PREFIX = "R"

fps_text = DEFAULT_FPS_TEXT
track_name = DEFAULT_TRACK_NAME
point_prefix = DEFAULT_POINT_PREFIX
range_prefix = DEFAULT_RANGE_PREFIX
auto_close_open_range = true

-- ====== state ======
markers = {}
point_count = 0
range_count = 0
open_range_start_ms = nil

rec_start_ns = nil
paused_accum_ns = 0
pause_start_ns = nil

hotkey_point_id = nil
hotkey_range_toggle_id = nil

-- ====== helpers ======
local function logi(s) obs.script_log(obs.LOG_INFO, "[premiere-markers] " .. s) end
local function logw(s) obs.script_log(obs.LOG_WARNING, "[premiere-markers] " .. s) end

local function xml_escape(s)
  if s == nil then return "" end
  s = tostring(s)
  s = s:gsub("&", "&amp;")
  s = s:gsub("<", "&lt;")
  s = s:gsub(">", "&gt;")
  s = s:gsub("\"", "&quot;")
  s = s:gsub("'", "&apos;")
  return s
end

local function parse_fps(s)
  s = tostring(s or ""):gsub("%s+", "")
  local a, b = s:match("^(%d+)%/(%d+)$")
  if a and b then
    local num = tonumber(a)
    local den = tonumber(b)
    if num and den and num > 0 and den > 0 then return num, den end
  end
  local n = tonumber(s)
  if n and n > 0 then return n, 1 end
  return 60, 1
end

local function ms_to_frames(ms, fps_num, fps_den)
  local v = (ms * fps_num) / (1000.0 * fps_den)
  return math.floor(v + 0.5)
end

local function get_scale_string(fps_num, fps_den)
  return tostring(fps_den) .. "/" .. tostring(fps_num) -- 60 -> 1/60, 60000/1001 -> 1001/60000
end

local function recording_active()
  return obs.obs_frontend_recording_active()
end

local function ensure_rec_clock_started()
  if rec_start_ns == nil then
    rec_start_ns = obs.os_gettime_ns()
    paused_accum_ns = 0
    pause_start_ns = nil
  end
end

local function get_rec_time_ms()
  if not recording_active() then return nil end
  ensure_rec_clock_started()

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
  local name = string.format("%s%03d", point_prefix, point_count)
  table.insert(markers, { name = name, comment = "point", start_ms = t_ms, end_ms = nil })
  logi("Point " .. name .. " @ " .. tostring(t_ms) .. " ms")
end

local function add_range_marker(s_ms, e_ms)
  if e_ms < s_ms then
    local tmp = s_ms; s_ms = e_ms; e_ms = tmp
  end
  range_count = range_count + 1
  local name = string.format("%s%03d", range_prefix, range_count)
  table.insert(markers, { name = name, comment = "range", start_ms = s_ms, end_ms = e_ms })
  logi("Range " .. name .. " " .. tostring(s_ms) .. ".." .. tostring(e_ms) .. " ms")
end

-- ====== XMP writer ======
local function build_xmp()
  local fps_num, fps_den = parse_fps(fps_text)
  local scale = get_scale_string(fps_num, fps_den)

  local out = {}
  table.insert(out, '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>')
  table.insert(out, '<x:xmpmeta xmlns:x="adobe:ns:meta/">')
  table.insert(out, '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"')
  table.insert(out, '           xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">')
  table.insert(out, '    <rdf:Description rdf:about="">')
  table.insert(out, '      <xmpDM:Tracks>')
  table.insert(out, '        <rdf:Bag>')
  table.insert(out, '          <rdf:li rdf:parseType="Resource">')
  table.insert(out, '            <xmpDM:trackName>' .. xml_escape(track_name) .. '</xmpDM:trackName>')
  table.insert(out, '            <xmpDM:trackType>Cue</xmpDM:trackType>')
  table.insert(out, '            <xmpDM:markers>')
  table.insert(out, '              <rdf:Seq>')

  for _, m in ipairs(markers) do
    local start_frames = ms_to_frames(m.start_ms, fps_num, fps_den)
    local dur_frames = 0
    if m.end_ms ~= nil then
      dur_frames = ms_to_frames((m.end_ms - m.start_ms), fps_num, fps_den)
      if dur_frames < 0 then dur_frames = 0 end
    end

    table.insert(out, '                <rdf:li rdf:parseType="Resource">')
    table.insert(out, '                  <xmpDM:name>' .. xml_escape(m.name) .. '</xmpDM:name>')
    table.insert(out, '                  <xmpDM:comment>' .. xml_escape(m.comment) .. '</xmpDM:comment>')
    table.insert(out, '                  <xmpDM:startTime rdf:parseType="Resource">')
    table.insert(out, '                    <xmpDM:value>' .. tostring(start_frames) .. '</xmpDM:value>')
    table.insert(out, '                    <xmpDM:scale>' .. scale .. '</xmpDM:scale>')
    table.insert(out, '                  </xmpDM:startTime>')
    table.insert(out, '                  <xmpDM:duration rdf:parseType="Resource">')
    table.insert(out, '                    <xmpDM:value>' .. tostring(dur_frames) .. '</xmpDM:value>')
    table.insert(out, '                    <xmpDM:scale>' .. scale .. '</xmpDM:scale>')
    table.insert(out, '                  </xmpDM:duration>')
    table.insert(out, '                </rdf:li>')
  end

  table.insert(out, '              </rdf:Seq>')
  table.insert(out, '            </xmpDM:markers>')
  table.insert(out, '          </rdf:li>')
  table.insert(out, '        </rdf:Bag>')
  table.insert(out, '      </xmpDM:Tracks>')
  table.insert(out, '    </rdf:Description>')
  table.insert(out, '  </rdf:RDF>')
  table.insert(out, '</x:xmpmeta>')
  table.insert(out, '<?xpacket end="w"?>')

  return table.concat(out, "\n")
end

local function write_sidecar_xmp_for_last_recording()
  local path = obs.obs_frontend_get_last_recording()
  if path == nil or path == "" then
    logw("No last recording path available; cannot write XMP.")
    return
  end

  local xmp_path = path .. ".xmp"
  local f, err = io.open(xmp_path, "w")
  if not f then
    logw("Failed to write XMP: " .. tostring(err) .. " path=" .. xmp_path)
    return
  end

  f:write(build_xmp())
  f:close()
  logi("Wrote XMP: " .. xmp_path .. " (markers=" .. tostring(#markers) .. ")")
end

-- ====== hotkeys ======
function on_hotkey_point(pressed)
  if not pressed then return end
  local t = get_rec_time_ms()
  if t == nil then return end
  add_point_marker(t)
end

function on_hotkey_range_toggle(pressed)
  if not pressed then return end
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

-- ====== OBS events ======
function on_frontend_event(event)
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
    -- вычислим финальное время записи (на случай auto-close)
    local final_ms = nil
    if rec_start_ns ~= nil then
      local now = obs.os_gettime_ns()
      local paused_now_ns = paused_accum_ns
      if pause_start_ns ~= nil then
        paused_now_ns = paused_now_ns + (now - pause_start_ns)
      end
      local elapsed_ns = (now - rec_start_ns) - paused_now_ns
      if elapsed_ns < 0 then elapsed_ns = 0 end
      final_ms = math.floor(elapsed_ns / 1000000 + 0.5)
    end

    if open_range_start_ms ~= nil and auto_close_open_range and final_ms ~= nil then
      add_range_marker(open_range_start_ms, final_ms)
      open_range_start_ms = nil
    end

    write_sidecar_xmp_for_last_recording()

    rec_start_ns = nil
    paused_accum_ns = 0
    pause_start_ns = nil
  end
end

-- ====== script UI / lifecycle ======
function script_description()
  return "Writes Premiere Pro clip markers (.mp4.xmp) from OBS hotkeys.\n" ..
         "Point marker + Range marker (toggle start/end)."
end

function script_defaults(settings)
  obs.obs_data_set_default_string(settings, "fps_text", DEFAULT_FPS_TEXT)
  obs.obs_data_set_default_string(settings, "track_name", DEFAULT_TRACK_NAME)
  obs.obs_data_set_default_string(settings, "point_prefix", DEFAULT_POINT_PREFIX)
  obs.obs_data_set_default_string(settings, "range_prefix", DEFAULT_RANGE_PREFIX)
  obs.obs_data_set_default_bool(settings, "auto_close_open_range", true)
end

function script_properties()
  local props = obs.obs_properties_create()
  obs.obs_properties_add_text(props, "fps_text", "FPS (например 60 или 60000/1001)", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_text(props, "track_name", "XMP Track name", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_text(props, "point_prefix", "Point prefix", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_text(props, "range_prefix", "Range prefix", obs.OBS_TEXT_DEFAULT)
  obs.obs_properties_add_bool(props, "auto_close_open_range", "Auto-close open range on stop")
  return props
end

function script_update(settings)
  fps_text = obs.obs_data_get_string(settings, "fps_text")
  track_name = obs.obs_data_get_string(settings, "track_name")
  point_prefix = obs.obs_data_get_string(settings, "point_prefix")
  range_prefix = obs.obs_data_get_string(settings, "range_prefix")
  auto_close_open_range = obs.obs_data_get_bool(settings, "auto_close_open_range")

  if fps_text == nil or fps_text == "" then fps_text = DEFAULT_FPS_TEXT end
  if track_name == nil or track_name == "" then track_name = DEFAULT_TRACK_NAME end
  if point_prefix == nil or point_prefix == "" then point_prefix = DEFAULT_POINT_PREFIX end
  if range_prefix == nil or range_prefix == "" then range_prefix = DEFAULT_RANGE_PREFIX end
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