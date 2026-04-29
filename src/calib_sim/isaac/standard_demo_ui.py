"""Browser UI for the slim tabletop standard demo."""

from __future__ import annotations

import html


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tabletop Auto-Demo Deck</title>
  <style>
    :root {{
      --bg: #081018;
      --panel: rgba(10, 19, 28, 0.94);
      --line: rgba(118, 160, 186, 0.22);
      --text: #eef6fb;
      --muted: #9db2c2;
      --accent: #2ad0be;
      --success: #1ad886;
      --danger: #ff766d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(42, 208, 190, 0.12), transparent 28%),
        linear-gradient(180deg, #0d1822 0%, #081018 65%, #05090f 100%);
    }}
    .shell {{
      width: min(1840px, calc(100vw - 24px));
      margin: 12px auto 18px;
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(360px, 0.85fr);
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font-size: clamp(1.45rem, 2.4vw, 2.2rem); }}
    h2 {{ font-size: 1rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.45;
      max-width: 62ch;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
    }}
    .status-pill {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(10, 18, 28, 0.95);
      border: 1px solid rgba(111, 155, 182, 0.26);
    }}
    .status-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
      box-shadow: 0 0 0 4px rgba(255, 118, 109, 0.16);
    }}
    .status-dot.live {{
      background: var(--success);
      box-shadow: 0 0 0 4px rgba(26, 216, 134, 0.16);
    }}
    .section {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(8, 17, 27, 0.82);
      border: 1px solid rgba(118, 160, 186, 0.16);
      margin-top: 16px;
    }}
    .controls, .metric-grid, .joint-grid, .observer-grid, .button-row {{
      display: grid;
      gap: 12px;
    }}
    .controls {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 12px;
    }}
    .metric-grid {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 12px;
    }}
    .joint-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 12px;
    }}
    .observer-grid {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 12px;
    }}
    .button-row {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 12px;
    }}
    .button-row.three {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    select, input[type="number"], button {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(118, 160, 186, 0.22);
      background: rgba(6, 13, 21, 0.92);
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
      background: linear-gradient(135deg, rgba(42, 208, 190, 0.22), rgba(19, 119, 176, 0.28));
    }}
    button.secondary {{
      background: rgba(8, 18, 30, 0.88);
    }}
    .metric {{
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(8, 18, 30, 0.86);
      border: 1px solid rgba(118, 160, 186, 0.14);
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    .metric .value {{
      margin-top: 6px;
      font-size: 0.98rem;
      line-height: 1.35;
      word-break: break-word;
    }}
    .preview-frame {{
      width: 100%;
      min-height: 260px;
      border-radius: 16px;
      border: 1px solid rgba(118, 160, 186, 0.14);
      background: linear-gradient(180deg, rgba(7, 16, 25, 0.94), rgba(6, 12, 20, 0.98));
      object-fit: contain;
      display: block;
      margin-top: 12px;
    }}
    .observer-frame {{
      min-height: 170px;
    }}
    .preview-tile {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(6, 13, 21, 0.9);
      border: 1px solid rgba(118, 160, 186, 0.14);
    }}
    .preview-heading {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }}
    .preview-meta, .small {{
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
    }}
    .placeholder {{
      margin-top: 10px;
      padding: 12px;
      border-radius: 14px;
      border: 1px dashed rgba(118, 160, 186, 0.16);
      color: var(--muted);
      text-align: center;
    }}
    pre {{
      margin: 10px 0 0;
      padding: 12px;
      border-radius: 14px;
      background: rgba(6, 13, 21, 0.94);
      border: 1px solid rgba(118, 160, 186, 0.12);
      overflow: auto;
      color: #dceaf2;
      font-size: 0.82rem;
      line-height: 1.45;
    }}
    @media (max-width: 1100px) {{
      .shell {{ grid-template-columns: 1fr; }}
      .controls, .metric-grid, .joint-grid, .observer-grid, .button-row, .button-row.three {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <div class="hero">
        <div>
          <h1>Tabletop Auto-Demo Deck</h1>
          <div class="subtitle">
            This slim app is locked to <strong>{default_profile_label}</strong>, the tabletop auto-path demo, and the <strong>new_pupil</strong> detector.
            It keeps the live camera views, robot controls, recording, and export-friendly runtime while dropping the old multi-profile and container experiments.
          </div>
        </div>
        <div class="status-pill">
          <span id="session-dot" class="status-dot"></span>
          <span id="session-state-text">Idle</span>
        </div>
      </div>

      <section class="section">
        <h2>Live Previews</h2>
        <div class="small">Mounted and observer previews are lightweight in-page monitoring views. The live mounted stream still uses Isaac WebRTC underneath.</div>
        <div class="preview-tile" style="margin-top: 12px;">
          <div class="preview-heading">
            <strong>Mounted Camera</strong>
            <span id="mounted-preview-meta" class="preview-meta">waiting for frames</span>
          </div>
          <img id="mounted-preview" class="preview-frame" alt="Mounted camera preview" />
          <div id="mounted-preview-placeholder" class="placeholder">Mounted preview appears once the tabletop session is live.</div>
        </div>
        <div class="observer-grid">
          <div class="preview-tile">
            <div class="preview-heading"><strong>room_wide</strong><span id="observer-room_wide-meta" class="preview-meta">waiting</span></div>
            <img id="observer-room_wide" class="preview-frame observer-frame" alt="room_wide observer preview" />
            <div id="observer-room_wide-placeholder" class="placeholder">room_wide is waiting for its first frame.</div>
          </div>
          <div class="preview-tile">
            <div class="preview-heading"><strong>side_overwatch</strong><span id="observer-side_overwatch-meta" class="preview-meta">waiting</span></div>
            <img id="observer-side_overwatch" class="preview-frame observer-frame" alt="side_overwatch observer preview" />
            <div id="observer-side_overwatch-placeholder" class="placeholder">side_overwatch is waiting for its first frame.</div>
          </div>
          <div class="preview-tile">
            <div class="preview-heading"><strong>top_oblique</strong><span id="observer-top_oblique-meta" class="preview-meta">waiting</span></div>
            <img id="observer-top_oblique" class="preview-frame observer-frame" alt="top_oblique observer preview" />
            <div id="observer-top_oblique-placeholder" class="placeholder">top_oblique is waiting for its first frame.</div>
          </div>
        </div>
      </section>

      <section class="section">
        <h2>Session</h2>
        <div class="controls">
          <label>Control Mode
            <select id="control-mode-select"></select>
          </label>
          <label>IMU Noise
            <select id="imu-noise-select"></select>
          </label>
          <label>Vision Noise
            <select id="vision-noise-select"></select>
          </label>
          <label>Actuation Noise
            <select id="actuation-noise-select"></select>
          </label>
        </div>
        <div class="button-row three">
          <button id="launch-button">Launch Session</button>
          <button id="reset-button" class="secondary">Reset Session</button>
          <button id="home-button" class="secondary">Home</button>
        </div>
        <div class="button-row">
          <button id="record-button" class="secondary">Toggle Recording</button>
          <button id="refresh-button" class="secondary">Refresh Snapshot</button>
        </div>
      </section>

      <section class="section">
        <h2>Manual Joint Control</h2>
        <div class="small">The tabletop auto-demo is the default mode, but we still keep manual joint and task controls for inspection and recovery.</div>
        <div id="joint-grid" class="joint-grid"></div>
        <div class="button-row">
          <button id="apply-joints-button">Apply Joint Targets</button>
          <button id="manual-joints-button" class="secondary">Switch To Joint Mode</button>
        </div>
      </section>

      <section class="section">
        <h2>Manual Task Nudges</h2>
        <div class="button-row">
          <button id="manual-task-button" class="secondary">Switch To Task Mode</button>
          <button id="auto-demo-button" class="secondary">Return To Auto Demo</button>
        </div>
        <div class="button-row three">
          <button data-delta="0.03,0,0">+X</button>
          <button data-delta="0,0.03,0">+Y</button>
          <button data-delta="0,0,0.03">+Z</button>
          <button data-delta="-0.03,0,0">-X</button>
          <button data-delta="0,-0.03,0">-Y</button>
          <button data-delta="0,0,-0.03">-Z</button>
        </div>
      </section>
    </div>

    <div class="panel">
      <section class="section" style="margin-top: 0;">
        <h2>Status</h2>
        <div class="metric-grid">
          <div class="metric"><div class="label">Profile</div><div id="metric-profile" class="value">{default_profile_key}</div></div>
          <div class="metric"><div class="label">Streaming</div><div id="metric-backend" class="value">webrtc</div></div>
          <div class="metric"><div class="label">Detector</div><div id="metric-detector-backend" class="value">new_pupil</div></div>
          <div class="metric"><div class="label">Anchor Visible</div><div id="anchor-visible" class="value">no</div></div>
          <div class="metric"><div class="label">Innovation Norm</div><div id="innovation-norm" class="value">n/a</div></div>
          <div class="metric"><div class="label">Tracking Error</div><div id="tracking-error" class="value">n/a</div></div>
          <div class="metric"><div class="label">Measured Accel</div><div id="imu-accel" class="value">[0, 0, 0]</div></div>
          <div class="metric"><div class="label">Measured Gyro</div><div id="imu-gyro" class="value">[0, 0, 0]</div></div>
          <div class="metric"><div class="label">Detections</div><div id="detector-count" class="value">0</div></div>
        </div>
        <div id="estimation-summary" class="small" style="margin-top: 12px;"></div>
        <pre id="connection-info">{{}}</pre>
      </section>

      <section class="section">
        <h2>Patterns</h2>
        <pre id="pattern-list">[]</pre>
      </section>

      <section class="section">
        <h2>Recording</h2>
        <pre id="recording-summary">{{}}</pre>
      </section>
    </div>
  </div>

  <script>
    const state = {{
      socket: null,
      latestSnapshot: null,
      recordingEnabled: false,
    }};

    const controlModeSelect = document.getElementById("control-mode-select");
    const imuNoiseSelect = document.getElementById("imu-noise-select");
    const visionNoiseSelect = document.getElementById("vision-noise-select");
    const actuationNoiseSelect = document.getElementById("actuation-noise-select");
    const jointGrid = document.getElementById("joint-grid");

    function fillSelect(select, options, value) {{
      select.innerHTML = "";
      options.forEach((item) => {{
        const option = document.createElement("option");
        option.value = item.value;
        option.textContent = item.label;
        select.appendChild(option);
      }});
      select.value = options.some((item) => item.value === value) ? value : options[0].value;
    }}

    function sendMessage(payload) {{
      if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {{
        return;
      }}
      state.socket.send(JSON.stringify(payload));
    }}

    function formatVector(values) {{
      if (!Array.isArray(values)) return "n/a";
      return `[${{values.map((value) => Number(value || 0).toFixed(3)).join(", ")}}]`;
    }}

    function applyPreview(imgId, placeholderId, metaId, frame) {{
      const image = document.getElementById(imgId);
      const placeholder = document.getElementById(placeholderId);
      const meta = document.getElementById(metaId);
      if (!frame || !frame.image_data_url) {{
        image.removeAttribute("src");
        placeholder.style.display = "block";
        meta.textContent = "waiting";
        return;
      }}
      image.src = frame.image_data_url;
      placeholder.style.display = "none";
      const frameIndex = frame.frame_index == null ? "n/a" : frame.frame_index;
      const timestamp = frame.timestamp_s == null ? "n/a" : Number(frame.timestamp_s).toFixed(3);
      meta.textContent = `frame ${{frameIndex}} @ ${{timestamp}} s`;
    }}

    function renderJointGrid(joints) {{
      jointGrid.innerHTML = "";
      (joints || []).forEach((joint, index) => {{
        const label = document.createElement("label");
        label.textContent = joint.name || `joint_${{index + 1}}`;
        const input = document.createElement("input");
        input.type = "number";
        input.step = "0.1";
        input.value = Number(joint.position_deg || 0).toFixed(2);
        input.dataset.jointIndex = String(index);
        label.appendChild(input);
        jointGrid.appendChild(label);
      }});
    }}

    function renderSnapshot(snapshot) {{
      state.latestSnapshot = snapshot;
      state.recordingEnabled = Boolean(snapshot.recording?.enabled);
      const sessionState = snapshot.session?.state || "idle";
      document.getElementById("session-state-text").textContent = sessionState;
      document.getElementById("session-dot").classList.toggle("live", sessionState === "live");
      document.getElementById("metric-profile").textContent = snapshot.session?.profile_key || "{default_profile_key}";
      document.getElementById("metric-backend").textContent = snapshot.session?.streaming_backend || "webrtc";
      document.getElementById("metric-detector-backend").textContent = snapshot.detections?.backend || "new_pupil";
      document.getElementById("anchor-visible").textContent = snapshot.estimation?.anchor_visible ? "yes" : "no";
      document.getElementById("innovation-norm").textContent = snapshot.estimation?.innovation_norm == null ? "n/a" : Number(snapshot.estimation.innovation_norm).toFixed(4);
      document.getElementById("tracking-error").textContent = snapshot.control?.tracking_error_norm_m == null ? "n/a" : Number(snapshot.control.tracking_error_norm_m).toFixed(4);
      document.getElementById("imu-accel").textContent = formatVector(snapshot.imu?.measured?.accel_mps2);
      document.getElementById("imu-gyro").textContent = formatVector(snapshot.imu?.measured?.gyro_rps);
      document.getElementById("detector-count").textContent = String(snapshot.detections?.count || 0);
      document.getElementById("estimation-summary").textContent = snapshot.estimation?.summary || "";
      document.getElementById("connection-info").textContent = JSON.stringify(snapshot.streaming?.connection_info || {{}}, null, 2);
      document.getElementById("pattern-list").textContent = JSON.stringify(snapshot.detections?.pattern_locations || [], null, 2);
      document.getElementById("recording-summary").textContent = JSON.stringify(snapshot.recording || {{}}, null, 2);

      controlModeSelect.value = snapshot.control?.control_mode || controlModeSelect.value;
      imuNoiseSelect.value = snapshot.noise?.imu_mode || imuNoiseSelect.value;
      visionNoiseSelect.value = snapshot.noise?.vision_mode || visionNoiseSelect.value;
      actuationNoiseSelect.value = snapshot.noise?.actuation_mode || actuationNoiseSelect.value;

      renderJointGrid(snapshot.robot?.arm_joints || []);
      applyPreview("mounted-preview", "mounted-preview-placeholder", "mounted-preview-meta", snapshot.frames?.primary);
      const observers = {{}};
      (snapshot.frames?.observers || []).forEach((item) => {{
        observers[item.name] = item;
      }});
      ["room_wide", "side_overwatch", "top_oblique"].forEach((name) => {{
        applyPreview(`observer-${{name}}`, `observer-${{name}}-placeholder`, `observer-${{name}}-meta`, observers[name]);
      }});
      document.getElementById("record-button").textContent = state.recordingEnabled ? "Stop Recording" : "Start Recording";
    }}

    async function loadInitialState() {{
      fillSelect(controlModeSelect, [
        {{ value: "auto_demo", label: "Auto Demo" }},
        {{ value: "manual_joint", label: "Manual Joint" }},
        {{ value: "manual_task", label: "Manual Task" }},
      ], "auto_demo");
      fillSelect(imuNoiseSelect, [
        {{ value: "ideal", label: "Ideal" }},
        {{ value: "nominal_phone", label: "Nominal Phone" }},
        {{ value: "stress_phone", label: "Stress Phone" }},
      ], "ideal");
      fillSelect(visionNoiseSelect, [
        {{ value: "clean", label: "Clean" }},
        {{ value: "nominal", label: "Nominal" }},
      ], "clean");
      fillSelect(actuationNoiseSelect, [
        {{ value: "none", label: "None" }},
        {{ value: "servo_nominal", label: "Servo Nominal" }},
        {{ value: "servo_stress", label: "Servo Stress" }},
      ], "none");
      const response = await fetch("/v1/session");
      const payload = await response.json();
      renderSnapshot(payload.snapshot);
    }}

    function connectSocket() {{
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${{protocol}}://${{window.location.host}}/ws/live`);
      state.socket = socket;
      socket.addEventListener("message", (event) => {{
        renderSnapshot(JSON.parse(event.data));
      }});
      socket.addEventListener("close", () => {{
        setTimeout(connectSocket, 1000);
      }});
    }}

    document.getElementById("launch-button").addEventListener("click", () => {{
      sendMessage({{ type: "launch_session", control_mode: controlModeSelect.value }});
    }});
    document.getElementById("reset-button").addEventListener("click", () => {{
      sendMessage({{ type: "reset_session" }});
    }});
    document.getElementById("home-button").addEventListener("click", () => {{
      sendMessage({{ type: "home" }});
    }});
    document.getElementById("record-button").addEventListener("click", () => {{
      sendMessage({{ type: "set_recording", enabled: !state.recordingEnabled }});
    }});
    document.getElementById("refresh-button").addEventListener("click", () => {{
      sendMessage({{ type: "refresh" }});
    }});
    document.getElementById("manual-joints-button").addEventListener("click", () => {{
      sendMessage({{ type: "set_control_mode", control_mode: "manual_joint" }});
    }});
    document.getElementById("manual-task-button").addEventListener("click", () => {{
      sendMessage({{ type: "set_control_mode", control_mode: "manual_task" }});
    }});
    document.getElementById("auto-demo-button").addEventListener("click", () => {{
      sendMessage({{ type: "set_control_mode", control_mode: "auto_demo" }});
    }});
    document.getElementById("apply-joints-button").addEventListener("click", () => {{
      const targets = Array.from(jointGrid.querySelectorAll("input[data-joint-index]")).map((input) => Number(input.value || 0));
      sendMessage({{ type: "set_control_mode", control_mode: "manual_joint" }});
      sendMessage({{ type: "set_joint_targets", joint_targets_deg: targets }});
    }});
    Array.from(document.querySelectorAll("button[data-delta]")).forEach((button) => {{
      button.addEventListener("click", () => {{
        const delta = button.dataset.delta.split(",").map((value) => Number(value));
        sendMessage({{ type: "nudge_task_target", delta_world_m: delta }});
      }});
    }});
    controlModeSelect.addEventListener("change", () => {{
      sendMessage({{ type: "set_control_mode", control_mode: controlModeSelect.value }});
    }});
    imuNoiseSelect.addEventListener("change", () => {{
      sendMessage({{ type: "set_noise_modes", imu_mode: imuNoiseSelect.value }});
    }});
    visionNoiseSelect.addEventListener("change", () => {{
      sendMessage({{ type: "set_noise_modes", vision_mode: visionNoiseSelect.value }});
    }});
    actuationNoiseSelect.addEventListener("change", () => {{
      sendMessage({{ type: "set_noise_modes", actuation_mode: actuationNoiseSelect.value }});
    }});

    loadInitialState().then(connectSocket);
  </script>
</body>
</html>
"""


def isaac_standard_dashboard_html(default_profile_key: str, default_profile_label: str) -> str:
    rendered = (
        _PAGE_TEMPLATE.replace("{default_profile_key}", html.escape(str(default_profile_key))).replace(
            "{default_profile_label}",
            html.escape(str(default_profile_label)),
        )
    )
    return rendered.replace("{{", "{").replace("}}", "}")


__all__ = ["isaac_standard_dashboard_html"]
