"""Browser UI for the interactive simulator."""

from __future__ import annotations

import html


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Calibration Sim Control Deck</title>
  <style>
    :root {
      --bg: #09111a;
      --panel: rgba(13, 23, 34, 0.84);
      --panel-strong: rgba(18, 32, 47, 0.95);
      --line: rgba(120, 167, 197, 0.28);
      --text: #ecf3f8;
      --muted: #91a8b9;
      --accent: #20d8ff;
      --accent-2: #ffbc3b;
      --success: #12ea79;
      --danger: #ff6d4d;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(32, 216, 255, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 188, 59, 0.16), transparent 24%),
        linear-gradient(180deg, #0f1a27 0%, #09111a 55%, #050b12 100%);
    }
    .shell {
      width: min(1500px, calc(100vw - 24px));
      margin: 12px auto;
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.9fr);
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .left-column {
      display: grid;
      grid-template-rows: auto auto;
      gap: 14px;
    }
    .hero {
      padding: 18px;
    }
    .titlebar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.5rem, 2.6vw, 2.4rem);
      letter-spacing: 0.03em;
    }
    .subtitle {
      color: var(--muted);
      max-width: 50ch;
      font-size: 0.95rem;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(21, 42, 61, 0.95);
      border: 1px solid rgba(67, 132, 173, 0.36);
      font-size: 0.92rem;
    }
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
      box-shadow: 0 0 0 4px rgba(255, 109, 77, 0.18);
    }
    .status-dot.live {
      background: var(--success);
      box-shadow: 0 0 0 4px rgba(18, 234, 121, 0.18);
    }
    .main-view {
      position: relative;
      overflow: hidden;
      border-radius: 18px;
      border: 1px solid rgba(146, 188, 214, 0.16);
      background: rgba(4, 10, 18, 0.75);
      min-height: 300px;
    }
    .main-view img {
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }
    .view-hud {
      position: absolute;
      left: 16px;
      right: 16px;
      bottom: 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
      font-size: 0.88rem;
    }
    .hud-box {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(4, 12, 20, 0.72);
      border: 1px solid rgba(119, 160, 187, 0.18);
    }
    .observer-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      padding: 18px;
    }
    .observer-card {
      overflow: hidden;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid rgba(124, 169, 199, 0.16);
    }
    .observer-card img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }
    .observer-meta {
      padding: 12px 14px 14px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 0.88rem;
    }
    .sidebar {
      padding: 18px;
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .section {
      padding: 16px;
      border-radius: 18px;
      background: rgba(7, 17, 27, 0.72);
      border: 1px solid rgba(116, 158, 187, 0.14);
    }
    .section h2 {
      margin: 0 0 12px;
      font-size: 1rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .row:last-child { margin-bottom: 0; }
    input[type="text"] {
      flex: 1;
      min-width: 0;
      border: 1px solid rgba(111, 157, 188, 0.18);
      border-radius: 12px;
      background: rgba(5, 12, 20, 0.92);
      color: var(--text);
      padding: 11px 12px;
      font: inherit;
    }
    select {
      flex: 1;
      min-width: 0;
      border: 1px solid rgba(111, 157, 188, 0.18);
      border-radius: 12px;
      background: rgba(5, 12, 20, 0.92);
      color: var(--text);
      padding: 11px 12px;
      font: inherit;
    }
    button {
      border: 0;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), #4de3c4);
      color: #021018;
      font: inherit;
      font-weight: 700;
      padding: 11px 14px;
      cursor: pointer;
    }
    button.secondary {
      background: linear-gradient(135deg, var(--accent-2), #ffd978);
    }
    .toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(112, 155, 184, 0.16);
      background: rgba(4, 12, 20, 0.8);
    }
    .toggle input {
      width: 20px;
      height: 20px;
      accent-color: var(--success);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(8, 18, 30, 0.86);
      border: 1px solid rgba(116, 158, 187, 0.16);
    }
    .metric .label {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .metric .value {
      font-size: 1rem;
      margin-top: 6px;
      font-weight: 700;
    }
    .servo {
      margin-bottom: 12px;
    }
    .servo:last-child {
      margin-bottom: 0;
    }
    .servo-label {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .small {
      font-size: 0.86rem;
      color: var(--muted);
      line-height: 1.45;
    }
    .detection-list {
      display: grid;
      gap: 10px;
      max-height: 260px;
      overflow: auto;
    }
    .detection-card {
      padding: 12px;
      border-radius: 14px;
      background: rgba(8, 18, 30, 0.86);
      border: 1px solid rgba(116, 158, 187, 0.16);
    }
    .detection-card strong {
      color: var(--success);
    }
    .kbd-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .kbd-box {
      padding: 12px;
      border-radius: 14px;
      background: rgba(8, 18, 30, 0.86);
      border: 1px solid rgba(116, 158, 187, 0.16);
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.5;
    }
    code {
      color: var(--accent-2);
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.9em;
    }
    @media (max-width: 1080px) {
      .shell {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="left-column">
      <div class="panel hero">
        <div class="titlebar">
          <div>
            <h1>Calibration Sim Control Deck</h1>
            <div class="subtitle">Drive the wrist servos, monitor IMU readings, inspect live detections, and review observer cameras from config.</div>
          </div>
          <div class="status-pill"><span id="status-dot" class="status-dot"></span><span id="status-text">connecting</span></div>
        </div>
        <div class="main-view">
          <img id="phone-view" alt="Primary phone camera view">
          <div class="view-hud">
            <div class="hud-box" id="hud-time">t = 0.00 s</div>
            <div class="hud-box" id="hud-run-dir">recording: idle</div>
          </div>
        </div>
      </div>
      <div class="panel observer-grid" id="observer-grid"></div>
    </section>
    <aside class="panel sidebar">
      <div class="section">
        <h2>Session</h2>
        <div class="row">
          <select id="scene-select"></select>
        </div>
        <div class="row">
          <select id="robot-arm-select"></select>
        </div>
        <div class="row">
          <input type="text" id="config-path" value="__DEFAULT_CONFIG__" />
          <button id="reload-config">Reload</button>
        </div>
        <div class="toggle">
          <div>
            <div>Record output dump</div>
            <div class="small">Save annotated camera frames and per-tick sensor + detection logs.</div>
          </div>
          <input type="checkbox" id="record-toggle">
        </div>
        <div class="toggle">
          <div>
            <div>Auto demo motion</div>
            <div class="small">Run a repeatable camera sweep so we can capture and analyze a full demo without manual input.</div>
          </div>
          <input type="checkbox" id="auto-demo-toggle">
        </div>
        <div class="row">
          <button id="run-analysis" class="secondary">Analyze Last Run</button>
        </div>
        <div class="small" id="config-summary">loading config summary...</div>
      </div>
      <div class="section">
        <h2>IMU</h2>
        <div class="metric-grid">
          <div class="metric"><div class="label">Accel X</div><div class="value" id="accel-x">0.000</div></div>
          <div class="metric"><div class="label">Accel Y</div><div class="value" id="accel-y">0.000</div></div>
          <div class="metric"><div class="label">Accel Z</div><div class="value" id="accel-z">0.000</div></div>
          <div class="metric"><div class="label">Gyro X</div><div class="value" id="gyro-x">0.0000</div></div>
          <div class="metric"><div class="label">Gyro Y</div><div class="value" id="gyro-y">0.0000</div></div>
          <div class="metric"><div class="label">Gyro Z</div><div class="value" id="gyro-z">0.0000</div></div>
        </div>
      </div>
      <div class="section">
        <h2>Servos</h2>
        <div class="servo">
          <div class="servo-label"><span>Base yaw</span><span id="servo-0-value">0.0 deg</span></div>
          <input id="servo-0" type="range" min="-45" max="45" step="1" value="0">
        </div>
        <div class="servo">
          <div class="servo-label"><span>Shoulder pitch</span><span id="servo-1-value">0.0 deg</span></div>
          <input id="servo-1" type="range" min="-10" max="78" step="1" value="28">
        </div>
        <div class="servo">
          <div class="servo-label"><span>Elbow pitch</span><span id="servo-2-value">0.0 deg</span></div>
          <input id="servo-2" type="range" min="-65" max="65" step="1" value="-18">
        </div>
        <div class="small">Keyboard: <code>Left / Right</code> yaw, <code>Up / Down</code> shoulder, <code>Q / A</code> elbow.</div>
      </div>
      <div class="section">
        <h2>Detections</h2>
        <div class="detection-list" id="detection-list"></div>
      </div>
      <div class="section">
        <h2>Analysis</h2>
        <div class="small" id="analysis-summary">No analysis has run yet.</div>
      </div>
      <div class="section">
        <h2>Quick Help</h2>
        <div class="kbd-grid">
          <div class="kbd-box">Primary view is the phone camera with live AprilTag detection overlay and 5-point visualization.</div>
          <div class="kbd-box">Observer panels come from config, so adding more cameras is just a YAML edit plus Reload.</div>
          <div class="kbd-box">Recorded dumps land under <code>output/interactive_runs</code> with raw video, IMU CSV, ground truth, and offline analysis artifacts.</div>
          <div class="kbd-box">Stopping a recording can auto-run undistortion, detection, and JAX pose estimation against ground truth.</div>
        </div>
      </div>
    </aside>
  </div>
  <script>
    const state = {
      socket: null,
      targets: [0, 28, -18],
      reconnectTimer: null,
    };

    const observerGrid = document.getElementById("observer-grid");
    const detectionList = document.getElementById("detection-list");
    const recordToggle = document.getElementById("record-toggle");
    const autoDemoToggle = document.getElementById("auto-demo-toggle");
    const configPath = document.getElementById("config-path");
    const sceneSelect = document.getElementById("scene-select");
    const robotArmSelect = document.getElementById("robot-arm-select");
    const servoInputs = [0, 1, 2].map((index) => document.getElementById(`servo-${index}`));
    const servoValueNodes = [0, 1, 2].map((index) => document.getElementById(`servo-${index}-value`));
    let selectorsPopulated = false;

    function setConnectionState(live, label) {
      document.getElementById("status-text").textContent = label;
      document.getElementById("status-dot").classList.toggle("live", live);
    }

    function sendMessage(payload) {
      if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
      state.socket.send(JSON.stringify(payload));
    }

    function syncServoLabels() {
      servoValueNodes.forEach((node, index) => {
        node.textContent = `${Number(servoInputs[index].value).toFixed(1)} deg`;
      });
    }

    function pushServoTargets() {
      state.targets = servoInputs.map((input) => Number(input.value));
      syncServoLabels();
      sendMessage({ type: "set_servos", targets_deg: state.targets });
    }

    function renderObservers(observerViews) {
      observerGrid.innerHTML = "";
      observerViews.forEach((view) => {
        const card = document.createElement("div");
        card.className = "observer-card";
        card.innerHTML = `
          <img src="${view.image_data_url}" alt="${view.name}">
          <div class="observer-meta">
            <strong>${view.name}</strong>
            <span>observer camera</span>
          </div>
        `;
        observerGrid.appendChild(card);
      });
    }

    function renderSelectors(snapshot) {
      const catalog = snapshot.catalog || {};
      const scenePresets = catalog.scene_presets || [];
      const robotArmPresets = catalog.robot_arm_presets || [];
      if (!selectorsPopulated || sceneSelect.options.length !== scenePresets.length) {
        sceneSelect.innerHTML = scenePresets
          .map((item) => `<option value="${item.config_path}">${item.label || item.name || item.config_path}</option>`)
          .join("");
      }
      if (!selectorsPopulated || robotArmSelect.options.length !== robotArmPresets.length) {
        robotArmSelect.innerHTML = robotArmPresets
          .map((item) => `<option value="${item.preset_path}">${item.label || item.name || item.preset_path}</option>`)
          .join("");
      }
      if (catalog.selected_scene_config_path) {
        sceneSelect.value = catalog.selected_scene_config_path;
        configPath.value = catalog.selected_scene_config_path;
      }
      if (catalog.selected_robot_arm_preset_path) {
        robotArmSelect.value = catalog.selected_robot_arm_preset_path;
      }
      selectorsPopulated = true;
    }

    function renderDetections(detections) {
      detectionList.innerHTML = "";
      if (!detections.length) {
        detectionList.innerHTML = `<div class="detection-card">No AprilTags visible in the phone camera right now.</div>`;
        return;
      }

      detections.forEach((item) => {
        const points = item.points5_xy
          .map((point, index) => `p${index}: (${point[0].toFixed(1)}, ${point[1].toFixed(1)})`)
          .join("<br>");
        const pose = item.pose_camera_tvec
          ? `pose z: ${item.pose_camera_tvec[2].toFixed(3)} m`
          : "pose unavailable";
        const node = document.createElement("div");
        node.className = "detection-card";
        node.innerHTML = `
          <div><strong>tag ${item.tag_id}</strong> <span class="small">${item.family}</span></div>
          <div class="small">${pose}</div>
          <div class="small">${points}</div>
        `;
        detectionList.appendChild(node);
      });
    }

    function renderSnapshot(snapshot) {
      renderSelectors(snapshot);
      document.getElementById("phone-view").src = snapshot.phone_view.image_data_url;
      document.getElementById("hud-time").textContent = `t = ${snapshot.sim_time_s.toFixed(2)} s`;
      const runDir = snapshot.recording.active_run_dir || snapshot.recording.last_run_dir || "idle";
      document.getElementById("hud-run-dir").textContent = `recording: ${runDir}`;
      document.getElementById("config-summary").textContent =
        `${snapshot.config.name} | ${snapshot.config.config_path} | arm: ${snapshot.config.robot_arm.label} | ${snapshot.phone_view.camera_model.name} | ${snapshot.config.observer_cameras.length} observer cams | tags: ${snapshot.config.tags.map((tag) => tag.tag_id).join(", ")}`;

      ["x", "y", "z"].forEach((axis, index) => {
        document.getElementById(`accel-${axis}`).textContent = snapshot.imu.accel_mps2[index].toFixed(3);
        document.getElementById(`gyro-${axis}`).textContent = snapshot.imu.gyro_rps[index].toFixed(4);
      });

      const limits = snapshot.config.robot_arm.servo_limits_deg || [];
      servoInputs.forEach((input, index) => {
        if (limits[index]) {
          input.min = limits[index][0];
          input.max = limits[index][1];
        }
      });

      servoInputs.forEach((input, index) => {
        if (document.activeElement !== input) {
          input.value = snapshot.servo_targets_deg[index];
        }
      });
      syncServoLabels();
      recordToggle.checked = Boolean(snapshot.recording.enabled);
      autoDemoToggle.checked = Boolean(snapshot.automation.auto_demo_enabled);
      renderDetections(snapshot.phone_view.detections);
      renderObservers(snapshot.observer_views);

      const analysis = snapshot.analysis || {};
      if (analysis.state === "completed" && analysis.summary) {
        const meanMm = analysis.summary.mean_position_error_m == null
          ? "n/a"
          : `${(analysis.summary.mean_position_error_m * 1000).toFixed(2)} mm`;
        document.getElementById("analysis-summary").textContent =
          `completed | run: ${analysis.last_run_dir} | mean pos err: ${meanMm} | report: ${analysis.report_path}`;
      } else if (analysis.state === "running") {
        document.getElementById("analysis-summary").textContent =
          `running on ${analysis.last_run_dir || "last run"}...`;
      } else if (analysis.state === "failed") {
        document.getElementById("analysis-summary").textContent =
          `failed | ${analysis.error || "unknown error"}`;
      } else {
        document.getElementById("analysis-summary").textContent = "No analysis has run yet.";
      }
    }

    function connect() {
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      state.socket = new WebSocket(`${protocol}://${location.host}/ws/live`);
      setConnectionState(false, "connecting");

      state.socket.addEventListener("open", () => {
        setConnectionState(true, "live");
        sendMessage({ type: "set_servos", targets_deg: state.targets });
      });

      state.socket.addEventListener("message", (event) => {
        const snapshot = JSON.parse(event.data);
        renderSnapshot(snapshot);
      });

      state.socket.addEventListener("close", () => {
        setConnectionState(false, "reconnecting");
        window.clearTimeout(state.reconnectTimer);
        state.reconnectTimer = window.setTimeout(connect, 1200);
      });
    }

    servoInputs.forEach((input) => {
      input.addEventListener("input", pushServoTargets);
    });
    syncServoLabels();

    document.getElementById("reload-config").addEventListener("click", () => {
      sendMessage({ type: "reload_config", config_path: configPath.value.trim() });
    });

    sceneSelect.addEventListener("change", () => {
      sendMessage({ type: "set_scene_preset", config_path: sceneSelect.value });
    });

    robotArmSelect.addEventListener("change", () => {
      sendMessage({ type: "set_robot_arm_preset", preset_path: robotArmSelect.value });
    });

    recordToggle.addEventListener("change", () => {
      sendMessage({ type: "set_recording", enabled: recordToggle.checked });
    });

    autoDemoToggle.addEventListener("change", () => {
      sendMessage({ type: "set_auto_demo", enabled: autoDemoToggle.checked });
    });

    document.getElementById("run-analysis").addEventListener("click", () => {
      sendMessage({ type: "run_analysis" });
    });

    window.addEventListener("keydown", (event) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
      const delta = [0, 0, 0];
      if (event.key === "ArrowLeft") delta[0] = -2;
      if (event.key === "ArrowRight") delta[0] = 2;
      if (event.key === "ArrowUp") delta[1] = 2;
      if (event.key === "ArrowDown") delta[1] = -2;
      if (event.key === "q" || event.key === "Q") delta[2] = 2;
      if (event.key === "a" || event.key === "A") delta[2] = -2;
      if (delta.every((value) => value === 0)) return;
      event.preventDefault();
      servoInputs.forEach((input, index) => {
        input.value = Number(input.value) + delta[index];
      });
      pushServoTargets();
    });

    connect();
  </script>
</body>
</html>
"""


def interactive_dashboard_html(default_config_path: str) -> str:
    return _PAGE_TEMPLATE.replace("__DEFAULT_CONFIG__", html.escape(default_config_path))
