const API = ""; // same-origin: FastAPI serves this frontend itself

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const footStep = document.getElementById("footStep");
const footDevice = document.getElementById("footDevice");

const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzoneText");
const fileInput = document.getElementById("fileInput");
const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const clipPreview = document.getElementById("clipPreview");
const clipName = document.getElementById("clipName");
const clipAudio = document.getElementById("clipAudio");
const clearClip = document.getElementById("clearClip");

const windowSlider = document.getElementById("windowSlider");
const windowVal = document.getElementById("windowVal");
const durationSlider = document.getElementById("durationSlider");
const durationVal = document.getElementById("durationVal");

const generateBtn = document.getElementById("generateBtn");
const progressCard = document.getElementById("progressCard");
const progressFill = document.getElementById("progressFill");
const progressPct = document.getElementById("progressPct");
const resultsCard = document.getElementById("resultsCard");
const inputPlayer = document.getElementById("inputPlayer");
const outputPlayer = document.getElementById("outputPlayer");
const groundTruthResult = document.getElementById("groundTruthResult");
const groundTruthPlayer = document.getElementById("groundTruthPlayer");
const errorCard = document.getElementById("errorCard");
const errorText = document.getElementById("errorText");

const modeSelfFeedback = document.getElementById("modeSelfFeedback");
const modeTeacherForced = document.getElementById("modeTeacherForced");
const modeHelp = document.getElementById("modeHelp");
const step2Title = document.getElementById("step2Title");
const selfFeedbackInputs = document.getElementById("selfFeedbackInputs");
const teacherForcedInputs = document.getElementById("teacherForcedInputs");
const settingsCard = document.getElementById("settingsCard");

const dropzoneUser = document.getElementById("dropzoneUser");
const fileInputUser = document.getElementById("fileInputUser");
const clipPreviewUser = document.getElementById("clipPreviewUser");
const clipNameUser = document.getElementById("clipNameUser");
const clipAudioUser = document.getElementById("clipAudioUser");
const clearClipUser = document.getElementById("clearClipUser");

const dropzoneAgent = document.getElementById("dropzoneAgent");
const fileInputAgent = document.getElementById("fileInputAgent");
const clipPreviewAgent = document.getElementById("clipPreviewAgent");
const clipNameAgent = document.getElementById("clipNameAgent");
const clipAudioAgent = document.getElementById("clipAudioAgent");
const clearClipAgent = document.getElementById("clearClipAgent");

const sourceUpload = document.getElementById("sourceUpload");
const sourceDrive = document.getElementById("sourceDrive");
const teacherForcedUploadInputs = document.getElementById("teacherForcedUploadInputs");
const teacherForcedDriveInputs = document.getElementById("teacherForcedDriveInputs");
const driveUserLink = document.getElementById("driveUserLink");
const driveAgentLink = document.getElementById("driveAgentLink");
const driveOffsetSlider = document.getElementById("driveOffsetSlider");
const driveOffsetVal = document.getElementById("driveOffsetVal");
const driveDurationSlider = document.getElementById("driveDurationSlider");
const driveDurationVal = document.getElementById("driveDurationVal");
const progressStage = document.getElementById("progressStage");

const MODE_HELP = {
  self_feedback:
    "The model generates its response window-by-window, feeding its own output back in as it goes — " +
    "this is the only mode that reflects how the model would actually be used.",
  teacher_forced:
    "Feeds the model the REAL agent audio (shifted one frame) instead of its own guess — not a real " +
    "usage mode, but useful for seeing what this checkpoint can do when it isn't compounding its own errors.",
};

let currentMode = "self_feedback";
let teacherForcedSource = "upload"; // "upload" | "drive"
let currentClipBlob = null;
let currentClipName = "clip.wav";
let userClipBlob = null;
let userClipName = "user.wav";
let agentClipBlob = null;
let agentClipName = "agent.wav";
let mediaRecorder = null;
let recordedChunks = [];

// ---- mode switching ----
function setMode(mode) {
  currentMode = mode;
  modeSelfFeedback.classList.toggle("active", mode === "self_feedback");
  modeTeacherForced.classList.toggle("active", mode === "teacher_forced");
  modeHelp.textContent = MODE_HELP[mode];
  selfFeedbackInputs.hidden = mode !== "self_feedback";
  teacherForcedInputs.hidden = mode !== "teacher_forced";
  settingsCard.hidden = mode === "teacher_forced"; // window/duration don't apply -- one-shot over the whole clip
  step2Title.textContent = mode === "self_feedback" ? "2. Provide a clip" : "2. Provide both clips";
  updateGenerateEnabled();
}
modeSelfFeedback.addEventListener("click", () => setMode("self_feedback"));
modeTeacherForced.addEventListener("click", () => setMode("teacher_forced"));

function setTeacherForcedSource(source) {
  teacherForcedSource = source;
  sourceUpload.classList.toggle("active", source === "upload");
  sourceDrive.classList.toggle("active", source === "drive");
  teacherForcedUploadInputs.hidden = source !== "upload";
  teacherForcedDriveInputs.hidden = source !== "drive";
  updateGenerateEnabled();
}
sourceUpload.addEventListener("click", () => setTeacherForcedSource("upload"));
sourceDrive.addEventListener("click", () => setTeacherForcedSource("drive"));

driveOffsetSlider.addEventListener("input", () => {
  driveOffsetVal.textContent = `${driveOffsetSlider.value}s`;
  updateGenerateEnabled();
});
driveDurationSlider.addEventListener("input", () => {
  driveDurationVal.textContent = `${driveDurationSlider.value}s`;
});
driveUserLink.addEventListener("input", updateGenerateEnabled);
driveAgentLink.addEventListener("input", updateGenerateEnabled);

function updateGenerateEnabled() {
  const serverReady = statusDot.className === "dot ok";
  let hasClip;
  if (currentMode === "self_feedback") {
    hasClip = !!currentClipBlob;
  } else if (teacherForcedSource === "upload") {
    hasClip = !!userClipBlob && !!agentClipBlob;
  } else {
    hasClip = driveUserLink.value.trim().length > 0 && driveAgentLink.value.trim().length > 0;
  }
  generateBtn.disabled = !(serverReady && hasClip);
}

// ---- health check ----
async function checkHealth() {
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    if (data.ready) {
      statusDot.className = "dot ok";
      statusText.textContent = `Model ready · step ${data.checkpoint_step}`;
      footStep.textContent = data.checkpoint_step;
      footDevice.textContent = data.device;
    } else {
      statusDot.className = "dot bad";
      statusText.textContent = `Checkpoint not found at ${data.checkpoint_dir}`;
    }
  } catch (e) {
    statusDot.className = "dot bad";
    statusText.textContent = "Can't reach the server";
  }
  updateGenerateEnabled();
}
checkHealth();
setInterval(checkHealth, 10000);
setMode("self_feedback");
setTeacherForcedSource("upload");

// ---- file picking / drag-drop ----
dropzone.addEventListener("click", (e) => {
  // label's default click->input behavior already handles this; no-op guard
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setClip(fileInput.files[0], fileInput.files[0].name);
});
["dragover", "dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => e.preventDefault());
});
dropzone.addEventListener("dragover", () => dropzone.classList.add("dragover"));
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) setClip(file, file.name);
});

function setClip(blob, name) {
  currentClipBlob = blob;
  currentClipName = name;
  clipName.textContent = name;
  clipAudio.src = URL.createObjectURL(blob);
  clipPreview.hidden = false;
  updateGenerateEnabled();
}

clearClip.addEventListener("click", () => {
  currentClipBlob = null;
  clipPreview.hidden = true;
  fileInput.value = "";
  updateGenerateEnabled();
});

// ---- teacher-forced dropzones (user + agent clips) ----
function wireDropzone(dropzoneEl, inputEl, onFile) {
  inputEl.addEventListener("change", () => {
    if (inputEl.files.length) onFile(inputEl.files[0], inputEl.files[0].name);
  });
  ["dragover", "dragleave", "drop"].forEach((evt) => dropzoneEl.addEventListener(evt, (e) => e.preventDefault()));
  dropzoneEl.addEventListener("dragover", () => dropzoneEl.classList.add("dragover"));
  dropzoneEl.addEventListener("dragleave", () => dropzoneEl.classList.remove("dragover"));
  dropzoneEl.addEventListener("drop", (e) => {
    dropzoneEl.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file, file.name);
  });
}

wireDropzone(dropzoneUser, fileInputUser, (blob, name) => {
  userClipBlob = blob;
  userClipName = name;
  clipNameUser.textContent = name;
  clipAudioUser.src = URL.createObjectURL(blob);
  clipPreviewUser.hidden = false;
  updateGenerateEnabled();
});
clearClipUser.addEventListener("click", () => {
  userClipBlob = null;
  clipPreviewUser.hidden = true;
  fileInputUser.value = "";
  updateGenerateEnabled();
});

wireDropzone(dropzoneAgent, fileInputAgent, (blob, name) => {
  agentClipBlob = blob;
  agentClipName = name;
  clipNameAgent.textContent = name;
  clipAudioAgent.src = URL.createObjectURL(blob);
  clipPreviewAgent.hidden = false;
  updateGenerateEnabled();
});
clearClipAgent.addEventListener("click", () => {
  agentClipBlob = null;
  clipPreviewAgent.hidden = true;
  fileInputAgent.value = "";
  updateGenerateEnabled();
});

// ---- mic recording ----
recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => recordedChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      recordBtn.classList.remove("recording");
      recordLabel.textContent = "Record from mic";

      // Browsers record as WebM/Opus, which the backend's torchaudio.load
      // often can't decode reliably -- decode it here and re-encode as a
      // plain WAV before upload, so the server always gets a format it
      // can trust regardless of browser/codec quirks.
      const webmBlob = new Blob(recordedChunks, { type: "audio/webm" });
      try {
        const wavBlob = await blobToWav(webmBlob);
        setClip(wavBlob, "recording.wav");
      } catch (e) {
        alert("Couldn't process the recording: " + e.message);
      }
    };
    mediaRecorder.start();
    recordBtn.classList.add("recording");
    recordLabel.textContent = "Stop recording";
  } catch (e) {
    alert("Couldn't access the microphone: " + e.message);
  }
});

// ---- settings ----
windowSlider.addEventListener("input", () => {
  windowVal.textContent = `${parseFloat(windowSlider.value).toFixed(1)}s`;
});
durationSlider.addEventListener("input", () => {
  durationVal.textContent = `${durationSlider.value}s`;
});

// ---- generation ----
generateBtn.addEventListener("click", async () => {
  errorCard.hidden = true;
  resultsCard.hidden = true;
  groundTruthResult.hidden = currentMode !== "teacher_forced";
  progressCard.hidden = false;
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";
  progressStage.textContent = "";
  generateBtn.disabled = true;
  document.getElementById("generateLabel").textContent = "Generating…";

  try {
    let job_id;
    if (currentMode === "self_feedback") {
      const form = new FormData();
      form.append("file", currentClipBlob, currentClipName);
      form.append("window_s", windowSlider.value);
      form.append("max_duration_s", durationSlider.value);
      const startRes = await fetch(`${API}/api/generate`, { method: "POST", body: form });
      if (!startRes.ok) throw new Error(await startRes.text());
      ({ job_id } = await startRes.json());
    } else if (teacherForcedSource === "upload") {
      const form = new FormData();
      form.append("user_file", userClipBlob, userClipName);
      form.append("agent_file", agentClipBlob, agentClipName);
      const startRes = await fetch(`${API}/api/generate_teacher_forced`, { method: "POST", body: form });
      if (!startRes.ok) throw new Error(await startRes.text());
      ({ job_id } = await startRes.json());
    } else {
      const startRes = await fetch(`${API}/api/generate_teacher_forced_from_drive`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_drive_link: driveUserLink.value.trim(),
          agent_drive_link: driveAgentLink.value.trim(),
          offset_s: parseFloat(driveOffsetSlider.value),
          duration_s: parseFloat(driveDurationSlider.value),
        }),
      });
      if (!startRes.ok) throw new Error(await startRes.text());
      ({ job_id } = await startRes.json());
    }

    await pollJob(job_id);
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    generateBtn.disabled = false;
    document.getElementById("generateLabel").textContent = "Generate";
  }
});

function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/jobs/${jobId}`);
        if (!res.ok) throw new Error(await res.text());
        const status = await res.json();

        const pct = Math.round(status.progress * 100);
        progressFill.style.width = `${pct}%`;
        progressPct.textContent = `${pct}%`;
        progressStage.textContent = status.stage || "";

        if (status.status === "done") {
          clearInterval(interval);
          progressCard.hidden = true;
          inputPlayer.src = `${API}/api/jobs/${jobId}/input_audio`;
          outputPlayer.src = `${API}/api/jobs/${jobId}/output_audio`;
          if (currentMode === "teacher_forced") {
            groundTruthPlayer.src = `${API}/api/jobs/${jobId}/ground_truth_audio`;
          }
          resultsCard.hidden = false;
          resolve();
        } else if (status.status === "error") {
          clearInterval(interval);
          progressCard.hidden = true;
          reject(new Error(status.error || "Generation failed"));
        }
      } catch (e) {
        clearInterval(interval);
        progressCard.hidden = true;
        reject(e);
      }
    }, 500);
  });
}

function showError(msg) {
  errorText.textContent = msg;
  errorCard.hidden = false;
}

// ---- WebM/Opus -> WAV conversion (decode in-browser, re-encode as PCM16 WAV) ----
async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const wavBuffer = encodeWav(audioBuffer);
  audioCtx.close();
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function encodeWav(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const numFrames = audioBuffer.length;

  // Downmix to mono by averaging channels -- the backend treats everything
  // as mono anyway, no point shipping stereo.
  const mono = new Float32Array(numFrames);
  for (let ch = 0; ch < numChannels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < numFrames; i++) mono[i] += data[i] / numChannels;
  }

  const bytesPerSample = 2; // 16-bit PCM
  const blockAlign = bytesPerSample;
  const dataSize = numFrames * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true); // byte rate
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    const sample = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}
