/* =========================================================
   Reusable webcam controller.
   Wraps getUserMedia + a hidden canvas so any page can start a camera,
   grab a base64 JPEG frame, and stop the stream cleanly.
   ========================================================= */
class CameraController {
  constructor(videoEl, statusEl) {
    this.videoEl = videoEl;
    this.statusEl = statusEl;
    this.stream = null;
    this.canvas = document.createElement("canvas");
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();
      this._setStatus("Camera ready.", "good");
      return true;
    } catch (err) {
      let message = "Could not access the camera.";
      if (err.name === "NotAllowedError") message = "Camera permission denied. Please allow camera access.";
      if (err.name === "NotFoundError") message = "No camera found on this device.";
      if (err.name === "NotReadableError") message = "Camera is unavailable or already in use by another app.";
      this._setStatus(message, "bad");
      showToast(message, "error");
      return false;
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
  }

  /** Captures the current frame as a base64 JPEG (no data-URL prefix). */
  captureFrameBase64(quality = 0.85) {
    const { videoWidth, videoHeight } = this.videoEl;
    if (!videoWidth || !videoHeight) return null;
    this.canvas.width = videoWidth;
    this.canvas.height = videoHeight;
    const ctx = this.canvas.getContext("2d");
    ctx.drawImage(this.videoEl, 0, 0, videoWidth, videoHeight);
    return this.canvas.toDataURL("image/jpeg", quality);
  }

  _setStatus(text, level) {
    if (!this.statusEl) return;
    this.statusEl.querySelector(".status-text").textContent = text;
    const dot = this.statusEl.querySelector(".status-dot");
    dot.className = `status-dot ${level}`;
  }
}
