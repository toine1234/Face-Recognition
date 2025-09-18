// Upload file
document.getElementById("uploadForm").onsubmit = async (e) => {
  e.preventDefault();
  let formData = new FormData();
  formData.append("file", document.getElementById("file").files[0]);

  let res = await fetch("/upload", { method: "POST", body: formData });
  let data = await res.json();
  document.getElementById("result-upload").innerText =
    data.success ? "Kết quả: " + data.name : data.msg;
};

// Webcam
const video = document.getElementById("video");
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  video.srcObject = stream;
});

document.getElementById("capture").onclick = async () => {
  let canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  let ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  let dataUrl = canvas.toDataURL("image/jpeg");
  let res = await fetch("/webcam", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  });
  let data = await res.json();
  document.getElementById("result-webcam").innerText =
    data.success ? "Kết quả: " + data.name : data.msg;
};
