import os
import uuid
from flask import Flask, request, jsonify
from agent import run_agent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gym Form Coach</title>
<style>
body{font-family:sans-serif;max-width:760px;margin:30px auto;padding:0 16px}
button{padding:10px 18px;font-size:16px}
.box{background:#f3f3f3;padding:12px;border-radius:8px;margin:10px 0}
img{max-width:100%} pre{white-space:pre-wrap;font-size:13px}
</style></head>
<body>
<h1>Gym Form Coach (Squat)</h1>
<p>Upload a side-view squat video: whole body in frame, 3 reps, steady phone.
This is a coaching aid, not medical advice.</p>
<input type="file" id="v" accept="video/*"> <button onclick="go()">Analyze</button>
<div id="out"></div>
<script>
async function go(){
  const f = document.getElementById('v').files[0];
  const out = document.getElementById('out');
  if(!f){ out.textContent = 'Choose a video first.'; return; }
  out.textContent = 'Analyzing... this can take a minute.';
  const fd = new FormData(); fd.append('video', f);
  try{
    const r = await fetch('/analyze', {method:'POST', body:fd});
    const d = await r.json();
    if(d.error){ out.textContent = d.error; return; }
    let h = '<div class="box"><b>Outcome:</b> ' + d.outcome + '<br>' + d.message + '</div>';
    if(d.feedback.length){
      h += '<div class="box"><b>Feedback</b><ul>' +
           d.feedback.map(x => '<li>' + x + '</li>').join('') + '</ul></div>';
    }
    if(d.snapshot){ h += '<p>Deepest point:</p><img src="' + d.snapshot + '">'; }
    h += '<p><a href="' + d.video + '">Download annotated video</a></p>';
    h += '<div class="box"><b>Agent trace</b><pre>' + JSON.stringify(d.trace, null, 2) + '</pre></div>';
    out.innerHTML = h;
  }catch(e){ out.textContent = 'Error: ' + e; }
}
</script></body></html>"""


@app.route("/")
def home():
    return PAGE


@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("video")
    if not f:
        return jsonify(error="No video uploaded"), 400
    uid = uuid.uuid4().hex[:8]
    ext = os.path.splitext(f.filename)[1] or ".mp4"
    path = os.path.join("uploads", uid + ext)
    f.save(path)
    snap = os.path.join("static", uid + ".jpg")
    vid = os.path.join("static", uid + ".mp4")
    result = run_agent(path, snapshot=snap, video_out=vid)
    result["snapshot"] = "/static/" + uid + ".jpg" if os.path.exists(snap) else None
    result["video"] = "/static/" + uid + ".mp4"
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)