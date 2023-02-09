from flask import Flask, request, render_template
import base64
app = Flask(__name__)

@app.route('/')
def recorder():
    return render_template('record_test.html')

@app.route("/send-audio", methods=["POST"])
def send_audio():
    audio = request.files["audio"]
    data = audio.read()
    audioContents = base64.b64encode(data).decode("utf8")
    print(audioContents)
    # Do something with the audio data
    message = 'success'
    return render_template('record_test.html', message = message)

if __name__ == "__main__":
    app.run()


