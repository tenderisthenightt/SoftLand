from flask import Flask, request, render_template

app = Flask(__name__)
@app.route('/')
def record():
    return render_template('record_test.html')

@app.route("/upload", methods=["POST"])
def upload():
    audio = request.files["audio"]
    audio.save("recorded-audio.wav")
    message = "Audio file uploaded successfully"
    return render_template('record_test.html', message = message)

if __name__ == "__main__":
    app.run()
