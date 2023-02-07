from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def record():
    return render_template('record_test.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.data
    with open('./wav/output.wav', 'wb') as f:
        f.write(file)
    return 'File saved'

if __name__ == '__main__':
    app.run()
