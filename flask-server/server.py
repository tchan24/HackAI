from flask import Flask

app = Flask(__name__)

#Members API Routes
@app.route('/members')
def members():
    return {"members": ["Member1"]}

if __name__ == '__main__':
    app.run(debug=True)
