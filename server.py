from flask import Flask
from flask.templating import render_template
from db.mysql_conn import MysqlDb

app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT Rv]]jkojkiyd'

@app.route("/", methods=['POST', 'GET'])
def index():
    mysql_db = MysqlDb()
    data = mysql_db.get_data()
    return render_template('index.html', data=data)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', debug=True)