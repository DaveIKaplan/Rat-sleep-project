import json

from flask import Flask, request, Response


from kmeans import label_new_data

app = Flask(__name__)


@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        data = request.get_data()
        solution = label_new_data(data)
        return Response(response=json.dumps(solution),
                        mimetype='application/json',
                        status=200)
    except Exception as e:
        response = str(e)
        return Response(response=json.dumps(response),
                        mimetype='application/json',
                        status=400)


if __name__ == "__main__":
    app.run()
