from flask import Flask
from routes.place import place_api
from routes.restaurant import restaurant_api
from routes.tour import tour_api

app = Flask(__name__)

app.register_blueprint(place_api)
app.register_blueprint(restaurant_api)
app.register_blueprint(tour_api)


if __name__=='__main__':
    app.run(debug=True, port=8080)