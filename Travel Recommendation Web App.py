from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load datasets
places = pd.read_csv('final_places.csv')
hotels = pd.read_csv('final_hotel.csv')

# Destination Recommendation Setup
places['combined'] = places['category'] + ' ' + places['city']
vectorizer = CountVectorizer()
place_vectors = vectorizer.fit_transform(places['combined'])

@app.route('/')
def home():
    return '''
    <html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel Recommendation System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .navbar {
            background-color: #333;
            color: white;
            padding: 15px;
            text-align: center;
            width: 100%;
            font-size: 1.5em;
            margin-bottom: 20px;
        }
        form {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            width: 350px;
            text-align: center;
        }
        input, select, button {
            width: 90%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #888;
            font-size: 1em;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            transition: 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #4CAF50;
        }
        button {
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            border: none;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="navbar">Travel Recommendation System</div>

    <h2>Destination Recommendation</h2>
    <form action="/destination" method="post">
        <select name="category" required>
            <option value="" disabled selected>Select a Category</option>
            <option value="Wildlife/Nature">Wildlife/Nature</option>
            <option value="Historical">Historical</option>
            <option value="Beaches">Beaches</option>
            <option value="Urban/Modern">Urban/Modern</option>
            <option value="Lakes/Rivers">Lakes/Rivers</option>
            <option value="Adventure">Adventure</option>
            <option value="Mountains">Mountains</option>
            <option value="Temple">Temple</option>
            <option value="Other">Other</option>
            <option value="Spiritual">Spiritual</option>
        </select>
        <button type="submit">Get Destination Recommendations</button>
    </form>

    <h2>Hotel Recommendation</h2>
    <form action="/hotel" method="post">
        <input type="text" name="hotel_city" placeholder="Enter hotel city" required>
        <input type="number" name="max_price" placeholder="Max price" required>
        <button type="submit">Get Hotel Recommendations</button>
    </form>
</body>
</html>
    '''

@app.route('/destination', methods=['POST'])
def destination_recommend():
    category = request.form.get('category')
    filtered_places = places[places['category'].str.strip().str.lower() == category.strip().lower()]

    if not filtered_places.empty:
        query_vector = vectorizer.transform([category])
        similarity_scores = cosine_similarity(query_vector, vectorizer.transform(filtered_places['combined']))[0]
        filtered_places['similarity'] = similarity_scores
        recommendations = filtered_places.sort_values(by='similarity', ascending=False).head(10)
        return '''<h2>Recommended Destinations:</h2>''' + recommendations[['places name', 'Rating', 'places description']].to_html(index=False) + '<br><a href="/">Back to Home</a>'
    else:
        return 'No destinations found. <br><a href="/">Back to Home</a>'

@app.route('/hotel', methods=['POST'])
def hotel_recommend():
    hotel_city = request.form.get('hotel_city')
    max_price = float(request.form.get('max_price'))

    scaler = MinMaxScaler()
    hotels['normalized_rating'] = scaler.fit_transform(hotels[['rating']])
    hotels['inverse_price'] = scaler.fit_transform(1 / hotels[['discount price']])
    hotels['score'] = 0.7 * hotels['normalized_rating'] + 0.3 * hotels['inverse_price']

    filtered_hotels = hotels[(hotels['destination name'].str.contains(hotel_city, case=False, na=False)) &
                             (hotels['discount price'] <= max_price)]

    if not filtered_hotels.empty:
        hotel_recommendations = filtered_hotels.sort_values(by='score', ascending=False).head(10)
        return '''<h2>Recommended Hotels:</h2>''' + hotel_recommendations[['hotel name', 'rating', 'actual price', 'discount price']].to_html(index=False) + '<br><a href="/">Back to Home</a>'
    else:
        return 'No hotels found. <br><a href="/">Back to Home</a>'

if __name__ == '__main__':
    app.run(debug=True)
