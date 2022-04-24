"""
Runs a website allowing a user to predict the genre of a book from its
description.
"""

from flask import Flask, render_template, request

app = Flask(__name__)

def predict_genre_from_description(description):
    """
    Returns the genre predicted for given description. This is temporary.

    Args:
        description (str): A Goodreads book description.

    Returns:
        predicted_genre (str or None): The predicted genre.
    """
    if description == 'test':
        return 'Fantasy'
    return None

@app.route('/', methods=['POST', 'GET'])
def index():
    """
    Renders the index.html page.

    Returns:
        template (str): The HTML to render.
    """
    if request.method == 'POST':
        description = request.form['description']
        genre = predict_genre_from_description(description)
        return render_template('result.html', genre=genre)
    else:
        return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result(genre):
    """
    Renders the result.html page for the given result.

    Args:
        genre (str): The genre to be given as a result.

    Returns:
        template (str): The HTML to render.
    """
    if request.method == 'POST':
        description = request.form['description']
        genre = predict_genre_from_description(description)
        return render_template('result.html', genre=genre)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
