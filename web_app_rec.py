import pandas as pd
import numpy as np
import re
from flask import Flask, render_template, request, Markup
import webbrowser, threading, os
from surprise.prediction_algorithms.matrix_factorization import NMF
import recommender_cs as rc



# Initialize app
app = Flask(__name__)

# Getting movie dataframes:
movies_df = pd.read_csv('data/movies/movies.csv')
ratings_df = pd.read_csv('data/movies/ratings.csv')

movie_rec = rc.MyMovieRecommender(NMF(), ratings_df, movies_df)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_tastes():
    unique_movies = movies_df['title'].unique()
    unique_movies = sorted(unique_movies, key=lambda m: m)
    unique_movies.append('No Selection')
    rating_vals = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 'No Rating']

    return render_template('input.html', unique_movies=unique_movies, rating_vals=rating_vals)

@app.route('/filter', methods = ['GET', 'POST'])
def filtering():
    genre_combos = [combo.split('|') for combo in movies_df['genres'].unique()]

    genres = set()
    for combo in genre_combos:
        for genre in combo:
            if genre != '(no genres listed)':
                genres.add(genre)

    genres = sorted(list(genres), key=lambda g: g)
    genres.append('Select Genre')

    all_years = set([int(re.findall(r'\(([1|2]\d\d\d)\)', t)[0]) for t in movies_df['title'].values if re.findall(r'\(([1|2]\d\d\d)\)', t) != []])
    all_years = sorted(list(all_years))
    start_years = all_years + ['From Earliest Date']
    end_years = all_years + ['To Latest Date']

    filter_options = ['Include', 'Exclude', 'Select Filter']

    message = Markup("Choose how you would like to filter your results:")
    ratings_dict = {}


    if request.form.get('movie1', '--N/A--') != '--N/A--':
        for i in range(1,31):
            title = request.form['movie{0}'.format(i)]
            rating = request.form['rating{0}'.format(i)]

            if title != 'No Selection' and rating != 'No Rating':
                ratings_dict[title] = float(rating)

        if ratings_dict == {}:
            message = Markup('''Looks like you didn't rate any movies. I can't give you your recommendations with any of your movie ratings.</br>Go back and re-enter you ratings here:</br></br>
            <form action="/input" method="POST" align="center" style="width:100%">
                <input type="submit" value="Enter in your Favorite Movies" style="width:50%"/>
            </form>''')

        else:
            movie_rec.get_my_recs(ratings_dict)

    return render_template('/filter.html', message=message, genres=genres, filter_options=filter_options, start_years=start_years, end_years=end_years)

@app.route('/recs', methods = ['GET', 'POST'])
def recommendations():
    message = "Your current filter:</br>"
    title_to_genre = {t : g for t, g in zip(movie_rec.movies_df['title'], movie_rec.movies_df['genres'])}

    my_recs = [m[0] for m in movie_rec.my_estimates if m[1] >= 0.5]
    for i in range(1,7):
        genre = request.form['genre{0}'.format(i)]
        f = request.form['filter{0}'.format(i)]

        if genre != 'Select Genre' and f != 'Select Filter':
            if f == 'Include':
                my_recs = [m for m in my_recs if genre in title_to_genre[m]]
                message += 'Including {0}, '.format(genre)
            else:
                my_recs = [m for m in my_recs if genre not in title_to_genre[m]]
                message += 'Excluding {0}, '.format(genre)

    message.strip(',')
    start_year = request.form['start_year']
    end_year = request.form['end_year']

    message += "</br>From {0} to {1}".format(start_year.replace('From', ''), end_year.replace('To', ''))
    if start_year != 'From Earliest Date':
        my_recs = [m for m in my_recs if re.findall(r'\(([1|2]\d\d\d)\)', m) != []]

        my_recs = [m for m in my_recs if int(re.findall(r'\(([1|2]\d\d\d)\)', m)[0]) >= int(start_year)]

    if end_year != 'To Latest Date':
        my_recs = [m for m in my_recs if re.findall(r'\(([1|2]\d\d\d)\)', m) != []]

        my_recs = [m for m in my_recs if int(re.findall(r'\(([1|2]\d\d\d)\)', m)[0]) <= int(end_year)]

    return render_template('recs.html', my_recs=my_recs, message=Markup(message))


@app.route('/bad_recs', methods = ['GET', 'POST'])
def bad_recommendations():
    message = "Your current filter:</br>"
    title_to_genre = {t : g for t, g in zip(movie_rec.movies_df['title'], movie_rec.movies_df['genres'])}

    my_recs = [m[0] for m in movie_rec.my_estimates if m[1] < 0.5][::-1]
    for i in range(1,6):
        genre = request.form['genre{0}'.format(i)]
        f = request.form['filter{0}'.format(i)]

        if genre != 'Select Genre' and f != 'Select Filter':
            if f == 'Include':
                my_recs = [m for m in my_recs if genre in title_to_genre[m]]
                message += 'Including {0}, '.format(genre)
            else:
                my_recs = [m for m in my_recs if genre not in title_to_genre[m]]
                message += 'Excluding {0}, '.format(genre)

    message.strip(',')
    start_year = request.form['start_year']
    end_year = request.form['end_year']

    message += "</br>From {0} to {1}".format(start_year.replace('From', ''), end_year.replace('To', ''))
    if start_year != 'From Earliest Date':
        my_recs = [m for m in my_recs if re.findall(r'\(([1|2]\d\d\d)\)', m) != []]

        my_recs = [m for m in my_recs if int(re.findall(r'\(([1|2]\d\d\d)\)', m)[0]) >= int(start_year)]

    if end_year != 'To Latest Date':
        my_recs = [m for m in my_recs if re.findall(r'\(([1|2]\d\d\d)\)', m) != []]

        my_recs = [m for m in my_recs if int(re.findall(r'\(([1|2]\d\d\d)\)', m)[0]) <= int(end_year)]

    return render_template('bad_recs.html', my_recs=my_recs, message=Markup(message))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
