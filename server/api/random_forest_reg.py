from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy
import shap

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# # Load data and split into training and testing sets
# data = ... # Load your data
# trainX, trainY, testX, testY = ... # Split your data


# Define API endpoint for making predictions
@app.route('/api/predict_tiktok', methods=['POST'])
def predict():
    # Get feature array from request data
    attributeValues = request.json['attributeValues']

    # print(attributeValues)
    # Train random forest regressor on training set
    tiktok_songs_2020 = pd.read_csv('../data/TikTok_songs_2020.csv')
    tiktok_songs_2021 = pd.read_csv('../data/TikTok_songs_2021.csv')
    tiktok_songs_2022 = pd.read_csv('../data/TikTok_songs_2022.csv')
    # spotify_top_charts_20 = pd.read_csv('../../../public/spotify_top_charts_20.csv')
    # spotify_top_charts_21 = pd.read_csv('../../../public/spotify_top_charts_21.csv')
    # spotify_top_charts_22 = pd.read_csv('../../../public/spotify_top_charts_22.csv')

    # spotify_df = pd.concat([spotify_top_charts_20,spotify_top_charts_21,spotify_top_charts_22])
    # spotify_df['artist_name'] = spotify_df['artist_names'] 
    # spotify_df = spotify_df.drop(['artist_names'],axis=1)
    tiktok_df = pd.concat([tiktok_songs_2020,tiktok_songs_2021,tiktok_songs_2022])

    #group by artist & track name and average the values of all numerical values
    tiktok_df_updated = tiktok_df.groupby(['track_name','artist_name'])[['artist_pop', 'track_pop','danceability', 'energy', 'loudness', 'mode', 'key', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','time_signature', 'duration_ms']].mean()

    #independent variables
    X = tiktok_df_updated[['artist_pop',
        'danceability', 'energy', 'loudness', 'mode', 'key', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'time_signature', 'duration_ms']]

    #dependent variables
    y = tiktok_df_updated['track_pop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print(attributeValues)

    rf = RandomForestRegressor(max_depth= 7,
    max_features= 'sqrt',
    n_estimators= 500,
    random_state= 18)
    rf.fit(X_train, y_train)
    attributeValues_df = pd.DataFrame(attributeValues, index=[0])
    attributeValues_arr = numpy.array(attributeValues_df)
   
    # Make prediction using trained model
    predictedValue = rf.predict(attributeValues_arr)
    print(predictedValue)

    # Create a SHAP explainer for your trained model
    explainer = shap.Explainer(rf, X_train)

    # Calculate SHAP values for the specific input data point
    shap_values = explainer(attributeValues_arr)
    shap_values_arr = explainer.shap_values(attributeValues_arr)
    print("shap values",shap_values_arr)

    

    # Return predicted value as JSON response
    return jsonify({
        'data': {
            'predictedValue':predictedValue.tolist(),
            'shapValues': shap_values_arr.tolist()

        }
    })

@app.route('/api/predict_spotify', methods=['POST'])

def predict_spotify():
    tiktok_songs_2020 = pd.read_csv('../data/TikTok_songs_2020.csv')
    tiktok_songs_2021 = pd.read_csv('../data/TikTok_songs_2021.csv')
    tiktok_songs_2022 = pd.read_csv('../data/TikTok_songs_2022.csv')
    spotify_top_charts_20 = pd.read_csv('../data/spotify_top_charts_20.csv')
    spotify_top_charts_21 = pd.read_csv('../data/spotify_top_charts_21.csv')
    spotify_top_charts_22 = pd.read_csv('../data/spotify_top_charts_22.csv')

    spotify_df = pd.concat([spotify_top_charts_20,spotify_top_charts_21,spotify_top_charts_22])
    spotify_df['artist_name'] = spotify_df['artist_names'] 
    spotify_df = spotify_df.drop(['artist_names'],axis=1)
    tiktok_df = pd.concat([tiktok_songs_2020,tiktok_songs_2021,tiktok_songs_2022])
    #merging using track_name and year since each track can have multiple rows (due to it ranking in different years)
    df_merged = tiktok_df.merge(spotify_df, on=['track_name','artist_name'],suffixes=('_tiktok','_spotify'),how='left')
    df_merged_updated = df_merged.dropna(subset=['peak_rank', 'weeks_on_chart',
       ])
    # Define a list of prefixes for which you want to calculate the mean
    prefixes_to_average = ['danceability', 'energy', 'loudness', 'mode', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'time_signature', 'duration_ms']

    column_names = df_merged_updated.columns
    for prefix in prefixes_to_average:
        # Find columns that start with the current prefix
        relevant_columns = [col for col in df_merged_updated.columns if col.startswith(prefix)]
        df_merged_updated[f'{prefix}_mean'] = df_merged_updated[relevant_columns].mean(axis=1)
        df_merged_updated.drop(relevant_columns, axis=1, inplace=True)
        #independent variables
    X = df_merged_updated[['artist_pop','track_pop',
        'danceability_mean', 'energy_mean', 'loudness_mean', 'mode_mean',
        'key_mean', 'speechiness_mean', 'acousticness_mean',
        'instrumentalness_mean', 'liveness_mean', 'tempo_mean',
        'time_signature_mean', 'duration_ms_mean']]
    

    #dependent variables
    y = df_merged_updated[['peak_rank', 'weeks_on_chart']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
   

    # Create a multi-output regression model (Random Forest, for example)
    multi_output_model = MultiOutputRegressor(RandomForestRegressor(max_depth= 3,
    max_features= 'sqrt',
    n_estimators= 300,
    random_state= 18))

    # Train the model using X_train and y_train, where y_train has multiple columns
    multi_output_model.fit(X_train, y_train)

    # Make predictions for multiple target variables
    y_pred = multi_output_model.predict(X_test)
    predicted_weeks_on_chart = y_pred[:, 1]  
    # Extract actual weeks on chart
    actual_weeks_on_chart = numpy.array(y_test)[:, 1]  
    track_popularity = X_test['track_pop']  
    # Extract actual peak rank
    predicted_peak_rank = y_pred[:, 0]  
    # Extract actual weeks on chart
    actual_peak_rank = numpy.array(y_test)[:, 0]  

    # Extract track information
    track_info = X_test[['artist_pop', 'track_pop', 'danceability_mean', 'energy_mean', 'loudness_mean', 'mode_mean', 'key_mean', 'speechiness_mean', 'acousticness_mean', 'instrumentalness_mean', 'liveness_mean', 'tempo_mean', 'time_signature_mean', 'duration_ms_mean']].reset_index()
    test_data_index = X_test.index
    # Create a list of dictionaries, where each dictionary represents a track and contains the relevant information
    output_list = []
    def create_track_dict(row):

        return {
            
            'artist_pop': row['artist_pop'],
            'track_pop': row['track_pop'],
            'danceability': row['danceability_mean'],
            'energy': row['energy_mean'],
            'loudness': row['loudness_mean'],
            'mode': row['mode_mean'],
            'key': row['key_mean'],
            'speechiness': row['speechiness_mean'],
            'acousticness': row['acousticness_mean'],
            'instrumentalness': row['instrumentalness_mean'],
            'liveness': row['liveness_mean'],
            'tempo': row['tempo_mean'],
            'time_signature': row['time_signature_mean'],
            'duration_ms': row['duration_ms_mean'],
            'actual': {'peak_rank': actual_peak_rank[row.name], 'weeks_on_chart': actual_weeks_on_chart[row.name]},
            'predicted': {'peak_rank': predicted_peak_rank[row.name], 'weeks_on_chart': predicted_weeks_on_chart[row.name]}
        }

    output_list = track_info.apply(create_track_dict, axis=1).tolist()
    # Return the output as JSON
    return jsonify({
        'output': output_list,
        'track_name': df_merged_updated.loc[test_data_index, 'track_name'].tolist(),
    })
            

if __name__ == '__main__':
    app.run()