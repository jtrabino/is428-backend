const { DataFrame } = require('pandas-js');
const csv = require('csv-parser');
const fs = require('fs');

const tiktok_songs_2020 = await new Promise((resolve, reject) => {
  const results = [];
  fs.createReadStream('TikTok_songs_2020.csv')
    .pipe(csv())
    .on('data', (data) => {
      results.push(data);
    })
    .on('end', () => {
      resolve(new DataFrame(results));
    });
});

const tiktok_songs_2021 = await new Promise((resolve, reject) => {
  const results = [];
  fs.createReadStream('TikTok_songs_2021.csv')
    .pipe(csv())
    .on('data', (data) => {
      results.push(data);
    })
    .on('end', () => {
      resolve(new DataFrame(results));
    });
});

const tiktok_songs_2022 = await new Promise((resolve, reject) => {
  const results = [];
  fs.createReadStream('TikTok_songs_2022.csv')
    .pipe(csv())
    .on('data', (data) => {
      results.push(data);
    })
    .on('end', () => {
      resolve(new DataFrame(results));
    });
});

const spotify_top_charts_20 = await new Promise((resolve, reject) => {
  const results = [];
  fs.createReadStream('spotify_top_charts_20.csv')
    .pipe(csv())
    .on('data', (data) => {
      results.push(data);
    })
    .on('end', () => {
      resolve(new DataFrame(results));
    });
});

const spotify_top_charts_21 = await new Promise((resolve, reject) => {
  const results = [];
  fs.createReadStream('spotify_top_charts_21.csv')
    .pipe(csv())
    .on('data', (data) => {
      results.push(data);
    })
    .on('end', () => {
      resolve(new DataFrame(results));
    });
});

const spotify_top_charts_22 = await new Promise((resolve, reject) => {
  const results = [];
  fs.createReadStream('spotify_top_charts_22.csv')
    .pipe(csv())
    .on('data', (data) => {
      results.push(data);
    })
    .on('end', () => {
      resolve(new DataFrame(results));
    });
});

// add year col to spotify dataset
spotify_top_charts_20.assign({ year: 2020 });
spotify_top_charts_21.assign({ year: 2021 });
spotify_top_charts_22.assign({ year: 2022 });

// add year col to tiktok dataset
tiktok_songs_2020.assign({ year: 2020 });
tiktok_songs_2021.assign({ year: 2021 });
tiktok_songs_2022.assign({ year: 2022 });

// combine spotify datasets 
const spotify_df = DataFrame.concat([spotify_top_charts_20, spotify_top_charts_21, spotify_top_charts_22]);
spotify_df.assign({ artist_name: spotify_df.get('artist_names') });
spotify_df.drop(['artist_names'], { axis: 1 }, (err, df) => {
  if (err) {
    console.error(err);
  } else {
    console.log(df.head());
  }
});

// combine tiktok datasets
const tiktok_df = DataFrame.concat([tiktok_songs_2020, tiktok_songs_2021, tiktok_songs_2022]);
const tiktok_df_updated = tiktok_df.groupby(['track_name', 'artist_name']).mean(['artist_pop', 'track_pop', 'danceability', 'energy', 'loudness', 'mode', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms']);
console.log(tiktok_df_updated.head());