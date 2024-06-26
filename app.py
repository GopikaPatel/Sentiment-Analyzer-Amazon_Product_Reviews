import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the TF-IDF vectorizer
with open('Tfidf_Vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load the trained models
with open('svr_lin_model.pkl', 'rb') as f:
    svr_lin_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Sentiment Analyzer", style={'textAlign': 'center', 'marginBottom': '30px','color':'brown'}),
    dcc.Textarea(
        id='input-text',
        placeholder='Enter your review...',
        value='',
        style={'width': '100%', 'height': 100, 'marginBottom': '15px'}
    ),
    html.Button('Submit', id='submit-button', n_clicks=0, style={'display': 'block', 'margin': 'auto','color':'green'}),
    html.Div(id='output-prediction', style={'marginTop': '30px', 'textAlign': 'center', 'fontSize': '18px'})
])

# Define callback to handle user input and perform sentiment analysis
@app.callback(
    Output('output-prediction', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def predict_sentiment(n_clicks, input_text):
    if n_clicks > 0:
        # Clean the input text (you should implement this function)
        cleaned_input = clean_text(input_text)

        # Transform the input using TF-IDF vectorizer
        input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        # Predict sentiment using the trained models
        svr_lin_sentiment = svr_lin_model.predict(input_tfidf)[0]
        random_forest_sentiment = random_forest_model.predict(input_tfidf)[0]

        # Determine sentiment label
        svr_lin_sentiment_label = 'Positive' if svr_lin_sentiment == 1 else 'Negative'
        random_forest_sentiment_label = 'Positive' if random_forest_sentiment == 1 else 'Negative'

        # Display the predicted sentiment
        return html.Div([
            html.P('Sentiment: {}'.format(svr_lin_sentiment_label)),
           
        ], style={'marginBottom': '20px', 'fontSize': '18px', 'fontWeight': 'bold'})

    else:
        return ''


def clean_text(text):
    # Implement your text cleaning process here
    # This is just a placeholder function
    cleaned_text = text.strip().lower()  # Example: converting text to lowercase and removing leading/trailing spaces
    return cleaned_text


if __name__ == '__main__':
    app.run_server(debug=True)
