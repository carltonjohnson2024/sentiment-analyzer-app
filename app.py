import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px

st.title("Customer Review Sentiment Analyzer")
st.markdown("This app analyzes the sentiment of customer reviews to gain insights into their opinions.")



# Import reviews.csv
#import pandas as pd
#df = pd.read_csv("reviews.csv")
#st.write(df)
def classify_sentiment_openai(review_text):
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content


# CSV file uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file with restaurant reviews", 
    type=["csv"])

# OpenAI API Key input
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    type="password", 
    help="You can find your API key at https://platform.openai.com/account/api-keys"
)

# Once the user uploads a csv file
if uploaded_file is not None:

    # Read the file
    reviews_df = pd.read_csv(uploaded_file)

    # Check if the data has a text column
    text_columns = reviews_df.select_dtypes(include=[object]).columns

    if len(text_columns) == 0:
        st.error("No text columns found in the uploaded file.")
    
    # Show a dropdown menu to select the review column
    review_column = st.selectbox(
        "Select the column that contains the reviews",
        text_columns
    )

    # Analyze the sentiments of the review column
    reviews_df["Sentiment"] = reviews_df[review_column].apply(classify_sentiment_openai)
    

    # Display sentiment distribution in metrics in 3 columns: positive, negative, neutral
    # Make the strings in the Sentiment column title case
    reviews_df["Sentiment"] = reviews_df["Sentiment"].str.title()
    sentiment_count = reviews_df["Sentiment"].value_counts()
    # st.write(reviews_df)
    # st.write(sentiment_count)

    # Create 3 columns to display the 3 metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        # Show the number of positive reviews and the percentage
        positive_count = sentiment_count.get("Positive", 0)
        st.metric("Positive", 
                  positive_count, 
                  f"{positive_count / len(reviews_df) * 100:.2f}%")
    
    with col2:
        # Show the number of neutral reviews and the percentage
        neutral_count = sentiment_count.get("Neutral", 0)
        st.metric("Neutral", 
                  neutral_count, 
                  f"{neutral_count / len(reviews_df) * 100:.2f}%")
        
    with col3:
        # Show the number of positive reviews and the percentage
        negative_count = sentiment_count.get("Negative", 0)
        st.metric("Negative", 
                  negative_count, 
                  f"{negative_count / len(reviews_df) * 100:.2f}%")
    
    # Display pie chart
    fig = px.pie(
        values=sentiment_count.values, 
        names=sentiment_count.index,
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig)


# Example usage
# Write the result to the app with title "Sentiment"
# st.title("Sentiment")
# st.write(classify_sentiment_openai(user_input))