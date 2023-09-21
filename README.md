# Anime-Recommendation-Web-App
This web application recommends similar anime based on user input. It utilizes a cosine similarity matrix created from an IMDb anime dataset to find anime with similar genres.
Getting Started
These instructions will help you set up and run the application on your local machine.

Open a terminal or command prompt and navigate to the project directory.

Install the required Python packages using pip:

Copy code
```pip install streamlit pandas scikit-learn```

# Usage
In the project directory, place the imdb_anime.csv dataset file.

Run the Streamlit app by executing the following command:

Copy code
```streamlit run app.py```

The web application will open in your default web browser.

Enter the name of an anime you like in the text input field and click the "Get Recommendations" button.

The application will display a list of recommended anime based on similarity to your input.

#Customization
You can customize the behavior and appearance of the application by modifying the app.py file. For example, you can change the default anime in the text input field, adjust the number of recommended anime to display, or apply CSS styling to the app.

Built With
Streamlit - The web framework used for building the user interface.
Pandas - For data manipulation and analysis.
scikit-learn - For creating TF-IDF vectors and computing cosine similarity.
