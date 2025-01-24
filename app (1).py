# -*- coding: utf-8 -*-
# Streamlit deployment code
import streamlit as st

def main():
    st.title("Fake and Real News Detection")

    # Load the model and vectorizer
    with open('random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # User input
    user_input = st.text_area("Enter news text to analyze:")

    if st.button("Analyze"):
        if user_input.strip():
            # Transform user input
            input_vectorized = vectorizer.transform([user_input])
            # Predict
            prediction = model.predict(input_vectorized)[0]
            # Display result
            if prediction == 1:
                st.success("This news is Real.")
            else:
                st.error("This news is Fake.")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
