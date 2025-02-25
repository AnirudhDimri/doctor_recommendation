import streamlit as st
import pandas as pd
import requests
import json

# Load dataset (Ensure the CSV file is in the correct location)
df = pd.read_csv("doctor_dataset.csv")  # Update the file path if needed

# Streamlit UI Setup
st.set_page_config(page_title="Doctor Recommendation", layout="wide")
st.title("Find the Best Doctor for Your Needs 🏥")
st.write("Use AI to discover top healthcare professionals worldwide!")

# User Input
location = st.selectbox("Select Location:", ["All"] + sorted(df["location"].unique().tolist()))
specialty = st.selectbox("Select Specialty:", ["All"] + sorted(df["specialties"].unique().tolist()))
search_query = st.text_input("Enter keywords (e.g., doctor name, expertise, etc.):")

# Filtering the dataset
filtered_df = df.copy()
if location != "All":
    filtered_df = filtered_df[filtered_df["location"] == location]
if specialty != "All":
    filtered_df = filtered_df[filtered_df["specialties"].str.contains(specialty, case=False, na=False)]
if search_query:
    filtered_df = filtered_df[filtered_df["name"].str.contains(search_query, case=False, na=False) |
                              filtered_df["overview"].str.contains(search_query, case=False, na=False)]

# Hugging Face API Integration for AI-powered Recommendation
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
HUGGINGFACE_API_KEY = "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"  # Replace with your key

def ai_recommendation(query, doctors):
    """ Get AI-powered doctor recommendations via Hugging Face API """
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    prompt = f"Find the best doctor for: {query}. Here are some options: {json.dumps(doctors, indent=2)}"

    data = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        try:
            return response.json()[0]['generated_text']
        except (KeyError, IndexError):
            return "AI could not generate a response."
    else:
        return f"AI service is currently unavailable. Error: {response.status_code}"

if search_query:
    ai_suggestions = ai_recommendation(search_query, filtered_df.to_dict(orient="records"))
    st.subheader("AI Recommendations:")
    st.write(ai_suggestions)

# Display Results
st.subheader("Doctor Recommendations:")
if filtered_df.empty:
    st.write("No matching doctors found. Try adjusting your filters.")
else:
    for _, row in filtered_df.iterrows():
        st.markdown(f"### [{row['name']}]({row['profile_link']})")
        st.write(f"**Location:** {row['location']}")
        st.write(f"**Specialty:** {row['specialties']}")
        st.write(f"{row['overview']}")
        st.markdown("---")

# Footer
st.write("💡 *Powered by AI to help you find the best healthcare professionals!*")
