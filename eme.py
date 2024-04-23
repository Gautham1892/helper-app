import streamlit as st
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from twilio.rest import Client
from testfile import dummy_fn 


client=Client('ACb1edc646cadb42b6dd179151b754a079','898cc1bf6c1d0680ca471854c359448e')

# Function to generate random Indian phone numbers
def generate_phone_number():
    # Generate a valid Indian phone number
    first_three_digits = np.random.randint(600, 999)
    remaining_digits = np.random.randint(1000000, 9999999)
    return f"+91-{first_three_digits}-{remaining_digits}"

# Generate 35 random user coordinates across Chennai
np.random.seed(42)
random_users = {
    f"User {i+1}": [
        np.random.uniform(12.9, 13.3),  # Latitude
        np.random.uniform(80.1, 80.3)    # Longitude
    ] 
    for i in range(35)
}

# Generate 50 random flood hazard coordinates across Chennai
random_hazards = {
    f"Hazard {i+1}": [
        np.random.uniform(12.9, 13.3),  # Latitude
        np.random.uniform(80.1, 80.3)    # Longitude
    ] 
    for i in range(50)
}

# Generate random phone numbers for users
random_phone_numbers = {user: generate_phone_number() for user in random_users.keys()}

# Display in Streamlit
st.title("User Locations in Chennai")
users_df = pd.DataFrame.from_dict(random_users, orient='index', columns=['Latitude', 'Longitude'])
users_df['Phone Number'] = random_phone_numbers.values()
st.write(users_df)

# File upload for flood hazard data
st.title("Upload Flood Hazard Data")
uploaded_file = st.file_uploader("Upload CSV file with flood hazards", type=["csv"])

# Generate random flood hazards if no file uploaded
if not uploaded_file:
    hazards_df = pd.DataFrame.from_dict(random_hazards, orient='index', columns=['Latitude', 'Longitude'])
else:
    hazards_df = pd.DataFrame.from_dict(random_hazards, orient='index', columns=['Latitude', 'Longitude'])

# Create a map centered around Chennai
m = folium.Map(location=[13.0827, 80.2707], zoom_start=11)

# Add flood hazards to the map
for index, row in hazards_df.iterrows():
    folium.Circle(
        location=[row['Latitude'], row['Longitude']],
        radius=200,  # Set radius as per your requirement
        color='red',
        fill=True,
        fill_color='red'
    ).add_to(m)

# Cluster users based on their locations
kmeans = KMeans(n_clusters=5, random_state=42).fit(list(random_users.values()))
cluster_labels = kmeans.labels_

# Update user table with nearby hazards and cluster labels
users_df['Cluster'] = cluster_labels

# Function to check hazards near user
def check_hazards_nearby(user_lat, user_lon):
    for _, row in hazards_df.iterrows():
        if np.sqrt((user_lat - row['Latitude'])**2 + (user_lon - row['Longitude'])**2) < 0.01:  # Adjust distance threshold as needed
            return "Flood Hazard"
    return "Safe"

users_df['Nearby Hazard'] = [check_hazards_nearby(lat, lon) for lat, lon in random_users.values()]

# Define colors for clusters
cluster_colors = ['blue', 'green', 'purple', 'orange', 'darkred']

# Add clustered users to the map
for i, cluster_center in enumerate(kmeans.cluster_centers_):
    folium.Marker(
        location=[cluster_center[0], cluster_center[1]],
        icon=folium.Icon(color=cluster_colors[i], icon='info-sign')
    ).add_to(m)

# Display map in Streamlit
html = m._repr_html_()
st.components.v1.html(html, height=600)

# Display updated table
st.write(users_df)

# Send alerts button
if st.button("Send Alerts"):
    result = dummy_fn()
    st.write("Alerts sent to users!")
