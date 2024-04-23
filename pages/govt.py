import streamlit as st
from gtts import gTTS
from PIL import Image
import google.generativeai as genai

# Configure the Gemini API
genai.configure(api_key="AIzaSyB8kszduNytZVO_u3oXsnxOvDjTZSNQLuo")

# Set up the model
# Set up the model
generation_config = {
  "temperature": 1.2,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["You are Jaya a chatbot developed by team 7 , part of their capstone project which is \" Drone aided disaster mapping and action\" You will be provided with distaster data and you shall help the national disaster relif team based on that data only. people will use you during disaster to get information about which areas are affect , how bad is it and how it can be solved. First introduce who you are with just your name and the function you do . then proceed to ask if they need any help and answer those with the distaster data fed to you. avoid answering an unwanted unrelated questions."]
  },
  {
    "role": "model",
    "parts": ["I am Jaya, a chatbot designed to assist with disaster relief efforts by providing information based on real-time data. \n\nDo you need any assistance with the current disaster situation? I can provide information on affected areas, the severity of the damage, and potential solutions."]
  },
  {
    "role": "user",
    "parts": ["Here is data the government has got using drones , use the folowing to answer the questions and give solution . Identified Problem: Rooftops for Food Package DropsExample 1:Location: Rooftop of a residential building near Anna Nagar West (13.0935° N, 80.2021° E)Proposed Solution: Dispatch rescue teams from the Relief Camp/Shelter at Jawaharlal Nehru Indoor Stadium (13.0414° N, 80.2359° E) with food packages and supplies.Example 2:Location: Rooftop of a commercial complex near Teynampet (13.0459° N, 80.2424° E)Proposed Solution: Coordinate assistance from the NGO Coordination Center at Kotturpuram (13.0174° N, 80.2417° E) to deliver food packages and supplies to stranded individuals.Example 3:Location: Rooftop of an apartment building near Velachery (12.9815° N, 80.2207° E)Proposed Solution: Utilize drones to drop food packages and supplies from nearby distribution centers to the identified rooftop.Example 4:Location: Rooftop of a school building near Kodambakkam (13.0514° N, 80.2173° E)Proposed Solution: Coordinate with local volunteers to deliver food packages and supplies to individuals stranded on the rooftop.Example 5:Location: Rooftop of a hospital near Guindy (13.0082° N, 80.2184° E)Proposed Solution: Deploy helicopters to airlift food packages and supplies to the rooftop, coordinated from nearby helipads.Identified Problem: Open Grounds for People to GatherExample 1:Location: Open ground near Besant Nagar Beach (13.0002° N, 80.2714° E)Proposed Solution: Establish temporary shelters and distribute relief materials, coordinated from the Relief Camp/Shelter at Jawaharlal Nehru Indoor Stadium (13.0414° N, 80.2359° E).Example 2:Location: Playground near Adyar (13.0078° N, 80.2510° E)Proposed Solution: Set up medical camps and provide basic amenities, coordinated with nearby NGOs such as Chennai Volunteers at Kotturpuram (13.0174° N, 80.2417° E).Example 3:Location: Open space near Marina Beach (13.0524° N, 80.2806° E)Proposed Solution: Deploy sanitation facilities and organize community kitchens, coordinated with local authorities and NGOs.Example 4:Location: Park near Nanganallur (12.9827° N, 80.1834° E)Proposed Solution: Arrange transportation for vulnerable groups and coordinate with relief teams for assistance, utilizing nearby police control rooms.Example 5:Location: Ground near Chennai Trade Centre (13.0081° N, 80.1894° E)Proposed Solution: Conduct awareness campaigns and provide psychosocial support, collaborating with NGOs and community leaders.Identified Problem: Fire DetectedExample 1:Location: Commercial building near Kilpauk (13.0786° N, 80.2419° E)Proposed Solution: Mobilize fire and rescue teams from the Fire and Rescue Deployment at Egmore Headquarters (13.0744° N, 80.2646° E) to extinguish the fire and prevent further spread.Example 2:Location: Factory near Ambattur (13.1079° N, 80.1670° E)Proposed Solution: Coordinate firefighting efforts with neighboring industrial units and deploy additional resources from nearby fire stations.Example 3:Location: Residential area near Mylapore (13.0358° N, 80.2657° E)Proposed Solution: Evacuate residents to safety and establish a perimeter to contain the fire, coordinated with police control rooms and disaster management authorities.Example 4:Location: Market place near T. Nagar (13.0361° N, 80.2340° E)Proposed Solution: Utilize aerial firefighting techniques and deploy specialized teams to tackle the blaze, coordinated with central command centers and aerial support units.Example 5:Location: Warehouse near Guindy (13.0115° N, 80.2204° E)Proposed Solution: Implement fire prevention measures and conduct regular inspections of hazardous materials, coordinated with regulatory agencies and industrial safety experts.Identified Problem: Roadway BlockageExample 1:Location: Major road near St. Thomas Mount (13.0046° N, 80.2097° E)Proposed Solution: Deploy road clearance teams with heavy machinery from the Disaster Management Control Center at Ripon Building (13.0830° N, 80.2717° E) to remove obstacles and restore traffic flow.Example 2:Location: Highway near Maduravoyal (13.0572° N, 80.1377° E)Proposed Solution: Coordinate with local authorities and transportation agencies to divert traffic and provide alternative routes, utilizing real-time traffic monitoring systems.Example 3:Location: Arterial road near Pallavaram (12.9756° N, 80.1485° E)Proposed Solution: Implement temporary measures such as makeshift bridges or culverts to restore connectivity, coordinated with engineering teams and construction contractors.Example 4:Location: Bridge near Velachery (12.9799° N, 80.2220° E)Proposed Solution: Conduct structural assessments and reinforce vulnerable infrastructure, coordinated with civil engineering experts and bridge maintenance crews.Example 5:Location: Expressway near Sholinganallur (12.8982° N, 80.2264° E)Proposed Solution: Utilize drones for aerial surveys and reconnaissance to identify alternative routes, coordinated with traffic management authorities and GIS specialists.Identified Problem: Flooded Residential AreaExample 1:Location: Residential neighborhood near Pallikaranai (12.9408° N, 80.2081° E)Proposed Solution: Deploy specialized rescue teams with boats and life-saving equipment from the Boat Launching Point at Adyar River Bank (13.0150° N, 80.2499° E) for swift water rescues.Example 2:Location: Housing colony near Perungudi (12.9685° N, 80.2437° E)Proposed Solution: Establish temporary shelters and distribution centers for relief materials, coordinated with nearby relief camps and community organizations.Example 3:Location: Apartment complex near Saidapet (13.0212° N, 80.2185° E)Proposed Solution: Conduct door-to-door evacuation and provide medical assistance to vulnerable populations, coordinated with local healthcare facilities and volunteer groups.Example 4:Location: Residential enclave near Velachery (12.9791° N, 80.2204° E)Proposed Solution: Implement flood mitigation measures such as sandbagging and drainage improvements, coordinated with municipal authorities and environmental engineers.Example 5:Location: Gated community near Adambakkam (12.9807° N, 80.1973° E)Proposed Solution: Coordinate with property management teams to ensure the safety and well-being of residents, providing essential services and support as needed."]
  },
  {
    "role": "model",
    "parts": ["Thank you for providing this detailed disaster data. I am now equipped to answer your questions and offer solutions based on the specific problems identified. \n\nPlease tell me what information you need or what questions you have about the disaster situation. I will do my best to provide accurate and helpful answers based on the data you have provided."]
  },
])


def generate_audio(text):
    tts = gTTS(text, lang='en', tld="us")
    tts.save('output.mp3')

def execute_prompt(prompt):
    response = model.generate_content(prompt)
    return response.text

def execute_prompt_with_image(prompt, image):
    response = vision_model.generate_content([prompt, image], stream=True)
    response.resolve()
    return response.text

def main():
    st.title("Jaya AI")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "parts": [prompt]})

        # Send user input to the Gemini model
        convo.send_message(prompt)

        # Add model response to chat history
        st.session_state.messages.append({"role": "model", "parts": [convo.last.text]})

        # Display model response in chat
        st.write_stream({"parts": [convo.last.text]})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message["role"]
        parts = message["parts"]
        with st.chat_message(role):
            for part in parts:
                if role == "user":
                    st.write(part)  # Display user input directly
                else:
                    st.markdown(part)  # Display model response with formatting

if __name__ == "__main__":
    main()