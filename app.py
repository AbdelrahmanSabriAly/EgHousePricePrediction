import streamlit as st  # Importing the Streamlit library for building web applications
import pickle  # Importing the pickle library for loading the trained model
import numpy as np  # Importing the NumPy library for numerical computations
import pandas as pd  # Importing the Pandas library for data manipulation
from streamlit_option_menu import option_menu  # Importing a custom option menu widget
import seaborn as sns  # Importing the Seaborn library for data visualization
import matplotlib.pyplot as plt  # Importing the Matplotlib library for plotting
from annotated_text import annotated_text  # Importing a custom annotated text widget

# CSS style to hide Streamlit's main menu and footer
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer{visibility: hidden;}
</style>
"""

# Applying the CSS style to hide Streamlit's main menu and footer
st.markdown(hide_st_style, unsafe_allow_html=True)

def format_money(number):
    """
    Function to format a number as a money string with commas separating thousands.

    Args:
        number (int): The number to be formatted.

    Returns:
        str: The formatted money string.
    """
    # Convert number to string
    number_str = str(number)

    # Reverse the string
    reversed_str = number_str[::-1]

    # Divide the reversed string into groups of three digits
    groups = [reversed_str[i:i+3] for i in range(0, len(reversed_str), 3)]

    # Join the groups with commas
    formatted_str = ",".join(groups)

    # Reverse the string again
    final_str = formatted_str[::-1]

    # Return the formatted string
    return final_str


# Loading the datasets and the trained model
df = pd.read_csv('data.csv')  # Dataset containing house price data
X = pd.read_csv("Preprocessed_data.csv")  # Preprocessed dataset without label column used for modeling
pickle_in = open("Model.pkl","rb")  # Pickle file containing the trained model
model = pickle.load(pickle_in)  # Loading the trained model from the pickle file


with st.sidebar:
    # Creating a sidebar menu with different options
    choose = option_menu("Main Menu", ["About", "EDA", "App", "Contact"],
                         icons=['house', 'file-slides', 'app-indicator', 'person lines fill'],
                         menu_icon="list", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#0E1117"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "About":
    # Displaying information about the project and relevant links
    st.title("House Price Prediction :house:")
    st.subheader("This project is based on an Egyptian dataset.")
    st.write("You can find the dataset in the following link:")
    st.markdown("[Egyptian Houses_price Data](https://www.kaggle.com/datasets/mohammedaltet/egypt-houses-price)")
    st.write("This work is inspired by the notebook of MOHAMMED ALTET, you can find the notebook in the following link")
    st.markdown("[Egypt_Houses_notebook](https://www.kaggle.com/code/mohammedaltet/egypt-houses-notebook)")
    st.subheader("You can find the code in the following GitHub repository")
    st.markdown("[GitHub Repo](https://github.com/AbdelrahmanSabriAly/EgHousePricePrediction.git)")

elif choose == "App":
    # Creating an interactive web form for house price prediction
    st.title("House Price Prediction :house:")
    Bed = st.slider("Enter number of Bedrooms", 1, 10)
    Bath = st.slider("Enter number of Bathrooms", 1, 10)
    Area = st.slider("Enter the area", 50, 200)
    Level = st.slider("Enter the level", 0, 12)

    types_list = ['Apartment', 'Chalet', 'Stand Alone Villa', 'Town House', 'Twin House', 'Duplex', 'Penthouse']
    Type = st.selectbox("Choose the type", types_list)

    Furnished = st.radio("Is the house furnished?", ["Yes", "No"])
    city_list = ["Nasr City", "Camp Caesar", "Smoha", "New Cairo - El Tagamoa", "Sheikh Zayed", "Shorouk City",
                 "Sidi Beshr", "Gesr Al Suez", "Mokattam", "New Capital City", "New Damietta", "Zahraa Al Maadi",
                 "6th of October", "Mansura", "New Heliopolis", "Badr City", "Imbaba", "Borg al-Arab",
                 "Mohandessin", "Glim", "Ain Sukhna", "Maadi", "Hadayek 6th of October", "Haram", "Madinaty",
                 "10th of Ramadan", "Heliopolis", "Rehab City", "North Coast", "Nakheel", "Hadayek al-Ahram",
                 "Obour City", "Hurghada", "Agouza", "Mandara", "Helmeyat El Zaytoun", "Katameya", "Alamein",
                 "Miami", "Hadayek al-Kobba", "Mostakbal City", "Agami", "Ras al-Bar", "Bolkly", "15 May City",
                 "Seyouf", "Sharq District", "Ismailia City", "Ain Shams", "Shubra al-Khaimah", "Laurent",
                 "Al Ibrahimiyyah", "New Nozha", "New Mansoura", "Dokki", "Sheraton", "Faisal", "Moharam Bik",
                 "Al Manial", "Sahl Hasheesh", "West Somid", "Gianaclis", "Asafra", "Gouna", "Zamalek", "Zagazig",
                 "Shubra", "Marsa Matrouh", "Giza District", "Almazah", "San Stefano", "Saba Pasha", "Kafr Abdo",
                 "Azarita", "Zezenia", "Maamoura", "Asyut City", "Tanta", "Stanley", "Roushdy", "Amreya", "Abu Talat",
                 "Sharm al-Sheikh", "Hammam", "Ras Sedr", "Dabaa"]

    City = st.selectbox("Choose the city", city_list)

    payment_list = ['cash', 'Cash or Installment', 'Installment', 'other']
    Payment_Option = st.selectbox("Choose the Payment Option", payment_list)

    delevary_list = ['Core & Shell', 'Finished', 'Not Finished', 'Semi Finished']
    Delivery_Term = st.selectbox("Choose the Delivery Term", delevary_list)

    def PREDICT():
        # Create an array of zeros with the same length as the feature columns
        x = np.zeros(len(X.columns))
        x[0] = Bed
        x[1] = Bath
        x[2] = Area
        x[3] = Level

        # Find the indices corresponding to the selected options and set them to 1
        Type_index = np.where(X.columns == ("Type_" + Type))[0]
        Furnished_index = np.where(X.columns == ("Furnished_" + Furnished))[0]
        City_index = np.where(X.columns == ("City_" + City))[0]
        Payment_Option_index = np.where(X.columns == ("Payment_Option_" + Payment_Option))[0]
        Delivery_Term_index = np.where(X.columns == ("Delivery_Term_" + Delivery_Term))[0]

        indices_to_set = np.concatenate([Type_index, Furnished_index, City_index, Payment_Option_index,
                                         Delivery_Term_index])
        np.put(x, indices_to_set, 1)

        # Make a prediction using the loaded model
        prediction = int(model.predict([x])[0])
        formatted_number = format_money(prediction)
        st.success("Your house price is: " + formatted_number + " EGP")

    st.button("Predict!", on_click=PREDICT)

elif choose == "EDA":
    st.title("House Price Prediction :house:")
    st.subheader("Question 1: Which city has the largest number of buildings?")
    mp = df['City'].value_counts()[0:20].sort_values()
    plt.style.use('dark_background')
    plt.rcParams['text.color'] = 'white'
    fig = plt.figure()
    # Set the style to dark
    sns.barplot(y=mp.index, x=mp.values)
    plt.title('The City With The Most Buildings')
    st.pyplot(fig)

    st.subheader("Question 2: Which city has the most expensive buildings?")
    lpm = df.groupby('City')['Price'].mean()[0:20].sort_values()
    plt.style.use('dark_background')
    plt.rcParams['text.color'] = 'white'
    fig = plt.figure()
    sns.barplot(y=lpm.index, x=lpm.values)
    plt.title('The City With The Higher Buildings Price')
    st.pyplot(fig)

    st.subheader("Question 3: Which building type is the most expensive?")
    lpp = df.groupby('Type')['Price'].mean().sort_values()
    plt.style.use('dark_background')
    plt.rcParams['text.color'] = 'white'
    fig = plt.figure()
    sns.barplot(y=lpp.index, x=lpp.values)
    plt.title('Most Expensive Property Type Building')
    st.pyplot(fig)

    st.subheader("Question 4: Which building type has the largest number of bedrooms?")
    lpb = df.groupby('Type')['Bedrooms'].mean().sort_values()
    plt.style.use('dark_background')
    plt.rcParams['text.color'] = 'white'
    fig = plt.figure()
    sns.barplot(y=lpb.index, x=lpb.values)
    plt.title('Property With Mean Bed Room Number')
    st.pyplot(fig)

else:
    # Contact information
    st.subheader("Abdelrahman Sabri Aly")
    st.write("Email: aaly6995@gmail.com")
    st.write("Phone: +201010681318")
    st.markdown("[WhatsApp:]( https://wa.me/201010681318)")
