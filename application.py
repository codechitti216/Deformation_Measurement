import streamlit as st
from interfaces.Deformation_interface import main as app1_main
from interfaces.Graphs_interface import main as app2_main

def main():
    st.sidebar.title("Select Application")
    app_choice = st.sidebar.radio("", ("Deformation Measurement", "Point Tracking and Graphs generation"))

    if app_choice == "Deformation Measurement":
        app1_main()
    elif app_choice == "Point Tracking and Graphs generation":
        app2_main()

if __name__ == "__main__":
    main()
