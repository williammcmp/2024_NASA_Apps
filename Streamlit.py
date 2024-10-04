#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 05 2024

A very basic interface and starting temply

@author: william.mcm.p
"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title="Hackathon Challenge", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Home", "Data Visualization", "Interactive Plot"])

# Home Page
if page == "Home":
    st.title("Welcome to our Hackathon Project")
    st.write("""
    This is a basic Streamlit app to showcase our hackathon challenge. You can navigate through the app using the sidebar to explore data visualizations and interactive plots.
    """)

# Data Visualization Page
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("This page displays basic plots.")

    # Generate random data for plotting
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create a Plotly line chart
    fig = px.line(x=x, y=y, labels={'x': 'X Axis', 'y': 'Sine Wave'})
    st.plotly_chart(fig)

    # Add some UI tools
    st.subheader("UI Components")
    slider_value = st.slider("Select a value", 0, 100, 50)
    st.write(f"Slider value: {slider_value}")

# Interactive Plot Page
elif page == "Interactive Plot":
    st.title("Interactive Plot")
    st.write("This page contains an interactive scatter plot with filters.")

    # Create sample data for the scatter plot
    df = pd.DataFrame({
        'x': np.random.rand(100),
        'y': np.random.rand(100),
        'category': np.random.choice(['A', 'B', 'C'], size=100)
    })

    # UI tools: dropdown for selecting category
    category = st.selectbox("Select category", df['category'].unique())
    
    # Filter data based on selected category
    filtered_df = df[df['category'] == category]

    # Create a Plotly scatter plot
    fig = px.scatter(filtered_df, x='x', y='y', color='category',
                     labels={'x': 'X Axis', 'y': 'Y Axis', 'category': 'Category'})
    st.plotly_chart(fig)

# Footer
st.sidebar.write("Created by: Your Team")