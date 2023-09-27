# Random Forest Classifier Implementation in Python

## Overview

This README provides a guide on how to install the required libraries and run a Random Forest Classifier project in
Python. Random Forest is a powerful ensemble learning technique used for classification tasks. Follow the steps below to
set up the project and start using the Random Forest Classifier.

## Installation

Before running the project, make sure you have Python installed on your system. Python 3.11 is required.

## Running the Project

Now that you have installed the required libraries, you can run the Random Forest Classifier project. Follow these
steps:

1. **Clone the Repository**:
   ```bash
   https://github.com/JindiHu/adult-census-income.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd adult-census-income
   ```

3. **Create a Virtual Environment (Optional but Recommended)**:

   Creating a virtual environment helps isolate project dependencies from your system-wide Python installation. To
   create a virtual environment, open your terminal/command prompt and run:

   ```bash
   python3 -m venv myenv
   ```
   Activate the virtual environment:

    - **On Windows**:

      ```bash
      myenv\Scripts\activate
      ```

    - **On macOS and Linux**:

      ```bash
      source myenv/bin/activate
      ```

4. **Install Required Libraries**:
   Install the necessary libraries using `pip3`:

   ```bash
   pip3 install -r requirements.txt
   ```

5. **Execute the Python Script**:
   ```bash
   python3 main.py
   ```
   The script is designed to generate figures for EDA and performance. To allow the script to continue running, please
   close each figure one by one as they appear.

   The script should load the `adult.data` dataset, split it into training and validation sets, create and train the
   classifier, validate and evaluate the performance of the trained model against on validation set. Finally, make
   predictions `adult.test` dataset.


6. **View the Results**:

   The script will display the EDA figures (collected under `/figures` directory), accuracy, and other performance
   metrics.


