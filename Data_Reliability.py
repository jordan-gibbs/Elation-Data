import matplotlib.pyplot as plt
import numpy as np
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
import pandas as pd
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# Define the function to calculate the Response Reliability Index
def calculate_response_reliability_index(data):
    # Initialize the Response Reliability Index with 10
    data['Response Reliability Index'] = 10

    # Social Desirability check
    social_desirability_questions = [
        "I feel the work I do makes a difference in the world.",
        "I have a desire to help those less fortunate and take tangible action to do so.",
        "I am able to sacrifice personal comforts for the benefit of others less fortunate."
    ]
    data['Social Desirability Score'] = data[social_desirability_questions].mean(axis=1)
    data.loc[data['Social Desirability Score'] > 80, 'Response Reliability Index'] -= 1

    # Thematic Similarity check
    thematic_similarity_questions = [
        "I am satisfied with the amount of freedom and independence I have when doing my job.",
        "I am satisfied with the amount of discretion I have when doing my job.",
        "I am satisfied with the level of freedom and discretion to change and adapt my job to better suit my personal needs."
    ]
    thematic_similarity_distance = data[thematic_similarity_questions].max(axis=1) - data[
        thematic_similarity_questions].min(axis=1)
    data.loc[thematic_similarity_distance > 50.001, 'Response Reliability Index'] -= 1

    # Reverse question similarity check
    reverse_question_1 = "I feel well-rested and refreshed in the mornings."
    reverse_question_2 = "I find it challenging to get up in the mornings."
    reverse_question_distance = abs(data[reverse_question_1] - data[reverse_question_2])
    data.loc[reverse_question_distance >= 25, 'Response Reliability Index'] -= 1
    data.loc[reverse_question_distance >= 50, 'Response Reliability Index'] -= 1
    data.loc[reverse_question_distance >= 75, 'Response Reliability Index'] -= 1
    data.loc[reverse_question_distance == 100, 'Response Reliability Index'] -= 1

    # Duration check
    data.loc[data['durationSeconds'] < 150, 'Response Reliability Index'] -= 3
    data.loc[
        (data['durationSeconds'] >= 150.001) & (data['durationSeconds'] <= 200.00), 'Response Reliability Index'] -= 2
    data.loc[(data['durationSeconds'] >= 200.01) & (data['durationSeconds'] <= 300), 'Response Reliability Index'] -= 1
    data.loc[(data['durationSeconds'] >= 300.01) & (data['durationSeconds'] <= 600), 'Response Reliability Index'] += 0
    data.loc[(data['durationSeconds'] >= 600.01) & (data['durationSeconds'] <= 1000), 'Response Reliability Index'] -= 1
    data.loc[data['durationSeconds'] > 1000.01, 'Response Reliability Index'] -= 2

    return data


def calculate_statistical_deviation_score(data):

    start_col = "I feel I belong in my organization."
    start_index = data.columns.get_loc(start_col)

    # Select only numeric response columns starting from the specified column
    response_columns = data.columns[start_index:]
    response_data = data[response_columns].apply(pd.to_numeric, errors='coerce')

    means = response_data.mean(axis=1)
    stds = response_data.std(axis=1)

    # Compute Absolute Z-scores for each response
    z_scores = (response_data.sub(means, axis=0)).div(stds, axis=0).abs()
    # data['Absolute Z-score'] = z_scores.sum(axis=1)/77
    data['Absolute Z-score'] = z_scores.sum(axis=1)

    # Determine the 95th Percentile Threshold
    threshold = np.percentile(data['Absolute Z-score'], 95)

    # Identify Outliers
    data['Above 95% threshold'] = np.where(data['Absolute Z-score'] > threshold, 'Yes', 'No')

    return data

def create_reliability_report(file_path):
    # Load the data from the specified path
    data = pd.read_csv(file_path)

    # Extract relevant columns
    columns_of_interest = [
        "Valid Response"
    ]

    survey_data = data[columns_of_interest]

    # Calculate percentages for each response type
    response_types = ["Yes", "No"]
    percentages = {col: survey_data[col].value_counts(normalize=True).reindex(response_types, fill_value=0) * 100 for col in columns_of_interest}

    data_values = np.array([
        percentages[columns_of_interest[0]].values
    ])

    # Colors for each category
    colorz = ['#5FBE78', '#F24837']
    categories = ['Valid', 'Invalid']  # Change legend labels

    fig, ax = plt.subplots(figsize=(12, 2))  # Adjust figure size for horizontal and thinner plot

    # Plotting the data
    cumulative_data = np.cumsum(data_values, axis=1)
    for i, (color, category) in enumerate(zip(colorz, categories)):
        ax.barh(columns_of_interest, data_values[:, i], left=cumulative_data[:, i] - data_values[:, i], color=color, label=category, edgecolor='white', height=0.5)  # Increase bar height

    # Adding percentage labels within the blocks
    for j in range(data_values.shape[0]):
        for i in range(data_values.shape[1]):
            if data_values[j, i] >= 5:
                ax.text(cumulative_data[j, i] - data_values[j, i] / 2, j, f"{data_values[j, i]:.0f}%", va='center', ha='center', color='white', fontsize=16, fontweight='bold')

    # Wrap y-tick labels and hide y-ticks
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.yaxis.set_ticks_position('none')

    # Customizing the plot to match the desired style
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5, frameon=False, prop={'size': 12})
    ax.set_xlim(0, 100)
    ax.xaxis.set_visible(False)  # Remove x-axis labels
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.3, right=0.7)  # Adjust left and right margins to center plot horizontally

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png', dpi=300)
    temp_file.close()

    # Create a PDF and embed the plot
    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=letter)
    width, height = letter

    # Get the dimensions of the image
    with Image.open(temp_file.name) as img:
        img_width, img_height = img.size

    # Calculate the dimensions to maintain the aspect ratio
    aspect = img_width / img_height
    new_width = width
    new_height = width / aspect

    # Ensure the image fits within the page
    if new_height > height:
        new_height = height
        new_width = height * aspect

    # Calculate the position for the image to be below the table
    table_height = 20  # approximate height of the table
    img_y_position = height - new_height - table_height - 50

    # Draw the image on the PDF below the table
    img = ImageReader(temp_file.name)
    c.drawImage(img, 0, img_y_position, width=new_width * 0.95,
                height=new_height * 0.95)  # Adjust dimensions to maintain aspect ratio

    c.showPage()
    c.save()
    pdf_buf.seek(0)

    # Clean up the temporary file
    os.unlink(temp_file.name)

    return pdf_buf

# file_path = r"C:\Users\jordan.gibbs\Downloads\fkh_processed_data (3).csv"
#
# pdf_buffer = create_reliability_report(file_path)
#
# # Save the PDF to a file for viewing
# with open("reliability_report.pdf", "wb") as f:
#     f.write(pdf_buffer.getbuffer())
