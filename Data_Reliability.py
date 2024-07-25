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

    # Select only numeric response columns starting from the specified column to the end column
    response_columns = data.columns[start_index:]
    response_data = data[response_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with all NaN values
    response_data = response_data.dropna(how='all')

    # Calculate means and standard deviations, dropping rows with zero standard deviation
    means = response_data.mean(axis=1)
    stds = response_data.std(axis=1)

    # Drop rows with zero standard deviation
    valid_rows = stds != 0
    response_data = response_data[valid_rows]
    means = means[valid_rows]
    stds = stds[valid_rows]

    # Compute Z-scores for each response
    z_scores = (response_data.sub(means, axis=0)).div(stds, axis=0)

    # Calculate the average Z-score for each row
    average_z_scores = z_scores.abs().mean(axis=1)

    mask = (z_scores < -3) | (z_scores > 3)

    # Count the number of True values in each row
    count_extreme_z_scores = mask.sum(axis=1)

    # Ensure the indices match the original data
    average_z_scores = average_z_scores.reindex(data.index, fill_value=np.nan)
    count_extreme_z_scores = count_extreme_z_scores.reindex(data.index, fill_value=np.nan)

    # Add the 'Average Z-score' to the original DataFrame
    data['Absolute Z-score'] = average_z_scores
    data['Z-score Anomaly Count'] = count_extreme_z_scores

    # Fill NaN values in 'Z-score Anomaly Count' with 0
    data['Z-score Anomaly Count'] = data['Z-score Anomaly Count'].fillna(0)

    data['Z-score Anomaly Count'] = data['Z-score Anomaly Count'].astype(float)

    # Determine the 95th Percentile Threshold
    threshold = np.percentile(data['Z-score Anomaly Count'].dropna(), 95)

    # Identify Outliers
    data['Above 95% threshold'] = np.where(data['Z-score Anomaly Count'] > threshold, 'Yes', 'No')

    return data


def create_reliability_report(final_df, total_members):
    # Extract relevant columns
    columns_of_interest = ["Valid Response"]

    survey_data = final_df[columns_of_interest]

    # Calculate counts for each response type
    response_types = ["Yes", "No"]
    counts = survey_data[columns_of_interest[0]].value_counts().reindex(response_types, fill_value=0)

    # Calculate valid, invalid, and non-respondents
    valid_count = counts.get("Yes", 0)
    invalid_count = counts.get("No", 0)
    non_respondents_count = total_members - valid_count - invalid_count

    # Calculate percentages
    valid_percentage = (valid_count / total_members) * 100
    invalid_percentage = (invalid_count / total_members) * 100
    non_respondents_percentage = (non_respondents_count / total_members) * 100

    # Prepare data for plotting
    data_values = np.array([[valid_percentage, invalid_percentage, non_respondents_percentage]])

    # Colors for each category
    colorz = ['#A6CF5C', '#DE9C95', '#D3D3D3']
    categories = ['Valid Responses', 'Invalid Responses', 'Non-respondents']

    fig, ax = plt.subplots(figsize=(12, 2))

    # Plotting the data
    cumulative_data = np.cumsum(data_values, axis=1)
    for i, (color, category) in enumerate(zip(colorz, categories)):
        ax.barh(columns_of_interest, data_values[:, i], left=cumulative_data[:, i] - data_values[:, i], color=color, label=category, edgecolor='white', height=0.5)

    # Adding percentage labels within the blocks
    for j in range(data_values.shape[0]):
        for i in range(data_values.shape[1]):
            if data_values[j, i] >= 5:
                ax.text(cumulative_data[j, i] - data_values[j, i] / 2, j, f"{data_values[j, i]:.0f}%",
                        va='center', ha='center', color='white', fontsize=16, fontweight='bold')

    ax.set_yticklabels([])
    ax.yaxis.set_ticks_position('none')

    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5, frameon=False, prop={'size': 14})
    ax.set_xlim(0, 100)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    # Save the plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png', dpi=600, transparent=True)
    plt.close()

    return temp_file.name

# # Example usage:
# file_path = r"C:\Users\jordan.gibbs\Downloads\fkh_processed_data (3).csv"
# total_members = 200  # Example total members
# pdf_buffer = create_reliability_report(file_path, total_members)
#
# # Save the PDF to a file for viewing
# with open("reliability_report.pdf", "wb") as f:
#     f.write(pdf_buffer.getbuffer())
