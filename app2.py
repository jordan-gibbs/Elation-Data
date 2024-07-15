import streamlit as st
import numpy as np
import pandas as pd
from report_data_parse import create_xlsx_report
import doc_creator3
import tempfile
from workplace_culture import create_culture_report, create_glossary_pdf, subgroup_table, merge_pdfs
from Data_Reliability import calculate_response_reliability_index, calculate_statistical_deviation_score


if 'demo' not in st.session_state:
    st.session_state['demo'] = None

if 'subgroup' not in st.session_state:
    st.session_state['subgroup'] = None


# Define function to extract scoring instructions
def extract_scoring_instructions(scoring_df):
    scoring_instructions = {}
    for _, row in scoring_df.iterrows():
        variable_name = row['Variable Name']
        question = row['Question']
        try:
            weight = float(row['Scoring'])
        except ValueError:
            # st.warning(f"Skipping row due to invalid weight: {row['Scoring']}")
            continue
        if variable_name not in scoring_instructions:
            scoring_instructions[variable_name] = []
        scoring_instructions[variable_name].append((question, weight))
    return scoring_instructions

# Define function to calculate scores
def calculate_scores(merged_df, scoring_instructions):
    for variable, instructions in scoring_instructions.items():
        total_score = sum([merged_df[question] * weight for question, weight in instructions])
        merged_df[variable] = total_score
    return merged_df

# Define function to back convert agree/disagree columns
def back_convert_agree_disagree(merged_df, columns, mapping):
    for column in columns:
        if column in merged_df.columns:
            merged_df[column] = merged_df[column].map(mapping)
    return merged_df

# Define function to extract final layout
def extract_final_layout(merged_df, final_layout_columns):
    final_df = merged_df[final_layout_columns]
    return final_df

# Streamlit app
st.title("Elation Data Processor")
org = st.text_input("Please Input the Organization Name")

st.header("Upload files")
demographics_file = st.file_uploader("Upload Demographics CSV", type="csv")
raw_data_file = st.file_uploader("Upload Raw Data CSV", type="csv")
scoring_instructions_df = pd.read_csv('CSVs/ER App - Scoring - 1. Assessment.csv')
final_layout_df = pd.read_csv('CSVs/ER App - Ideal Data Output Format.csv')

if demographics_file and raw_data_file:
    demographics_df = pd.read_csv(demographics_file)
    total_demo_df=demographics_df
    raw_data_df = pd.read_csv(raw_data_file)

    # Ensure the columns for merging are aligned
    demographics_df.rename(columns={'id': 'respondentId'}, inplace=True)

    # Merge data on respondentId
    merged_df = pd.merge(demographics_df, raw_data_df, on="respondentId", how="inner")

    # Check and fill Demographic 1 with groupName if groupName exists
    if 'groupName' in merged_df.columns:
        merged_df['Demographic 1'] = merged_df['groupName']
        print("'Demographic 1' assigned from 'groupName'")

    # Check if there are any columns ending with '_x'
    x_columns = [col for col in merged_df.columns if col.endswith('_x') and not col.startswith('score_')]

    if x_columns:
        demographic_columns = ['Demographic 2', 'Demographic 3', 'Demographic 4']
        for demo_col, x_col in zip(demographic_columns, x_columns):
            merged_df[demo_col] = merged_df[x_col]
            print(f"'{demo_col}' assigned from '{x_col}'")
            merged_df.drop(columns=[x_col], inplace=True)

    ### INSERT DEMOGRAPHIC 4 HERE IF NEEDED

    # Rename durationSeconds to Duration
    merged_df.rename(columns={'durationSeconds': 'Duration'}, inplace=True)

    # Extract scoring instructions
    scoring_instructions = extract_scoring_instructions(scoring_instructions_df)

    # Calculate scores
    merged_df = calculate_scores(merged_df, scoring_instructions)

    # Transform categorical data
    categorical_mapping = {
        100: 'Strongly agree',
        75: 'Agree',
        50: 'Neutral',
        25: 'Disagree',
        0: 'Strongly disagree'
    }

    agree_disagree_columns = [
        'Our leaders treat staff with respect.',
        'Staff treat each other with respect.',
        'My co-workers trust in me and each other.',
        'In my area, employees are treated fairly.',
        'I am empowered to investigate problems and explore new ideas at work.'
    ]

    xlsx_df = merged_df
    merged_df = back_convert_agree_disagree(merged_df, agree_disagree_columns, categorical_mapping)

    # Replace the userId column with respondentId (renamed to userId)
    merged_df['userId'] = merged_df['respondentId']


    # Create the list of final layout columns, including all columns from final_layout_df
    final_layout_columns = list(final_layout_df.columns)

    # Ensure Duration is in the final layout columns and handle capitalization
    if 'Duration' not in final_layout_columns:
        final_layout_columns.append('Duration')

    # Insert the new userId at the beginning of the final layout columns
    if 'userId' not in final_layout_columns:
        final_layout_columns.insert(0, 'userId')

    # Ensure final_layout_columns only includes columns that exist in merged_df
    final_layout_columns = [col for col in final_layout_columns if col in merged_df.columns]

    # Proceed with extracting the final layout
    final_df = merged_df[final_layout_columns]

        raw_data_df.rename(columns={'respondentId': 'userId'}, inplace=True)

    # Calculate the Response Reliability Index using raw_data_df
    raw_data_df = calculate_response_reliability_index(raw_data_df)

    raw_data_df = calculate_statistical_deviation_score(raw_data_df)

    # Append the new columns to final_df
    final_df = final_df.merge(raw_data_df[['userId', 'Response Reliability Index', 'Social Desirability Score', 'Absolute Z-score', 'Above 95% threshold']],
                              on='userId', how='left')

    final_df['Valid Response'] = np.where(
        ((final_df['Response Reliability Index'] <= 6).astype(int) +
         (final_df['Social Desirability Score'] <= 50).astype(int) +
         (final_df['Above 95% threshold'] == 'Yes').astype(int)) >= 2,
        'No', 'Yes'
    )

    # Round all values to the nearest whole number
    final_df = final_df.round()

    # Download link for the final dataframe
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Processed Data (Raw)",
        data=csv,
        file_name=f'{org}_processed_data.csv',
        mime='text/csv',
    )
