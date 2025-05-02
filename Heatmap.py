import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
file_path = "Student Learning & Coding Assessment – Adaptive Evaluation Form  (Responses) - Form Responses 1.csv"
data = pd.read_csv(file_path)

# Clean column names by removing extra spaces
data.columns = [col.strip() for col in data.columns]

# Define columns of interest using partial matching
def get_column_by_partial_match(df, partial_name):
    matches = [col for col in df.columns if partial_name.lower() in col.lower()]
    if matches:
        return matches[0]
    else:
        print(f"Warning: No column matching '{partial_name}' found")
        return None

# Get key columns
confidence_col = get_column_by_partial_match(data, "confident are you in your programming")
practice_col = get_column_by_partial_match(data, "practice coding")

# Identify code question columns
code_questions = []
for col in data.columns:
    if "What is the output of the following code" in col:
        code_questions.append(col)

print(f"Found {len(code_questions)} code questions")

# Define correct answers using keyword matching
correct_answers = {}
for question in code_questions:
    if "print(2 + 3 * 4)" in question:
        correct_answers[question] = "14"
    elif "num = 8" in question and "if num % 2 == 0" in question:
        correct_answers[question] = "Option 1"  # "Even"
    elif "x = \"Hello\"" in question and "y = \"World\"" in question:
        correct_answers[question] = "Hello World"
    elif "a = 10" in question and "b = 5" in question and "a, b = b, a" in question:
        correct_answers[question] = "5 10"
    elif "fruits = [\"apple\", \"banana\", \"cherry\"]" in question:
        correct_answers[question] = "banana"
    elif "len(\"Python\")" in question:
        correct_answers[question] = "6"
    elif "x = [1, 2, 3]" in question and "x.append(4)" in question:
        correct_answers[question] = "[1, 2, 3, 4]"
    elif "def greet(name)" in question:
        correct_answers[question] = "Hello Alex"
    elif "type(3.14)" in question:
        correct_answers[question] = "<class 'float'>"  # Also accepting "float"
    elif "my_dict = {\"name\": \"Alex\", \"age\": 21}" in question:
        correct_answers[question] = "Alex"

# Get difficulty rating columns
difficulty_cols = [col for col in data.columns if "How easy was" in col]
print(f"Found {len(difficulty_cols)} difficulty rating columns")

# Get post-exam assessment columns
post_exam_cols = [
    get_column_by_partial_match(data, "confident were you before the exam"),
    get_column_by_partial_match(data, "confident were you during the exam"),
    get_column_by_partial_match(data, "understand the questions provided"),
    get_column_by_partial_match(data, "challenging did you find the exam"),
    get_column_by_partial_match(data, "relevant were the questions"),
    get_column_by_partial_match(data, "exam format suit your learning")
]

# Create accuracy columns for each question
accuracy_cols = []
for i, question in enumerate(code_questions):
    if question in correct_answers:
        correct_answer = correct_answers[question]
        col_name = f"Q{i+1}_Correct"
        
        # For type question, accept multiple formats
        if "type(3.14)" in question:
            data[col_name] = data[question].apply(
                lambda x: 1 if (x == correct_answer or x == "float" or x == "<type 'float'>") else 0
            )
        # For Hello World, accept with and without space
        elif "Hello World" in correct_answer:
            data[col_name] = data[question].apply(
                lambda x: 1 if (str(x).strip() == "Hello World" or str(x).strip() == "HelloWorld") else 0
            )
        else:
            data[col_name] = data[question].apply(
                lambda x: 1 if str(x).strip() == correct_answer else 0
            )
        accuracy_cols.append(col_name)

# Map practice frequency to numeric values
practice_mapping = {
    "Never": 0, 
    "Rarely": 1, 
    "Once a week": 2, 
    "A few times a week": 3, 
    "Daily": 4
}

# Map confidence to numeric (in case it's not already)
confidence_mapping = {
    "Very Low": 1,
    "Low": 2,
    "Average": 3,
    "High": 4,
    "Very High": 5
}

# Create numeric columns
if practice_col:
    data["Practice_Numeric"] = data[practice_col].map(practice_mapping)
else:
    data["Practice_Numeric"] = 0  # Default if column not found

if confidence_col:
    # Check if confidence is already numeric
    if data[confidence_col].dtype == 'object':
        data["Confidence_Numeric"] = data[confidence_col].map(confidence_mapping)
    else:
        data["Confidence_Numeric"] = data[confidence_col]
else:
    data["Confidence_Numeric"] = 0  # Default if column not found

# Calculate total score and percentage
data["Total_Score"] = data[accuracy_cols].sum(axis=1)
data["Score_Percentage"] = (data["Total_Score"] / len(accuracy_cols)) * 100

# Calculate individual difficulty perceptions
for i, col in enumerate(difficulty_cols):
    col_name = f"Difficulty_Q{i+1}"
    data[col_name] = data[col].astype(float)

# Calculate average difficulty perception
data["Avg_Perceived_Difficulty"] = data[difficulty_cols].mean(axis=1)

# Prepare data for our heatmap
heatmap_data = pd.DataFrame()

# Add columns to heatmap data
heatmap_data["Self_Reported_Confidence"] = data["Confidence_Numeric"]
heatmap_data["Practice_Frequency"] = data["Practice_Numeric"]
heatmap_data["Score_Percentage"] = data["Score_Percentage"]
heatmap_data["Avg_Perceived_Difficulty"] = data["Avg_Perceived_Difficulty"]

# Add post-exam columns if they exist
column_names = ["Pre_Exam_Confidence", "During_Exam_Confidence", 
               "Question_Understanding", "Exam_Challenge_Level",
               "Question_Relevance", "Learning_Style_Fit"]

for i, col in enumerate(post_exam_cols):
    if col and i < len(column_names):
        heatmap_data[column_names[i]] = data[col]

# Calculate correlations
correlation_matrix = heatmap_data.corr()

#---------------------------------------------------------------------------
# 1. Create Heatmap
#---------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap: Student Assessment Parameters", fontsize=16)
plt.tight_layout()
plt.savefig("student_assessment_heatmap.png", dpi=300)
# plt.show() # Show heatmap separately if desired, or wait until the end

#---------------------------------------------------------------------------
# Prepare data for graphical breakdowns
#---------------------------------------------------------------------------

# Confidence-Performance Data
confidence_labels = {1: "Very Low", 2: "Low", 3: "Average", 4: "High", 5: "Very High"}
data['Confidence_Label'] = data['Confidence_Numeric'].map(confidence_labels)
confidence_groups = data.groupby('Confidence_Label', observed=False)["Score_Percentage"].agg(['mean', 'count'])
confidence_groups['percentage'] = (confidence_groups['count'] / confidence_groups['count'].sum()) * 100
confidence_groups = confidence_groups.reindex(confidence_labels.values()) # Ensure order

# Practice Frequency Data
practice_labels = {0: "Never", 1: "Rarely", 2: "Once a week", 3: "Few times/week", 4: "Daily"}
data['Practice_Label'] = data['Practice_Numeric'].map(practice_labels)
practice_groups = data.groupby('Practice_Label', observed=False)["Score_Percentage"].agg(['mean', 'count'])
practice_groups['percentage'] = (practice_groups['count'] / practice_groups['count'].sum()) * 100
practice_groups = practice_groups.reindex(practice_labels.values()) # Ensure order
daily_score = practice_groups.loc['Daily', 'mean'] if 'Daily' in practice_groups.index else None

# Perception-Reality Data
data['Difficulty_Category'] = pd.cut(
    data['Avg_Perceived_Difficulty'], 
    bins=[0, 2, 3, 4, 5.1], 
    labels=['Very Difficult (1-2)', 'Difficult (2-3)', 'Easy (3-4)', 'Very Easy (4-5)'],
    right=False # Bins include left edge, exclude right edge
)
difficulty_groups = data.groupby('Difficulty_Category', observed=False)["Score_Percentage"].agg(['mean', 'count'])
difficulty_groups['percentage'] = (difficulty_groups['count'] / difficulty_groups['count'].sum()) * 100

# Confidence Stability Data
if post_exam_cols[0] and post_exam_cols[1]:
    data['Pre_Confidence'] = data[post_exam_cols[0]].astype(float)
    data['During_Confidence'] = data[post_exam_cols[1]].astype(float)
    data['Confidence_Change'] = data['During_Confidence'] - data['Pre_Confidence']
    
    data['Change_Category'] = pd.cut(
        data['Confidence_Change'],
        bins=[-float('inf'), -1.5, -0.5, 0.5, 1.5, float('inf')], # Adjusted bins for clarity
        labels=['Major Decrease (<-1)', 'Decrease (-1)', 'Stable (0)', 
                'Increase (1)', 'Major Increase (>1)']
    )
    change_groups = data.groupby('Change_Category', observed=False)["Score_Percentage"].agg(['mean', 'count'])
    change_groups['percentage'] = (change_groups['count'] / change_groups['count'].sum()) * 100
else:
    change_groups = None # Cannot calculate if columns are missing

#---------------------------------------------------------------------------
# 2. Generate Graphical Breakdowns
#---------------------------------------------------------------------------

# Create a figure with subplots for the breakdowns
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Parameter Breakdowns', fontsize=20, y=1.03)

# --- Plot 1: Confidence-Performance ---
ax1 = axes[0, 0]
bars1 = sns.barplot(x=confidence_groups.index, y=confidence_groups['mean'], ax=ax1, palette="viridis")
ax1.set_title('1. Confidence vs. Performance', fontsize=14)
ax1.set_ylabel('Average Score (%)', fontsize=12)
ax1.set_xlabel('Self-Reported Confidence', fontsize=12)
ax1.set_ylim(0, 105)
ax1.tick_params(axis='x', rotation=45)
# Add annotations (percentage of students)
for i, bar in enumerate(bars1.patches):
    height = bar.get_height()
    percentage = confidence_groups['percentage'].iloc[i]
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
             f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

# --- Plot 2: Practice Frequency Impact ---
ax2 = axes[0, 1]
bars2 = sns.barplot(x=practice_groups.index, y=practice_groups['mean'], ax=ax2, palette="magma")
ax2.set_title('2. Practice Frequency vs. Performance', fontsize=14)
ax2.set_ylabel('Average Score (%)', fontsize=12)
ax2.set_xlabel('Practice Frequency', fontsize=12)
ax2.set_ylim(0, 105)
ax2.tick_params(axis='x', rotation=45)
# Add annotations (percentage of students)
for i, bar in enumerate(bars2.patches):
    height = bar.get_height()
    percentage = practice_groups['percentage'].iloc[i]
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
             f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

# --- Plot 3: Perception-Reality Gap ---
ax3 = axes[1, 0]
bars3 = sns.barplot(x=difficulty_groups.index, y=difficulty_groups['mean'], ax=ax3, palette="coolwarm")
ax3.set_title('3. Perceived Difficulty vs. Performance', fontsize=14)
ax3.set_ylabel('Average Score (%)', fontsize=12)
ax3.set_xlabel('Average Perceived Difficulty', fontsize=12)
ax3.set_ylim(0, 105)
ax3.tick_params(axis='x', rotation=45)
# Add annotations (percentage of students)
for i, bar in enumerate(bars3.patches):
    height = bar.get_height()
    percentage = difficulty_groups['percentage'].iloc[i]
    ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
             f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

# --- Plot 4: Confidence Stability ---
ax4 = axes[1, 1]
if change_groups is not None:
    bars4 = sns.barplot(x=change_groups.index, y=change_groups['mean'], ax=ax4, palette="plasma")
    ax4.set_title('4. Confidence Stability vs. Performance', fontsize=14)
    ax4.set_ylabel('Average Score (%)', fontsize=12)
    ax4.set_xlabel('Confidence Change (During - Before)', fontsize=12)
    ax4.set_ylim(0, 105)
    ax4.tick_params(axis='x', rotation=45)
    # Add annotations (percentage of students)
    for i, bar in enumerate(bars4.patches):
        height = bar.get_height()
        percentage = change_groups['percentage'].iloc[i]
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
else:
    ax4.text(0.5, 0.5, 'Confidence stability data not available\n(Missing pre/during confidence columns)', 
             ha='center', va='center', fontsize=12, color='red')
    ax4.set_title('4. Confidence Stability vs. Performance', fontsize=14)
    ax4.set_xticks([])
    ax4.set_yticks([])

# Adjust layout and display plots
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
plt.show()