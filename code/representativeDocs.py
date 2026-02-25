"""
Extract representative documents from BERTopic model and export to Excel.
Each topic gets its own sheet with representative docs and metadata.
"""

import os
import pandas as pd
from bertopic import BERTopic
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

# Configuration
project_root = os.path.expanduser("~/Uncivil-Religion-2.0")
model_path = os.path.join(project_root, "bertopicOutput/Rescraped01/bertopic_model")
output_excel = os.path.join(project_root, "bertopicOutput/Rescraped01/representative_docs.xlsx")

print("=" * 80)
print("📊 EXTRACTING REPRESENTATIVE DOCUMENTS FROM BERTOPIC MODEL")
print("=" * 80)

# Load the saved model
print(f"\n📂 Loading BERTopic model from: {model_path}")
topic_model = BERTopic.load(model_path)
print("✅ Model loaded successfully")

# Get topic information
topic_info = topic_model.get_topic_info()
print(f"\n📈 Found {len(topic_info)} topics (including outliers)")

# Get representative documents for each topic
print("\n🔍 Extracting representative documents...")

all_data = []

for topic_id in topic_info['Topic'].tolist():
    # Get topic details
    topic_row = topic_info[topic_info['Topic'] == topic_id].iloc[0]
    topic_name = topic_row['Name'] if 'Name' in topic_row else f"Topic_{topic_id}"
    topic_count = topic_row['Count']
    
    # Get representative documents for this topic
    rep_docs = topic_model.get_representative_docs(topic_id)
    
    if rep_docs:
        # Get top words for context
        topic_words = topic_model.get_topic(topic_id)
        top_words = ", ".join([word for word, _ in topic_words[:10]]) if topic_words else "N/A"
        
        # Add each representative doc as a row
        for idx, doc in enumerate(rep_docs, 1):
            all_data.append({
                'Topic_ID': topic_id,
                'Topic_Name': topic_name,
                'Topic_Size': topic_count,
                'Top_Words': top_words,
                'Rep_Doc_Number': idx,
                'Document_Length': len(doc),
                'Document_Text': doc
            })
    
    print(f"  Topic {topic_id}: {len(rep_docs)} representative docs")

# Create DataFrame
df = pd.DataFrame(all_data)

print(f"\n✅ Extracted {len(df)} total representative documents across {len(topic_info)} topics")

# Export to Excel with formatting
print(f"\n💾 Saving to Excel: {output_excel}")

# Create Excel writer
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Write summary sheet
    summary_df = df.groupby('Topic_ID').agg({
        'Topic_Name': 'first',
        'Topic_Size': 'first',
        'Top_Words': 'first',
        'Rep_Doc_Number': 'count'
    }).reset_index()
    summary_df.columns = ['Topic_ID', 'Topic_Name', 'Topic_Size', 'Top_Words', 'Num_Rep_Docs']
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Write all representative docs to one sheet
    df.to_excel(writer, sheet_name='All_Representative_Docs', index=False)

# Format the Excel file
print("🎨 Applying formatting...")
wb = load_workbook(output_excel)

for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    
    # Format header row
    for cell in ws[1]:
        cell.font = Font(bold=True, size=12)
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.font = Font(bold=True, color="FFFFFF", size=12)
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Auto-adjust column widths
    for idx, column in enumerate(ws.columns, 1):
        column_letter = get_column_letter(idx)
        max_length = 0
        
        for cell in column:
            if cell.value:
                cell_length = len(str(cell.value))
                if cell_length > max_length:
                    max_length = cell_length
        
        # Set width (with limits)
        adjusted_width = min(max_length + 2, 100)  # Cap at 100
        if column_letter == get_column_letter(ws.max_column):  # Document text column
            adjusted_width = 80  # Fixed width for document text
        
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Wrap text in document column
    if sheet_name != 'Summary':
        doc_col = get_column_letter(ws.max_column)
        for cell in ws[doc_col]:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    
    # Freeze header row
    ws.freeze_panes = 'A2'

wb.save(output_excel)

print("✅ Excel file created and formatted successfully")

# Print summary statistics
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print(f"Total topics analyzed: {len(topic_info)}")
print(f"Total representative documents: {len(df)}")
print(f"Average docs per topic: {len(df) / len(topic_info):.1f}")
print(f"Output file: {output_excel}")
print("\n📁 Excel contains:")
print("  • Summary sheet: Overview of all topics")
print("  • All_Representative_Docs: All docs in one sheet")
print("=" * 80)