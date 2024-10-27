import PyPDF2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import glob
import os


# color vars for nice display
MAIN_COLOUR = '\033[94m' #blue
SECOND_COLOUR  = '\033[96m' #light blue
RED = '\033[91m'
GRAY = '\033[90m'
ENDC = '\033[0m'  #resets to default

# functions
def split_text_into_paragraphs(text, paragraph_size=100):
    # Split the text into words
    words = text.split(' ')
    paragraphs = []
    # Create paragraphs with the specified number of words
    for i in range(0, len(words), paragraph_size):
        paragraph = ' '.join(words[i:i + paragraph_size])
        paragraphs.append(paragraph)
    # merge the last paragraph if too short
    merge_threshold = paragraph_size//2
    if len(paragraphs) > 1 and len(paragraphs[-1].split()) < merge_threshold:
        additional_part = paragraphs.pop()
        paragraphs[-1] += ' ' + additional_part
    return paragraphs


#--- main code starts here ---

# load the semantich model
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f'{MAIN_COLOUR}------ WELCOME TO SEMANTIC SEARCH APPLICATION! ------')
print('Note: this model supports English, Spanish, French, German, Italian, Dutch and Portuguese')

# while brakes when the user decide to stop searching
while True:
    # user input + input check
    new_kb = input(f'{SECOND_COLOUR}Is the knowledge base changed since the last time? Have you uploaded or deleted pdf? (yes/no)\n{GRAY}Type your response here --> {ENDC}').lower().strip()
    while new_kb not in ['yes', 'no']:
        new_kb = input(f'{RED}The input is not valid. Please enter "yes" or "no"\n{GRAY}Type your response here --> {ENDC}')

    # computing the vector embeddings
    if new_kb == 'yes':
        df = pd.DataFrame(columns=['file_name', 'page', 'text', 'embedding'])
        # list all .pdf in the directory (relative paths)
        files = glob.glob('knowledge_base/*.pdf')
        print(f'{MAIN_COLOUR}PDF files found in "knowledge_base" and processed:')
        for file in files:
            file_name = os.path.basename(file)
            print('\t'+file_name)
            # read pdf and populate the df
            with open(file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    emb_text = model.encode(text)
                    paragraphs = split_text_into_paragraphs(text)
                    for par in paragraphs:
                        emb_text = model.encode(par)
                        df.loc[len(df)] = [file_name, page_num+1, par, emb_text]

            # save df
            df.to_pickle('df.pkl')
    else:
        df = pd.read_pickle('df.pkl')

    # user query
    query = input(f"{SECOND_COLOUR}Please describe what you are looking for:\n{GRAY}Type your response here --> {ENDC}")
    query_embedding = model.encode(query)

    # compute the cosine similarity
    embeddings_matrix = np.vstack(df['embedding'].values) # convertion to a NumPy array for better efficiency
    similarities = model.similarity(query_embedding, embeddings_matrix)[0]
    df['similarity_score'] = similarities

    # evaluate each page based on the avg of the top 3 paragraphs (the other are dropped, so ininfluential)
    # row number over ordered partition + keeps rows where row_number <= 2 + group by average
    df = df.sort_values(['file_name', 'page', 'similarity_score'], ascending=[True, True, False])  # Sort by partition column and then by value within each partition
    df['row_number'] = df.groupby(['file_name', 'page']).cumcount()
    df = df[df['row_number'] <= 2]
    df_agg = df.groupby(['file_name', 'page'])['similarity_score'].mean().reset_index()
    df_agg = df_agg.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    
    # nicely print the search results
    df_agg['rank'] = range(1, len(df_agg)+1)
    df_agg['rank'] = df_agg['rank'].apply(lambda x: f'{x: <4}') # for displaying nicely
    print(f'\n{MAIN_COLOUR}The table below shows where you can find relevant info\n')
    print(df_agg[['rank','file_name','page','similarity_score']].head(50).to_string(index=False, col_space=10))
    
    # user input + input check
    keep_searching = input(f'{SECOND_COLOUR}\nWould you like to do another search? (yes/no)\n{GRAY}Type your response here --> {ENDC}').lower().strip()
    while keep_searching not in ['yes', 'no']:
        keep_searching = input(f'{RED}The input is not valid. Please enter "yes" or "no"\n{GRAY}Type your response here --> {ENDC}')
    if keep_searching == 'no':
        print(f'{MAIN_COLOUR}Okay, good bye\n')
        break
    