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
        df = pd.DataFrame(columns=['text','page','file_name','embedding'])
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
                    df.loc[len(df)] = [text, page_num+1, file_name, emb_text]

            # save df
            df.to_pickle('df.pkl')
    else:
        df = pd.read_pickle('df.pkl')

    # user query
    query = input(f"{SECOND_COLOUR}Please describe what you are looking for:\n{GRAY}Type your response here --> {ENDC}")
    query_embedding = model.encode(query)

    # Compute the cosine similarity
    embeddings_matrix = np.vstack(df['embedding'].values) # convertion to a NumPy array for better efficiency
    similarities = model.similarity(query_embedding, embeddings_matrix)[0]

    # nicely print the search results
    df['similarity_score'] = similarities
    df = df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df)+1)
    df['rank'] = df['rank'].apply(lambda x: f'{x: <4}') # for displaying nicely
    print(f'\n{MAIN_COLOUR}The table below shows where you can find relevant info\n')
    print(df[['rank','file_name','page','similarity_score']].head(50).to_string(index=False, col_space=10))
    
    # user input + input check
    keep_searching = input(f'{SECOND_COLOUR}\nWould you like to do another search? (yes/no)\n{GRAY}Type your response here --> {ENDC}').lower().strip()
    while keep_searching not in ['yes', 'no']:
        keep_searching = input(f'{RED}The input is not valid. Please enter "yes" or "no"\n{GRAY}Type your response here --> {ENDC}')
    if keep_searching == 'no':
        print(f'{MAIN_COLOUR}Okay, good bye\n')
        break
    
