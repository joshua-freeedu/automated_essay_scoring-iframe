# To run streamlit, go to terminal and type: 'streamlit run app.py'
# Core Packages ###########################
import os
import shutil
from datetime import datetime

import docx2txt
import PyPDF2

import streamlit as st
import pandas as pd

from model import BertLightningModel

import openai
import base64

#######################################################################################################################
current_path = os.path.abspath(os.path.dirname(__file__))
#######################################################################################################################
@st.cache(allow_output_mutation=True)
def load_model():
    CONFIG = dict(
        model_name="microsoft/deberta-v3-large",
        num_classes=6,
        lr=2e-5,

        batch_size=8,
        num_workers=8,
        max_length=512,
        weight_decay=0.01,

        accelerator='gpu',
        max_epochs=5,
        accumulate_grad_batches=4,
        precision=16,
        gradient_clip_val=1000,
        train_size=0.8,
        num_cross_val_splits=5,
        num_frozen_layers=20,  # out of 24 in deberta
    )
    model = BertLightningModel.load_from_checkpoint(os.path.join(current_path,'tf_model.ckpt'),config=CONFIG, map_location='cpu')

    return model

def predict(_input, _model):
    tokens = _model.tokenizer([_input], return_tensors='pt')
    outputs = _model(tokens)[0].tolist()

    df = pd.DataFrame({
        'Criterion': ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'],
        'Grade': outputs
    })

    return df

def convert_ave_to_score_range(score, max, min):
    fg = (score-1) * ((max-min)/3) + min
    return fg

def run_model(answer, min_score, max_score):
    evaluation = 0

    st.write('Grading essay..')
    evaluation = predict(answer,st.session_state['model'])

    # get the average of the score evaluations
    ave = evaluation['Grade'].mean()

    grade = convert_ave_to_score_range(ave, max_score, min_score)
    grade = round(grade)
    final_grade = max_score if max_score < grade else grade

    return evaluation, final_grade


def run_model_on_list(answers, min_score, max_score):
    evaluations = []
    final_grades = []

    for answer in answers:
        st.write(f'Grading essay #{answers.index(answer)+1}..')
        evaluations.append(predict(answer,st.session_state['model']))
        ave = evaluations[answers.index(answer)]['Grade'].mean()

        grade = convert_ave_to_score_range(ave, max_score, min_score)
        grade = round(grade)
        final_grades.append(max_score if max_score < grade else grade)

    return evaluations, final_grades

def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)
    count = len(pdfReader.pages)
    all_page_text = ""
    for i in range(count):
        page = pdfReader.pages[i]
        all_page_text += page.extract_text()

    return all_page_text

def openai_chat(prompt, model, max_tokens):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = response.choices[0].text.strip()
    return message

def run_chatgpt(essay_list, min_score, max_score):
    st.markdown("***")
    st.subheader("ChatGPT Evaluation")
    openai.api_key = os.environ["JOSHUA_FREEEDU_OPENAI_API_KEY"]

    chatgpt_prompts = []
    chatgpt_responses = []
    prompt = f"Evaluate the following essay using the Criterion: [cohesion, syntax, vocabulary, phraseology, grammar, conventions]. " \
             f"Use a {min_score} to {max_score} score range for each, and provide one final score using the same score range. " \
             f"Give some explanation for each score on each criteria, and one summarized feedback on the whole essay.\n"
    for i, answer in enumerate(essay_list):
        if i == 0:
            prompt += "\nEssay: \n"
        else:
            prompt = "Essay: \n"
        prompt += answer
        response = openai_chat(prompt=prompt, model="text-davinci-003", max_tokens=1024)
        # response = openai_chat(prompt=prompt, model="text-curie-001", max_tokens=1024)
        chatgpt_prompts.append(prompt)
        chatgpt_responses.append(response)

    chatgpt_prompt_val = ""
    chatgpt_response_val = ""
    for i, val in enumerate(chatgpt_prompts):
        chatgpt_prompt_val = chatgpt_prompt_val + val + "\n"
        chatgpt_response_val = chatgpt_response_val + chatgpt_responses[i] + "\n"
    chatgpt_prompt_ta = st.text_area("ChatGPT Prompt",
                                     placeholder="Prompt used on ChatGPT will display here.",
                                     value=chatgpt_prompt_val, height=500, disabled=True)
    chatgpt_response_ta = st.text_area("ChatGPT Response",
                                       placeholder="ChatGPT's evaluations will display here.",
                                       value=chatgpt_response_val, height=500, disabled=True)

    return chatgpt_response_ta

def main():
    uploaded_files = st.file_uploader('Upload Files', accept_multiple_files=True, type=['docx','txt','pdf'])
    essays = [] # List of essays extracted from uploaded files
    filenames = [] # list of the filenames; used in the final output dataframe
    ta_val = "" # Value for the text area
    upload_flag = False
    eval_flag = False
    st.session_state['model'] = load_model()

    #If a file/s is uploaded, disable input in the text area; then, display the essays list
    if uploaded_files:
        upload_flag = True

        # Create fresh temp folder for the uploaded files
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        os.makedirs("temp")

        # iterate through each uploaded file
        for uploaded_file in uploaded_files:
            contents = ""
            filenames.append(uploaded_file.name) # Add each file name to the list

            # Save each uploaded file to temp folder
            with open(os.path.join("temp", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

                # Parse the contents of the uploaded file according to their extension txt docx or pdf
                if uploaded_file.name.split(".")[-1] == "docx": # docx files
                    contents += docx2txt.process(os.path.join("temp", uploaded_file.name)) + "\n"

                elif uploaded_file.name.split(".")[-1] == "pdf": # pdf files
                    contents += read_pdf(uploaded_file) + "\n"

                else: # txt files
                    for line in uploaded_file.getvalue().decode().splitlines():
                        contents += line + "\n"

            #Add the compiled contents of the file into the 'essays' list before going to the next uploaded file
            essays.append(contents)

            #ta_val will be the preview of all the essays in the text area; display index numbering if there are more than one file
            ta_val += f"[{uploaded_files.index(uploaded_file)}]\n" + contents + "\n" if len(uploaded_files)>1 else contents

        shutil.rmtree("temp")

    # text area input for the essay, button to run the model, other widgets
    response_ta = st.text_area("Essay:",placeholder="Input your essay here.",height=500, value=ta_val, disabled=upload_flag)
    col1,col2,col3 = st.columns(3)
    min_score = col1.number_input('Minimum Score',0,100,0)
    max_score = col2.number_input('Maximum Score',0,100,10)
    run_button = st.button("Grade Essay")
    enable_chatgpt = st.checkbox("Evaluate with ChatGPT?", help="Works best on one essay at a time.")

    # run the model when the button is clicked
    if run_button:
        if not response_ta: # if the text area is empty:
            st.error("Please input the essay in the corresponding text area.")
        elif min_score >= max_score:
            st.error("Minimum score must be less than maximum score.")
        else: # run model
            if not upload_flag:
                eval_df, score = run_model(answer=response_ta, min_score=min_score, max_score=max_score)

                # output message template
                msg = f"Your essay score is: {score} (Minimum Possible Score: {min_score} | Maximum Possible Score: {max_score})"
                st.write(msg)
                st.write("Score breakdown (1-4):")
                st.dataframe(eval_df)

            else:
                # 'evals' is a list of dataframes [DataFrame]
                # 'scores' is a list of the grades [int]
                evals, scores = run_model_on_list(essays, min_score, max_score)

                # Display the final grade for each uploaded file
                grades_df = pd.DataFrame({'Filename':filenames,'Final Grade':scores})
                st.write("Grading done!")
                st.dataframe(grades_df)

                st.write("Criteria are graded within the range of 1-4. \nMerging grades with evaluations..")

                # Add a column 'Filename' to each set of evaluation, and set the value to the corresponding file name
                for f in filenames:
                    evals[filenames.index(f)]['Filename'] = f

                # Combine the list of evaluation dataframes 'evals' into one single dataframe 'evals_df'
                evals_df = pd.concat([df for df in evals])

                # Combine the Grades with the Evaluations, then show it
                final_df = grades_df.merge(evals_df, on='Filename')
                st.dataframe(final_df)

                eval_flag = True
                st.session_state["final_df"] = final_df

            # ChatGPT Evaluation Section
            if enable_chatgpt:
                chatgpt_response = run_chatgpt(essays, min_score, max_score)

    if eval_flag:
        # Old button for downloading the combined grades and evaluations into a csv file
        # st.download_button("Download results", data=downloadfile, file_name=f'aes_result_{curr_time}.csv', mime='text/csv')

        # New: Download links (links don't refresh the web page after clicking
        downloadfile = final_df.to_csv().encode('utf-8')
        curr_time = datetime.now().strftime("%b-%d-%Y %H:%M:%S")

        b64 = base64.b64encode(downloadfile).decode()
        download_link = f'<a href="data:text/csv;base64,{b64}" download="aes_result_{curr_time}.csv">Download results</a>'
        st.markdown(download_link, unsafe_allow_html=True)

        if enable_chatgpt:
            # Add a download link to the file
            b64 = base64.b64encode(chatgpt_response.encode()).decode()
            chatgpt_download_link = f'<a href="data:text/plain;base64,{b64}" download="aes_chatgpt_result_{curr_time}.txt">Download ChatGPT Feedback</a>'
            st.markdown(chatgpt_download_link, unsafe_allow_html=True)

    ###################################################################################################################
    # examples section
    st.subheader("")
    st.markdown("***")
    st.subheader("")

    # generate examples dropdown
    st.subheader("Here are a few example essays:")
    examples = {}
    examples_fnames = []
    examples_dir = os.path.join(current_path,'examples')
    for ex in os.listdir(examples_dir):
        examples[ex] = open(os.path.join(examples_dir, ex), 'rb')
        examples_fnames.append(ex)

    selected_example = st.multiselect('Select an example essay:',examples_fnames)
    ex_names = []
    ex_essays = []
    ta_val_ex = ""


    # iterate through each selected example
    for example in selected_example:
        contents_ex = ""  # Compile all the essays from each file and display them on the text area
        ex_names.append(example)  # Add each file name to the list

        # Parse the contents of the selected file according to their extension txt docx or pdf
        if example.split(".")[-1] == "docx":  # docx files
            contents_ex += docx2txt.process(os.path.join("examples", example)) + "\n"

        elif example.split(".")[-1] == "pdf":  # pdf files
            contents_ex += read_pdf(open(os.path.join("examples", example),'rb')) + "\n"

        else:  # txt files
            for line in examples[example].read().decode().splitlines():
                contents_ex += line + "\n"

        # Add the compiled contents of the file into the 'essays' list before going to the next uploaded file
        ex_essays.append(contents_ex)
        # ta_val will be the preview of all the essays in the text area; display index numbering if there are more than one file
        ta_val_ex += f"[{selected_example.index(example)}]\n" + contents_ex + "\n" if len(selected_example) > 1 else contents_ex

    # widgets and button to run on examples
    response_ta_ex = st.text_area("Essay/s:",placeholder="Your selected example essay/s will display here.",value=ta_val_ex,key='response_ta_ex',height=500, disabled=True)
    col1_ex, col2_ex, col3_ex = st.columns(3)
    min_score_ex = col1_ex.number_input('Minimum Score',0,100,0,key='min_score_ex')
    max_score_ex = col2_ex.number_input('Maximum Score',0,100,10,key='max_score_ex')
    run_button_ex = st.button("Grade Example Essay/s")
    enable_chatgpt_ex = st.checkbox("Evaluate example with ChatGPT?", help="Works best on one essay at a time.")

    # button is clicked
    if run_button_ex:
        if not response_ta_ex: # if any text area is empty:
            st.error("Please input the essay in their corresponding text area.")
        if min_score_ex >= max_score_ex:
            st.error("Minimum score must be less than maximum score.")
        else: # run model
            # 'evals' is a list of dataframes [DataFrame]
            # 'scores' is a list of the grades [int]
            evals_ex, scores_ex = run_model_on_list(ex_essays, min_score_ex, max_score_ex)

            # Display the final grade for each uploaded file
            grades_df_ex = pd.DataFrame({'Filename': ex_names, 'Final Grade': scores_ex})
            st.write("Grading done!")
            st.dataframe(grades_df_ex)

            st.write("Criteria are graded within the range of 1-4. \nMerging grades with evaluations..")

            # Add a column 'Filename' to each set of evaluation, and set the value to the corresponding file name
            for f in ex_names:
                evals_ex[ex_names.index(f)]['Filename'] = f

            # Combine the list of evaluation dataframes 'evals' into one single dataframe 'evals_df'
            evals_df_ex = pd.concat([df for df in evals_ex])

            # Combine the Grades with the Evaluations, then show it
            final_df_ex = grades_df_ex.merge(evals_df_ex, on='Filename')
            st.dataframe(final_df_ex)

            # ChatGPT Evaluation Section
            if enable_chatgpt_ex:
                run_chatgpt(ex_essays, min_score_ex, max_score_ex)

if __name__ == '__main__':
    main()

# To run streamlit, go to terminal and type: 'streamlit run app-source.py'
