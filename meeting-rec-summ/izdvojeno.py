import streamlit as st
from openai import OpenAI
import os

def dugacki_iz_kratkih(uploaded_file, entered_prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



    if uploaded_file is not None:
        all_prompts = {
            "p_system_1": "You are a helpful assistant that identifies topics in a provided text.",
            "p_user_1": "Please provide a numerated list of topics described in the text - one topic per line. \
                Be sure not to write anything else.",
            "p_system_2": "You are a helpful assistant that corrects stuctural mistakes in a provided text. \
                You only check if the rules were followed, and then you correct the mistakes if there are any.",
            "p_user_2": f"Please check if the previous assistant generated a response that is inline with the following request: \
                'Please provide a numerated list of topics described in the text - one topic per line. Be sure not to write anything else.' \
                    If there are any mistakes, please correct them; e.g. if there is a short intro before the topic list or similar. \
                        If there are no mistakes, just send the text back.",
            "p_system_3": "You are a helpful assistant that summarizes only parts of the provided text that are related to the requested topic.",
            "p_user_3": "Please summarize the above text focusing only on the topic: {topic}. \
                Add a simple title (don't write hashes or similar) and 2 empty lines before and after the summary. \
                    Be sure to always write in Serbian." + f"{entered_prompt}"\
        }
        file_content = uploaded_file.read().decode(encoding="utf-8-sig")
        

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": all_prompts["p_system_1"]},
                {"role": "user", "content": file_content},
                {"role": "user", "content": all_prompts["p_user_1"]}
            ]
        )
        content = response.choices[0].message.content.strip()
        

        # ovaj double check je veoma moguce bespotreban, no sto reskirati
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": all_prompts["p_system_2"]},
                {"role": "user", "content": content},
                {"role": "user", "content": all_prompts["p_user_2"]}
            ]
        )
        content = response.choices[0].message.content.strip().split('\n')
        topics = [item for item in content if item != ""]  # just in case - triple check


        final_summary = ""
        i = 1
        imax = len(topics)
        for topic in topics:
            st.info(f"Summarizing topic: {topic} - {i}/{imax}")
            i += 1
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": all_prompts["p_system_3"]},
                    {"role": "user", "content": file_content},
                    {"role": "user", "content": f"{all_prompts['p_user_3'].format(topic=topic)}"}
                ]
            )
            summary = response.choices[0].message.content.strip()

            final_summary += f"{summary}\n\n"
        
        return {"content": final_summary}
