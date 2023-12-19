from openai import OpenAI
import os
import streamlit as st


def dugacki_iz_kratkih(uploaded_text, entered_prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    uploaded_text = uploaded_text[0].page_content

    if uploaded_text is not None:
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
                    Be sure to always write in Serbian." + f"{entered_prompt}",
            "p_system_4": "You are a helpful assistant that creates a conclusion of the provided text.",
            "p_user_4": "Please create a conclusion of the above text."
        }
        

        def get_response(p_system, p_user_ext):
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": all_prompts[p_system]},
                    {"role": "user", "content": uploaded_text},
                    {"role": "user", "content": p_user_ext}
                ]
            )
            return response.choices[0].message.content.strip()


        response = get_response("p_system_1", all_prompts["p_user_1"])
        st.write(response)
        # ovaj double check je veoma moguce bespotreban, no sto reskirati
        response = get_response("p_system_2", all_prompts["p_user_2"]).split('\n')
        topics = [item for item in response if item != ""]  # just in case - triple check

        final_summary = ""
        i = 0
        imax = len(topics)

        pocetak_summary = "At the begining of the text write the date (dd.mm.yy), topics that vere discussed and participants."

        for topic in topics:
            if i == 1:
                summary = get_response("p_system_3", f"{(pocetak_summary + all_prompts['p_user_3']).format(topic=topic)}")
                i += 1
            else:
                summary = get_response("p_system_3", f"{all_prompts['p_user_3'].format(topic=topic)}")
                i += 1

            st.info(f"Summarizing topic: {topic} - {i}/{imax}")
            final_summary += f"{summary}\n\n"

        final_summary += f"{get_response('p_system_4', all_prompts['p_user_4'])}"
        st.write(final_summary)
        return final_summary
    
    else:
        return "Please upload a text file."