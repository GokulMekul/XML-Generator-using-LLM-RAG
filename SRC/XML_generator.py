import yaml
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -----------------------------
# 2. LLM SYSTEM PROMPT
# -----------------------------
SURVEY_SUMMARY_SYSTEM_PROMPT = """
You are a survey-question interpreter.

Your task:
Given any user-written survey question (text extracted from OCR or typed by the user), produce a clean, structured summary.

You must extract and return ONLY the following fields in a YAML-like structure:

QuestionNumber: <number or null>
QuestionText: "<text>"

Comment: "<text or null>" # after the question text comment start with "(" and end with ")" and keep. at end of the sentence

QuestionType: <single-select | multi-select | grid | text | numeric | rating | drop-down | ranking | other>

Rows:
  - text: "<option text>"
    row_label: "<r1 | r2 | r3 ...>"   # The nearest label in text; prefix with 'r'
    anchor: true/false
    exclusive: true/false
    other-specify: true/false

Columns:
  - text: "<option text>"
    column_label: "<c1 | c2 | c3 ...>"   # The nearest label in text; prefix with 'c'
    anchor: true/false
    exclusive: true/false
    other-specify: true/false
  # If the question is not a grid, return: Columns: none

Choice:
  # For ranking questions:
  # Identify how many ranks are required.
  # Example:
  #   - text: "Rank1"
  #     choice_label: "ch1"
  #   - text: "Rank2"
  #     choice_label: "ch2"
  # If not a ranking question, return: Choice: none

AdditionalInstructions:
  shuffle: true/false
  randomization: true/false
  display_logic: true/false        # If question contains display/skip logic
  termination: true/false          # If question contains termination/end instruction
  note: "Original text contained bold or underlined emphasis."

If the question mentions screening / must-select / must-be logic such as:
- "MUST BE S5/1, IF NOT TERMINATE"
- “Only continue if respondent selects 1”
- "Terminate if they do not choose option 2"
- “Only rows 1 and 3 qualify”

Then:
1. Set "termination": true
2. Identify the exact row(s) required.
3. Output termination_condition = "Q{question-number}.r{row-number}"
4. termination logical and, or, not condition based on the python only

Examples:
- “If respondant select the S5/1 or if respondant selects the option (something they not provide row name example terminate if respondant selects ad) ”
  → termination_condition = "S5.r1"

- "Terminate if respondant selects S5/1-3, S5/1,2,3"
 -> termination _condition = "S5.r1 or S5.r2 or S5.r3"

- "Terminate if NOT selecting S5/r2"
  → termination_condition = "not(S5.r2)"

- "Terminate if respondant selects row 2 and row 3"
  → termination_condition = "S5.r2 and S5.r3"

- "Only continue if choosing 1 or 3"
  → termination_condition = "not(S5.r1 or S5.r3)"



RULES:
- avoid print the instruction in the code which contain inside the [] square bracket example [MULTIPLE SELECT; RANDOMIZE]
- avoid spelling mistake
- please more focus on the / and . in text 
- Do NOT generate any survey programming code.
- Do NOT generate XML, HTML, or tags such as <radio>.
- Output ONLY the above YAML-like structure.
- If any field is missing, still include it with null/none.
- The result must be plain text, not JSON.

"""
def summarize_question(final_input):
    response = client.chat.completions.create(
    model="openai/gpt-oss-120b",   # recommended stable Groq model
    messages=[
        {"role": "system", "content": SURVEY_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": final_input}
    ])

    summary_text = response.choices[0].message.content
    print(summary_text)    
    return summary_text

def parse_summary(summary_text):
    documents = list(yaml.safe_load_all(summary_text))
    return documents[0] if documents else None   

def generate_survey_code(summary):
    qtype = summary.get("QuestionType")
    qnum = summary.get("QuestionNumber")
    qtext = summary.get("QuestionText")
    comment = summary.get("Comment")
    rows = summary.get("Rows", [])

        # ---------- SAFE ROWS ----------
    rows = summary.get("Rows")
    if not isinstance(rows, list):
        rows = []
    # ---------- SAFE COLUMNS ----------
    columns = summary.get("Columns")
    if not isinstance(columns, list):
        columns = []

    # ---------- SAFE CHOICES ----------
    choices = summary.get("Choice")
    if not isinstance(choices, list):
        choices = []
    add = summary.get("AdditionalInstructions", {})
    shuffle = add.get("shuffle", False)
    termination = add.get("termination", False)
    randomization = add.get("randomization", False)
    display_logic = add.get("display_logic", False)
    termination = add.get("termination", False)
    term_cond = summary.get("termination_condition")
    display_cond = summary.get("DisplayLogic")
   

    print("Display cond",display_cond)

     
    code =""

    if qtype == "single-select":
        radio_tag = f'<radio label="{qnum}"'

        if display_logic or display_cond:          
            radio_tag += f'cond="{display_cond}"'
        if shuffle or randomization:
            radio_tag += ' shuffle="rows"'
        radio_tag += ">"
        code += radio_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'

            code += row_tag + "\n"
        code += "</radio>\n"
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'        
        return code  

    if qtype == "multi-select":
        checkbox_tag = f'<checkbox label="{qnum}" atleast="1"'
        if shuffle or randomization:
            checkbox_tag += ' shuffle="rows"'
        if display_logic and display_cond:
            checkbox_tag += f' cond="{display_cond}"' 
        checkbox_tag += ">"
        code += checkbox_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'

            code += row_tag + "\n"
        code += "</checkbox>\n"
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code  

    if qtype == "text":
        text_tag = f'<textarea label="{qnum}" optional="0"'
        if display_logic or display_cond:
            text_tag += f' cond="{display_cond}"' 
        text_tag += ">"
        code += text_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        code+=f'<validate>CheckBlank({qnum},1)</validate>' 
        code += "</textarea>\n"         
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code         
    
    
    if qtype == "single-select-grid":
        single_radio_tag = f'<radio label="{qnum}"'

        if display_logic and display_cond:
             single_radio_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            single_radio_tag += ' shuffle="rows"'

        single_radio_tag += ">"
        code += single_radio_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
             code += f'  <comment><i>{comment}</i></comment>\n'

    # --- Rows ---
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"

    # --- Columns ---
    
        for col in columns:
            col_label = col.get("column_label", "cX")
            text = col.get("text", "")
            anchor = col.get("anchor", False)
            exclusive = col.get("exclusive", False)
            other_spec = col.get("other-specify", False)
            col_tag = f'  <col label="{col_label}"'
            if anchor:
                col_tag += ' randomization="0"'
            if other_spec:
                col_tag += ' open="1" openSize="25"'
            if exclusive:
                col_tag += ' exclusive="1"'
            col_tag += f'>{text}</col>'
        
        code += col_tag + "\n"

        code += "</radio>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code


    if qtype == "multi-select-grid":
        multi_check_tag = f'<checkbox label="{qnum}"'

        if display_logic and display_cond:
            multi_check_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            multi_check_tag += ' shuffle="rows"'

        multi_check_tag += ">"
        code += multi_check_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'

    # --- Rows ---
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"

    # --- Columns ---
    
        for col in columns:
            col_label = col.get("column_label", "cX")
            text = col.get("text", "")
            anchor = col.get("anchor", False)
            exclusive = col.get("exclusive", False)
            other_spec = col.get("other-specify", False)
            col_tag = f'  <col label="{col_label}"'
            if anchor:
                col_tag += ' randomization="0"'
            if other_spec:
                col_tag += ' open="1" openSize="25"'
            if exclusive:
                col_tag += ' exclusive="1"'
            col_tag += f'>{text}</col>'
            code += col_tag + "\n"

        code += "</checkbox>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code
    

    if qtype == 'information':
        info_tag = f'<html label="{qnum}"'
        if display_logic and display_cond:
            info_tag += f' cond="{display_cond}"'
        info_tag += ">"
        code += info_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        code += "</html>\n"
        code += "<suspend/>\n"    
        return code        


    

    if qtype == 'ranking':
        select_tag = f'<select uses="ranksort.4" label="{qnum}"'
        if display_logic and display_cond:
            select_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            select_tag += ' shuffle="rows"'
        select_tag += ">"
        code += select_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'

    # --- Rows ---
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"

    # --- Choices ---
        
        for choice in choices:
            cho_label = choice.get("choice_label", "chX")
            text = choice.get("text", "")
            anchor = choice.get("anchor", False)
            exclusive = choice.get("exclusive", False)
            other_spec = choice.get("other-specify", False)
            cho_tag = f'  <choice label="{cho_label}"'
            if anchor:
                cho_tag += ' randomization="0"'
            if exclusive:
                cho_tag += ' exclusive="1"'
            cho_tag += f'>{text}</choice>'
            code += cho_tag + "\n"

        code += "</select>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code

    if qtype == "numeric":
        number_tag = f'<number size="2" verify="range(0,100)" label="{qnum}"'
        if display_logic and display_cond:
            number_tag += f' cond="{display_cond}"'
        number_tag += ">"
        code += number_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        code += "</number>\n"
        code += "<suspend/>\n"   
        if termination or term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code  

    if qtype == 'drop-down':
        select_tag = f'<select label="{qnum}"'
        if display_logic and display_cond:
            select_tag += f' cond="{display_cond}"'
        if shuffle or randomization:
            select_tag += ' shuffle="rows"'
        select_tag += ">"
        code += select_tag + "\n"
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'

          
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'
            code += row_tag + "\n"
    # --- Choices ---
        
        for choice in choices:
            cho_label = choice.get("choice_label", "chX")
            text = choice.get("text", "")
            anchor = choice.get("anchor", False)
            exclusive = choice.get("exclusive", False)
            other_spec = choice.get("other-specify", False)
            cho_tag = f'  <choice label="{cho_label}"'
            if anchor:
                cho_tag += ' randomization="0"'
            if exclusive:
                cho_tag += ' exclusive="1"'
            cho_tag += f'>{text}</choice>'
            code += cho_tag + "\n"

        code += "</select>\n"
        code += "<suspend/>\n"
        if termination and term_cond:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'
        return code      

    if qtype == "autosum":
        num_tag = f'<number amount="100" size="2" autosum:prefill="0" label="{qnum}"'

        if display_logic and display_cond:          
            num_tag += f'cond="{display_cond}"'
        if shuffle or randomization:
            num_tag += ' shuffle="rows"'
        num_tag += ">"
        code += num_tag + "\n"  
        code += f'  <title>{qtext}</title>\n'
        if comment:
            code += f'  <comment><i>{comment}</i></comment>\n'
        for row in rows:
            row_label = row.get("row_label", "rX")
            text = row.get("text", "")
            anchor = row.get("anchor", False)
            exclusive = row.get("exclusive", False)
            row_logic = row.get("row-display")
            other_spec = row.get("other-specify", False)
            row_tag = f'  <row label="{row_label}"'
            if row_logic:
                row_tag += f' cond="{row_logic}"'
            if anchor:
                row_tag += ' randomization="0"'
            if other_spec:
                row_tag += ' open="1" openSize="25"'
            if exclusive:
                row_tag += ' exclusive="1"'
            row_tag += f'>{text}</row>'

            code += row_tag + "\n"
        code += "</number>\n"
        code += "<suspend/>\n"
        if termination:
            code += f'<term cond="{term_cond}">Term at {qnum}</term>\n'        
        return code  

    if qtype=="other":
        code = "We will update it"
    return code 



def generate_xml(user_input: str) -> str:
    summary_text = summarize_question(user_input)
    summary = parse_summary(summary_text)
    if not summary:
        return "Error: Could not parse summary."
    return generate_survey_code(summary)



    



