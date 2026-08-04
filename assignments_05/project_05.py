# --- Task 1: Setup and System Prompt ---  
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are an AI job application coach who helps job seekers create stronger job application materials.

Your role is to help users:
- Rewrite resume bullet points to highlight transferable skills.
- Draft and improve cover letters.
- Ask thoughtful follow-up questions when more information is needed.
- Help users present their skills and experience clearly for job applications.

Behavior guidelines:
- Stay focused on job application materials and career transition support.
- Always remind users to review, edit, and personalize your suggestions before submitting them anywhere.
- Explain that you may not know the exact expectations or norms of the user's specific industry, and encourage users to apply their own judgment and verify requirements.
- Provide helpful suggestions while avoiding making decisions for the user.
"""

# Deliberate choice: I included detailed behavior guidelines because a specific system prompt
# makes the assistant's responses more consistent and focused on job application tasks.

result = get_completion([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "How long should my resume be?"}
])

print("Task 1 Output:")
print(result)

print("\n" + "="*50 + "\n")

# --- Task 2: Bullet Point Rewriter ---  

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result


    response = get_completion(messages, temperature=0)

    # Parse JSON safely
    try:
        clean_response = response.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_response)
        for item in result:
            print("Original:", item["original"])
            print("Improved:", item["improved"])
            print("-" * 40)
        return result
    
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        print(response)
        return []


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewrite_bullets(bullets)
print("\n" + "="*50 + "\n")

# These bullets are weak because they are too generic, lack measurable details,
# and do not clearly show impact. The model improved them by using stronger
# action verbs and making the descriptions more professional and results focused.
# During testing, json.loads() succeeded without errors.


# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result

    result = get_completion(messages, temperature=0)

    return result.strip()


job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."


cover_letter = generate_cover_letter(job_title, background)
print("Task 3 Output:")
print(cover_letter)

# I chose these examples because they demonstrate the confident, specific,
# and career-change-focused style I want the model to follow.
# The few-shot pattern helps control the tone, structure, and level of detail
# of the output while reducing generic cover letter openings.
# Few-shot examples provide a clear template for the model to follow, which
# helps improve relevance and quality for complex tasks like this.


# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("I am unable to process that request. Please rephrase your message.")
        print("Triggered categories:", result.results[0].categories)
        return False
    return True

print("Safe input: ", is_safe("I want to take a trip to Singapore."))
print("Unsafe input: ", is_safe("I want to slap myself."))

# --- Task 5: The Chatbot Loop ---

def run_chatbot():
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue 

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            rewrite_bullets(raw_bullets)

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            cover_letter = generate_cover_letter(job_title, background)
            print("\nJob Application Helper: Here's a draft opening paragraph for your cover letter. Customize it to fit your needs:\n")
            print(cover_letter)

        else:
            messages.append({"role": "user", "content": user_input})
            response = get_completion(messages)
            print("Job Application Helper:", response)
            messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    run_chatbot()

# --- Task 6: Ethics Reflection ---

# Option A — Comment block

"""
1)
A chatbot can give biased advice because it learns from text created by many different people, and that 
information may include existing biases. It may favor certain communication styles, industries, or cultural 
backgrounds because those examples appear more often in its training data. For example, it might suggest 
resume styles or job advice that works well in one industry or country but may not fit everyone.

2)
If a job seeker submits the bot's output without reviewing it, the application may include incorrect or 
exaggerated information. The bot could add skills, experience, or achievements that the person does not 
actually have. It may also contain mistakes or sound too generic, which could hurt the person's chances of 
getting hired. Reviewing and editing the output helps make sure it is accurate and personal.

"""
