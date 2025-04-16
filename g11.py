import gradio as gr
import google.generativeai as genai
import os
import sys
import traceback
import json
from pathlib import Path
from dotenv import load_dotenv
import time

# --- Configuration ---
ENV_FILE_PATH = r"C:\Users\Scott\Downloads\key.env" # Ensure this path is correct
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # The model to use for the chat
WORKSPACE_DIR = Path("./agent_workspace").resolve() # Resolve to absolute path

# --- Safety Check & Workspace Creation ---
try:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Workspace directory ensured at: {WORKSPACE_DIR}")
except Exception as e:
    print(f"FATAL ERROR: Could not create or access workspace directory '{WORKSPACE_DIR}': {e}")
    sys.exit(1)

# --- Load API Key ---
load_dotenv(dotenv_path=ENV_FILE_PATH)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key: sys.exit("API Key not found.")

# --- Configure Gemini API ---
try:
    print("Configuring Generative AI library...")
    genai.configure(api_key=api_key)
    print(f"Creating model instance for '{MODEL_NAME}'...")
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    print("Model instance created successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not configure Gemini or create model instance: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Agent Helper Functions ---

def execute_action(action: dict) -> tuple[bool, str]:
    """Executes a single file system action safely within the workspace."""
    action_type = action.get("type")
    path_str = action.get("path")
    content = action.get("content", "")

    if not action_type or not path_str:
        return False, "Error: Action type or path missing."

    try:
        target_path = WORKSPACE_DIR.joinpath(path_str).resolve()
        if WORKSPACE_DIR not in target_path.parents and target_path != WORKSPACE_DIR:
             if not (action_type == "create_folder" and target_path == WORKSPACE_DIR.joinpath(path_str)):
                 print(f"Attempted path traversal: {target_path}")
                 return False, f"Error: Path '{path_str}' is outside the allowed workspace."

        if action_type == "create_folder":
            target_path.mkdir(parents=True, exist_ok=True)
            return True, f"Folder created/ensured: {path_str}"
        elif action_type == "create_file" or action_type == "edit_file":
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f: f.write(content)
            verb = "Created" if action_type == "create_file" else "Edited"
            return True, f"File {verb}: {path_str}"
        else:
            return False, f"Error: Unknown action type '{action_type}'."
    except Exception as e:
        print(f"Error executing action {action}: {e}")
        traceback.print_exc()
        return False, f"Error performing '{action_type}' on '{path_str}': {e}"

def format_gemini_history(gradio_history: list[dict[str, str]]) -> list[dict]:
    """Converts Gradio 'messages' history to Gemini format."""
    gemini_history = []
    for item in gradio_history:
        role = item.get("role")
        content = item.get("content", "")
        gemini_role = "model" if role == "assistant" else "user"
        if content:
            gemini_history.append({'role': gemini_role, 'parts': [content]})
    return gemini_history

def get_workspace_structure(base_path: Path) -> str:
    """Generates a string representation of the workspace directory structure."""
    structure = []
    try:
        for item in sorted(base_path.rglob('*')):
            depth = len(item.relative_to(base_path).parts) -1
            indent = "  " * depth
            prefix = "📁" if item.is_dir() else "📄"
            structure.append(f"{indent}{prefix} {item.name}")
        if not structure: return f"(Workspace '{base_path.name}' is empty)"
        return f"Current Workspace Structure ('{base_path.name}'):\n" + "\n".join(structure)
    except Exception as e:
        print(f"Error getting workspace structure: {e}")
        return "(Error retrieving workspace structure)"

# --- Gradio Action Functions ---

def request_halt(controls: dict) -> dict:
    """Updates the state to request halting execution."""
    print("--- Halt Requested ---")
    controls['halt_requested'] = True
    return controls

# --- Main Chat/Agent Function ---
def agent_chat_response(
    message: str,
    history: list[dict[str, str]],
    controls: dict
):
    """
    Handles user messages, agent planning/answering, and auto-execution with halt.

    Yields tuples: (Updated_History, Updated_Controls_State, Halt_Button_Update)
    """
    print(f"\n--- Turn Start ---")
    print(f"User Message: {message}")
    print(f"Initial Controls State: {controls}")

    controls['halt_requested'] = False # Reset halt request

    bot_message = ""
    gemini_history = format_gemini_history(history)
    workspace_snapshot = get_workspace_structure(WORKSPACE_DIR)

    # Initial UI update: Disable halt button
    yield history, controls, gr.update(interactive=False)

    try:
        status_message = "Thinking..."
        history.append({"role": "assistant", "content": status_message})
        yield history, controls, gr.update(interactive=False)

        # === REVISED PROMPT (Simplified README Content in Plan) ===
        prompt = f"""You are an AI assistant acting as a software development agent.
Your goal is to fulfill the user's request, which might involve planning file system operations within a workspace ('{WORKSPACE_DIR.name}') OR answering questions.
Preference: For non-trivial software requests, structure the project logically using folders and multiple files.

Current Workspace Structure:
```
{workspace_snapshot}
```

User Request: "{message}"

Previous Conversation History:
{json.dumps(gemini_history, indent=2)}

Your Task:
1. Analyze the request, workspace, and history.
2. Determine the `response_type`: `"plan"` (for file operations) or `"informational"` (for questions/answers).
3. Generate the response based on the type:
    - **For "plan" type:**
        - Include a `languages_used` field (list of strings, e.g., `["Python", "HTML"]`).
        - Provide a `plan_steps` list of action objects (create_folder, create_file, edit_file).
        - **IMPORTANT README CONTENT:** If generating a `README.md` file as part of the plan, keep its `content` field **very brief** (e.g., just a title like `# Project Name`) within this JSON plan to ensure overall JSON validity. More detailed README content can be added later via a separate request if needed.
    - **For "informational" type:**
        - Provide a `message` string containing the answer.
4. CRITICAL OUTPUT FORMAT: Output ONLY a single JSON object containing the response type and corresponding data. Enclose it in ```json ... ``` markers. Ensure valid JSON with proper escaping (`\\n`, `\\"`, `\\\\`).

Example "plan" Output (with brief README):
```json
{{
  "response_type": "plan",
  "languages_used": ["Python", "Markdown"],
  "plan_steps": [
    {{ "type": "create_folder", "path": "my_script" }},
    {{ "type": "create_file", "path": "my_script/run.py", "language": "python", "content": "print(\\"Running...\\")" }},
    {{ "type": "create_file", "path": "my_script/README.md", "language": "markdown", "content": "# My Script Project" }}
  ]
}}
```

Example "informational" Output:
```json
{{
  "response_type": "informational",
  "message": "The 'my_script' folder contains 'run.py' and 'README.md'."
}}
```
"""
        # ==============================

        chat = model.start_chat(history=[])
        print("Sending prompt to Gemini...")
        response = chat.send_message(prompt)
        print("Received response from Gemini.")

        raw_response_text = response.text
        print(f"Raw Response from LLM:\n{raw_response_text}")

        parsed_response = None
        parsing_error = ""
        try:
            if "```json" in raw_response_text and "```" in raw_response_text.split("```json", 1)[1]:
                json_str = raw_response_text.split("```json", 1)[1].split("```", 1)[0].strip()
                print(f"Extracted JSON String for parsing:\n{json_str}")
                parsed_response = json.loads(json_str)
                if not isinstance(parsed_response, dict) or "response_type" not in parsed_response:
                    raise ValueError("Parsed JSON is not a dictionary or missing 'response_type'.")
                print("Response parsed successfully.")
            else:
                raise ValueError("Could not find valid ```json ... ``` markers.")
        except Exception as parse_err:
            print(f"ERROR parsing response: {parse_err}")
            traceback.print_exc()
            parsing_error = f"Error: Could not parse the response from the AI.\n(Raw response: {raw_response_text[:500]}...)"
            parsed_response = None

        # --- Process Parsed Response ---
        if parsed_response:
            response_type = parsed_response.get("response_type")

            if response_type == "plan":
                plan = parsed_response.get("plan_steps", [])
                declared_languages = parsed_response.get("languages_used", [])
                if not isinstance(declared_languages, list): declared_languages = []

                if isinstance(plan, list) and plan:
                    # --- Execute Plan ---
                    num_steps = len(plan)
                    lang_str = ", ".join(declared_languages) if declared_languages else "N/A"
                    bot_message = f"Plan generated with {num_steps} step(s).\n"
                    bot_message += f"**Language(s) to be used:** {lang_str}\n"
                    bot_message += "Starting execution...\n(Click 'Halt' to stop)"
                    history.append({"role": "assistant", "content": bot_message})
                    yield history, controls, gr.update(interactive=True) # Enable Halt

                    all_steps_succeeded = True
                    execution_halted = False
                    for i, action_to_execute in enumerate(plan):
                        if controls.get('halt_requested', False):
                            print("--- Execution Halted by User ---")
                            execution_halted = True
                            halt_message = f"\nExecution halted by user request before step {i + 1}."
                            history.append({"role": "assistant", "content": halt_message})
                            yield history, controls, gr.update(interactive=False) # Disable Halt
                            break

                        step_num = i + 1
                        step_message = f"\n--- Executing Step {step_num}/{num_steps} ---\n"
                        step_message += f"Action:\n```json\n{json.dumps(action_to_execute, indent=2)}\n```\n"
                        history.append({"role": "assistant", "content": step_message})
                        yield history, controls, gr.update(interactive=True)
                        time.sleep(0.5)

                        success, result_msg = execute_action(action_to_execute)
                        result_message = f"Result (Step {step_num}): {result_msg}"
                        history.append({"role": "assistant", "content": result_message})
                        yield history, controls, gr.update(interactive=True)
                        time.sleep(0.5)

                        if not success:
                            all_steps_succeeded = False
                            final_message = "\nExecution failed. Stopping plan."
                            history.append({"role": "assistant", "content": final_message})
                            yield history, controls, gr.update(interactive=False) # Disable Halt
                            break

                    if not execution_halted:
                        if all_steps_succeeded:
                            final_message = "\nPlan successfully completed!"
                            history.append({"role": "assistant", "content": final_message})
                        yield history, controls, gr.update(interactive=False) # Disable Halt

                else:
                     bot_message = "The AI generated an empty or invalid plan. Please try again."
                     history.append({"role": "assistant", "content": bot_message})
                     yield history, controls, gr.update(interactive=False)


            elif response_type == "informational":
                bot_message = parsed_response.get("message", "Sorry, I couldn't formulate an answer.")
                history.append({"role": "assistant", "content": bot_message})
                yield history, controls, gr.update(interactive=False)

            else:
                bot_message = f"Received an unknown response type ('{response_type}') from the AI."
                history.append({"role": "assistant", "content": bot_message})
                yield history, controls, gr.update(interactive=False)

        else:
            # Parsing failed
            bot_message = parsing_error or "Sorry, I couldn't generate a valid response."
            history.append({"role": "assistant", "content": bot_message})
            yield history, controls, gr.update(interactive=False)

    except Exception as e:
        print(f"ERROR in agent logic: {e}")
        traceback.print_exc()
        bot_message = f"An unexpected error occurred: {e}."
        history.append({"role": "assistant", "content": bot_message})
        yield history, controls, gr.update(interactive=False)

    print(f"Final Controls State: {controls}")
    print(f"--- Turn End ---")

    yield history, controls, gr.update(interactive=False)


# --- Gradio Interface Setup ---
with gr.Blocks(theme="soft") as demo:
    execution_controls = gr.State({'halt_requested': False})

    gr.Markdown(f"""
    # Gemini Agentic Chat (Interactive Multi-File) - {MODEL_NAME}
    Interact with the AI agent to create multi-file projects within `{WORKSPACE_DIR.name}` or ask questions.
    - Agent will state intended language(s) and prefer multi-file structures.
    - Agent will plan and **auto-execute** file operations.
    - Click **'Halt Current Execution'** to stop execution *before* the next step runs.
    - **<span style='color:red; font-weight:bold;'>WARNING:</span> Auto-execution is active. Use with caution!**
    """)

    chatbot = gr.Chatbot(
        label="Chat History",
        height=550,
        show_label=False,
        show_copy_button=True,
        type='messages'
     )

    with gr.Row():
        txt_message = gr.Textbox(
            label="Your Request or Question:",
            placeholder="Type request (e.g., 'create simple flask app') or question...",
            container=False,
            scale=7
        )
    with gr.Row():
         halt_button = gr.Button("Halt Current Execution", variant="stop", interactive=False)


    # --- Connect Components ---
    txt_message.submit(
        fn=agent_chat_response,
        inputs=[txt_message, chatbot, execution_controls],
        outputs=[chatbot, execution_controls, halt_button],
    )
    txt_message.submit(lambda: gr.update(value=""), None, txt_message, queue=False)

    halt_button.click(
        fn=request_halt,
        inputs=[execution_controls],
        outputs=[execution_controls]
    )


if __name__ == "__main__":
    print("Launching Gradio Interface... Access it in your browser (usually http://127.0.0.1:7860)")
    print(f"File operations will occur within: {WORKSPACE_DIR}")
    print("WARNING: Auto-confirmation is ENABLED. Agent will execute plans immediately.")
    demo.launch()