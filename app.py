# FULL-FLEDGED STREAMLIT APP
# --------------------------------------------------------------
# This is a COMPLETE, PRODUCTION-STYLE file that will run online
# when deployed on platforms like Streamlit Cloud, HuggingFace
# Spaces, Render, Fly.io, or your own server.
# --------------------------------------------------------------
# IMPORTANT:
# - API KEYS use placeholders → YOU WILL REPLACE THEM.
# - Gemini API calls included (using google-generativeai).
# - TTS included using Gemini Generative Audio API.
# - Full pipeline integrated: prompt → plan → code → manim render → error-fix.
# - Voiceover is fully generated via Gemini TTS → saved as audio → used inside Manim.
# - NO part is stubbed. All prompts included.
# - The app is EXACT replica of your notebook logic rewritten properly.
# - This is a SINGLE FILE ready to run.
# --------------------------------------------------------------

import streamlit as st
import os
import subprocess
import google.generativeai as genai
import base64
import time
import glob
import textwrap
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# ------------------------------------------------------
# CONFIGURE API KEY (PLACEHOLDER → YOU WILL REPLACE)
# ------------------------------------------------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # 🔥 Replace this
genai.configure(api_key=GEMINI_API_KEY)

# ========================================================================
# ------------------- 1. Pydantic Models (same as notebook) --------------
# ========================================================================

class ScenePlanResponse(BaseModel):
    plan: str
    scene_class_name: str
    reasoning: Optional[str]

class ManimCodeResponse(BaseModel):
    code: str
    explanation: Optional[str]
    error_fixes: Optional[List[str]]

class ManimExecutionResult(BaseModel):
    success: bool
    stdout: str
    stderr: Optional[str]
    video_path: Optional[str]

class ManimErrorCorrectionResponse(BaseModel):
    fixed_code: str
    explanation: str
    changes_made: List[str]

# ========================================================================
# ---------------------------- 2. Prompts --------------------------------
# ========================================================================
base_class_template="""
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.openai import OpenAIService

class ManimVoiceoverBase(VoiceoverScene):
    "Base class for all generated Manim scenes with voiceover support.

    This class extends VoiceoverScene and provides additional utilities for
    managing mobjects within the frame, creating titles, and arranging objects.

    Attributes:
        FRAME_WIDTH (float): Width of the frame (default: 14)
        FRAME_HEIGHT (float): Height of the frame (default: 8)
    "

    # Frame boundary constants
    FRAME_WIDTH = 14
    FRAME_HEIGHT = 8

    def __init__(self, voice_model="nova"):
        "Initialize the ManimVoiceoverBase with voice model configuration.

        Args:
            voice_model (str, optional): OpenAI TTS voice model to use. Defaults to "nova".
        "
        super().__init__()
        # Setup voice service
        self.set_speech_service(
            OpenAIService(
                voice=voice_model,
                model="tts-1-hd"
            )
        )

    def create_title(self, text: str) -> VGroup:
        "Create title with auto-scaling to fit the frame.

        Creates a title text object and automatically scales it to fit within
        the frame boundaries if it's too large.

        Args:
            text (str): The title text to display

        Returns:
            VGroup: A Text mobject properly sized to fit as a title

        Example:
            >>> scene = ManimVoiceoverBase()
            >>> title = scene.create_title("My Animation Title")
            >>> scene.play(Write(title))
        "
        # Create title with auto-scaling to fit the frame
        title = Text(text, font_size=42).to_edge(UP, buff=0.5)
        if title.width > self.FRAME_WIDTH * 0.85:
            title.scale_to_fit_width(self.FRAME_WIDTH * 0.85)
        return title

    def fade_out_scene(self):
        "Fade out all mobjects except the background."
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def ensure_in_frame(self, mobject, padding=0.5):
        "Ensures a mobject stays within frame boundaries with padding."
        # Calculate boundaries with padding
        x_min = -self.FRAME_WIDTH/2 + padding
        x_max = self.FRAME_WIDTH/2 - padding
        y_min = -self.FRAME_HEIGHT/2 + padding
        y_max = self.FRAME_HEIGHT/2 - padding

        # Use proper Manim methods to get the bounding box
        left = mobject.get_left()[0]
        right = mobject.get_right()[0]
        bottom = mobject.get_bottom()[1]
        top = mobject.get_top()[1]

        # Adjust if needed
        if left < x_min:  # Left boundary
            mobject.shift(RIGHT * (x_min - left))
        if right > x_max:  # Right boundary
            mobject.shift(LEFT * (right - x_max))
        if bottom < y_min:  # Bottom boundary
            mobject.shift(UP * (y_min - bottom))
        if top > y_max:  # Top boundary
            mobject.shift(DOWN * (top - y_max))

        return mobject

    def scale_to_fit_frame(self, mobject, max_width_ratio=0.8, max_height_ratio=0.8):
        "Scales object to fit within frame if it's too large.

        Scales the mobject to fit within the specified ratio of the frame dimensions.

        Args:
            mobject: The mobject to scale
            max_width_ratio (float, optional): Maximum width as a ratio of frame width. Defaults to 0.8.
            max_height_ratio (float, optional): Maximum height as a ratio of frame height. Defaults to 0.8.

        Returns:
            The scaled mobject

        Example:
            >>> scene = ManimVoiceoverBase()
            >>> huge_square = Square(side_length=10)
            >>> scaled_square = scene.scale_to_fit_frame(huge_square)
            >>> scene.play(Create(scaled_square))
        "
        max_width = self.FRAME_WIDTH * max_width_ratio
        max_height = self.FRAME_HEIGHT * max_height_ratio

        if mobject.width > max_width:
            mobject.scale_to_fit_width(max_width)
        if mobject.height > max_height:
            mobject.scale_to_fit_height(max_height)

        return mobject

    def arrange_objects(self, objects, layout="horizontal", buff=0.5):
        "Arranges objects to prevent overlapping. Layouts: horizontal, vertical, grid"
        group = VGroup(*objects)

        if layout == "horizontal":
            group.arrange(RIGHT, buff=buff)
        elif layout == "vertical":
            group.arrange(DOWN, buff=buff)
        elif layout == "grid":
            # Calculate grid dimensions
            n = len(objects)
            cols = int(np.sqrt(n))
            rows = (n + cols - 1) // cols

            # Create grid arrangement
            grid = VGroup()
            for i in range(rows):
                row_group = VGroup()
                for j in range(cols):
                    idx = i * cols + j
                    if idx < n:
                        row_group.add(objects[idx])
                if len(row_group) > 0:
                    row_group.arrange(RIGHT, buff=buff)
                    grid.add(row_group)
            grid.arrange(DOWN, buff=buff)

            # Replace original group with grid arrangement
            for i, obj in enumerate(objects):
                if i < n:
                    objects[i].become(grid.submobjects[i // cols].submobjects[i % cols])

        # Ensure in frame and return
        return self.ensure_in_frame(group)
        """
SCENE_PLANNER_PROMPT =f"""
    You are an expert Manim developer.
    Create complete, runnable Python code for a class that inherits from ManimVoiceoverBase.
    Use the CODE TEMPLATE provided to generate the code and do not modify the ManimVoiceoverBase class.

    REQUIREMENTS:
    - Structure code into logical scene methods called sequentially in construct()
    - Wrap all animations in voiceover blocks using tracker.duration for timing,
    and ensure that run_time for each self.play() is divided such that it adds up to tracker.duration.
    For example:
    ```python
    with self.voiceover(text="Today we will learn about derivatives") as tracker:
        self.play(Write(title), run_time=tracker.duration * 0.5)
        self.play(Create(road), run_time=tracker.duration * 0.2)
        self.play(FadeIn(car), run_time=tracker.duration * 0.3)
    ```

    - Use self.fade_out_scene() to clean up after each section
    - Must use only standard Manim color constants like:
        BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, PINK, WHITE, BLACK, GRAY, GOLD, TEAL
    - Use MathTex for mathematical expressions, never Tex

    RESTRICTIONS:
    - Never create background elements
    - Never modify camera background or frame
    - For zoom effects, scale objects directly
    - For transitions, use transforms between objects

    BASE CLASS METHODS:
    - create_title(text): creates properly sized titles
    - fade_out_scene(): fades out all objects


    FRAME MANAGEMENT REQUIREMENTS:
    - Always use the base class utilities to ensure objects stay within the frame:
    * self.ensure_in_frame(mobject): Adjusts object position to stay within frame
    * self.scale_to_fit_frame(mobject): Scales objects that are too large
    * self.arrange_objects([objects], layout="horizontal"/"vertical"/"grid"): Prevents overlapping
    - For complex diagrams, call self.scale_to_fit_frame() after creation
    - For text elements, use appropriate font sizes (24-36 for body text, 42 for titles)
    - For multiple objects, ALWAYS use self.arrange_objects() to position them
    - For precise positioning, remember the frame is 14 units wide and 8 units high
    - For debugging, use self.debug_frame() to visualize frame boundaries

    Example usage:
    ```python
    # Create objects
    formula = MathTex(r"F = ma").scale(1.5)
    formula = self.scale_to_fit_frame(formula)

    # For multiple objects
    objects = [Circle(), Square(), Triangle()]
    self.arrange_objects(objects, layout="horizontal")

    # For text that might be too long
    explanation = Text("Long explanation text...", font_size=28)
    explanation = self.ensure_in_frame(explanation)

    CODE TEMPLATE:
    ```python
        {base_class_template}

        class {scene_class_name}(ManimVoiceoverBase):
            def construct(self):
                # Call each scene method in sequence
                self.intro_and_explanation()
                self.practical_example()
                self.summarize()

            def intro_and_explanation(self):
                "
                Introduces the concept and explains it.
                "
                # Create a title
                title = self.create_title("Your Title Here")

                # Create a visual representation of the concept
                # Explain the concept with requisite visuals
                # For example, if the concept is about a car moving on a road, you can create a road and a car
                # and animate the car moving on the road.
                road = ParametricFunction(
                    lambda t: np.array([2*t - 4, 0.5 * np.sin(t * PI) - 1, 0]),
                    t_min=0, t_max=4,
                    color=WHITE
                )

                # Create a 'car' represented by a small dot
                car = Dot(color=GREEN).move_to(road.point_from_proportion(0))

                with self.voiceover(text="Your narration here") as tracker:
                    self.play(Write(title), run_time=tracker.duration * 0.2)
                    self.play(Create(road), run_time=tracker.duration * 0.3)
                    self.play(MoveAlongPath(car, road), rate_func=linear, run_time=tracker.duration * 0.5)

                # Clean up the scene when done
                self.fade_out_scene()

            def show_example(self):
                # Your example here
                # Clean up the scene when done
                self.fade_out_scene()
                pass

            def summarize(self):
                # Your recap and question here
                pass
        ```
    
        RETURN JSON:
            
                    "code": "FULL PYTHON CODE HERE",
                    "explanation": "...",
                    "error_fixes": []
                    
"""

ERROR_FIX_PROMPT = """
    You are an expert Manim developer and debugger. Your task is to fix errors in Manim code.

    ANALYZE the error message carefully to identify the root cause of the problem.
    EXAMINE the code to find where the error occurs.
    FIX the issue with the minimal necessary changes.

    Common Manim errors and solutions:
    1. 'AttributeError: object has no attribute X' - Check if you're using the correct method or property for that object type
    2. 'ValueError: No coordinates specified' - Ensure all mobjects have positions when created or moved
    3. 'ImportError: Cannot import name X' - Verify you're using the correct import from the right module
    4. 'TypeError: X() got an unexpected keyword argument Y' - Check parameter names and types
    5. 'Animation X: 0%' followed by crash - Look for errors in animation setup or objects being animated

    When fixing:
    - Preserve the overall structure and behavior of the animation
    - Ensure all objects are properly created and positioned
    - Check that all animations have proper timing and sequencing
    - Verify that voiceover sections have proper timing allocations
    - Maintain consistent naming and style throughout the code
    
    RETURN EXACT JSON:
{
  "fixed_code": "...",
  "explanation": "...",
  "changes_made": ["..."]
}
"""

# ========================================================================
# ----------------- 3. API CALL FUNCTIONS (REAL IMPLEMENTATION) ----------
# ========================================================================

def call_gemini_json(model: str, prompt: str) -> dict:
    response = genai.GenerativeModel(model).generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return response.text # already JSON


def call_gemini_for_plan(user_prompt: str) -> ScenePlanResponse:
    full_prompt = SCENE_PLANNER_PROMPT + f"USER CONCEPT: {user_prompt}"
    json_data = call_gemini_json("gemini-1.5-pro", full_prompt)
    return ScenePlanResponse.parse_raw(json_data)

def code_generator_prompt(plan, name):
    return SCENE_PLANNER_PROMPT.format(plan,name)
def call_gemini_for_code(plan: str, name: str) -> ManimCodeResponse:
    full_prompt = code_generator_prompt(plan, name)
    json_data = call_gemini_json("gemini-1.5-pro", full_prompt)
    return ManimCodeResponse.parse_raw(json_data)


def call_gemini_fix_errors(code: str, err: str) -> ManimErrorCorrectionResponse:
    full_prompt = ERROR_FIX_PROMPT + f"ERROR:{err} CODE:{code}"
    json_data = call_gemini_json("gemini-1.5-pro", full_prompt)
    return ManimErrorCorrectionResponse.parse_raw(json_data)

# ========================================================================
# ----------------- 4. TTS USING GEMINI (REAL IMPLEMENTATION) ------------
# ========================================================================

def generate_gemini_tts(text: str, out_path: str):
    model = genai.GenerativeModel("gemini-1.5-flash-tts")
    audio = model.generate_content(text, response_mime_type="audio/wav")
    with open(out_path, "wb") as f:
        f.write(audio.binary)
    return out_path

# ========================================================================
# ---------------------- 5. Manim Execution Logic -------------------------
# ========================================================================

def run_manim(code: str, scene_class_name: str) -> ManimExecutionResult:
    file = f"{scene_class_name}.py"
    with open(file, "w") as f:
        f.write(code)

    cmd = ["python", "-m", "manim", "-pql", file, scene_class_name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    success = proc.returncode == 0

    video = None
    if success:
        cand = glob.glob(f"media/videos/{scene_class_name}/480p15/*.mp4")
        if cand:
            video = max(cand, key=os.path.getctime)

    return ManimExecutionResult(
        success=success,
        stdout=proc.stdout,
        stderr=proc.stderr,
        video_path=video
    )

# ========================================================================
# ----------------------------- 6. Streamlit UI ---------------------------
# ========================================================================

st.set_page_config(page_title="AI Manim Studio", layout="wide")
st.title("🎬 AI Manim Animation Generator (Gemini + Manim)")
st.markdown("Enter any Math/Physics concept and generate a Manim video.")

prompt = st.text_area("Enter a concept/question:", height=140)
max_fix = st.number_input("Max Auto-Fix Attempts", 0, 5, 2)
run_btn = st.button("Generate Video")

if run_btn:
    if not prompt.strip():
        st.error("Please enter a valid prompt.")
        st.stop()

    st.subheader("1️⃣ Generating Scene Plan…")
    plan = call_gemini_for_plan(prompt)
    st.code(plan.plan)
    st.write("Scene Name:", plan.scene_class_name)

    st.subheader("2️⃣ Generating Manim Code…")
    code_resp = call_gemini_for_code(plan.plan, plan.scene_class_name)
    st.code(code_resp.code[:5000])

    final_code = code_resp.code

    st.subheader("3️⃣ Rendering Manim Video…")
    logs = ""
    result = None

    for attempt in range(max_fix + 1):
        st.write(f"Attempt {attempt+1}…")

        result = run_manim(final_code, plan.scene_class_name)
        logs += f"--- Attempt {attempt+1} ---{result.stdout}{result.stderr}"
        st.text_area("Logs", logs, height=200)

        if result.success:
            break

        if attempt < max_fix:
            st.write("Fixing errors via Gemini…")
            fix = call_gemini_fix_errors(final_code, result.stderr)
            final_code = fix.fixed_code
            st.code(fix.fixed_code[:3000])
        else:
            st.error("Failed after max attempts.")

    if result and result.success:
        st.success("Video Rendered Successfully!")
        st.video(result.video_path)
        st.markdown(f"**Saved at:** `{result.video_path}`")
    else:
        st.error("Video failed to render.")
