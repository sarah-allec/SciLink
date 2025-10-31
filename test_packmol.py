import os
import logging
from scilink.agents.sim_agents.packmol_agent import PackmolGeneratorAgent


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyArXF8SVnInKM7RK9zvGuIjW3j1FiqRaNo")
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
OUTPUT_DIR = "packmol_test_output"

#user_request = "Create a 1.0 M ZnOTf solution in H2O with ethyl isopropyl sulfone cosolvent in a 40x40x40 Angstrom box."    
#user_request = "Create 2,3DHP with 6/7LiOH supporting electrolyte at concentration 0.1 M in a 40x40x40 Angstrom box"
user_request = "Create 0.15 M NaCl in water in a 40x40x40 Angstrom box"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """
    Initializes and runs the PackmolGeneratorAgent with a test prompt.
    """
    print("🚀 Initializing PackmolGeneratorAgent...")

    try:
        agent = PackmolGeneratorAgent(
            api_key=GOOGLE_API_KEY,
            model_name=MODEL_NAME,
            working_dir=OUTPUT_DIR
        )
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("Please ensure the 'packmol' executable is in your system's PATH.")
        return

    
    print(f"\n📝 Sending request to agent:\n   '{user_request}'")
    print("-" * 50)
    
    result = agent.generate_structure(description=user_request)

    print("-" * 50)
    if result.get("status") == "success":
        print("✅ Success!")
        print(f"   Generated Structure File: {result.get('output_file')}")
    else:
        print("❌ Failure!")
        print(f"   Error Message: {result.get('message')}")

if __name__ == "__main__":
    main()