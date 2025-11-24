from typing import Dict, Any, Optional

def display_plan_summary(result: Dict[str, Any]) -> None:
    """
    Parses the agent's results and prints a structured, pretty-printed 
    summary to the console for human review.
    """
    # 1. Error Handling
    if result.get("error"):
        print(f"\n❌ Agent finished with an error: {result['error']}\n")
        return

    # 2. Structure Validation
    experiments = result.get("proposed_experiments")
    if not experiments or not isinstance(experiments, list):
        print("\n⚠️  The agent returned a result, but no experiments were found.")
        # Optional: Print raw if debugging needed
        # print(json.dumps(result, indent=2))
        return

    # 3. Header
    print("\n" + "="*80)
    print("✅ PROPOSED EXPERIMENTAL PLAN")
    print("="*80)

    # 4. Loop through Experiments
    for i, exp in enumerate(experiments, 1):
        
        # --- Name & Hypothesis ---
        print(f"\n🔬 EXPERIMENT {i}: {exp.get('experiment_name', 'Unnamed Experiment')}")
        print("-" * 80)
        print(f"\n> 🎯 Hypothesis:\n> {exp.get('hypothesis', 'N/A')}")

        # --- Experimental Steps (Numbered) ---
        print("\n--- 🧪 Experimental Steps ---")
        steps = exp.get('experimental_steps', [])
        if steps:
            for j, step in enumerate(steps, 1):
                print(f" {j}. {step}")
        else:
            print("  (No steps provided)")
        
        # --- Equipment ---
        print("\n--- 🛠️  Required Equipment ---")
        equipment = exp.get('required_equipment', [])
        if equipment:
            # Print as a clean comma-separated list if short, or bullets if long
            if len(equipment) > 5:
                for item in equipment: print(f"  * {item}")
            else:
                print(f"  {', '.join(equipment)}")
        else:
            print("  (No equipment specified)")

        # --- Outcome & Justification (Critical for Review) ---
        print("\n--- 📈 Expected Outcome ---")
        print(f"  {exp.get('expected_outcome', 'N/A')}")

        print("\n--- 💡 Justification ---")
        print(f"  {exp.get('justification', 'N/A')}")
        
        # --- Source Documents ---
        print("\n--- 📄 Source Documents ---")
        sources = exp.get('source_documents', [])
        if sources:
            for src in sources:
                print(f"  - {src}")
        else:
            print("  (No sources listed)")

        # --- Code Indicator (If generated) ---
        if "implementation_code" in exp:
            print("\n--- 💻 Implementation Code ---")
            print("  ✅ Python script generated (saved to file).")

    print("\n" + "="*80)


def get_user_feedback() -> Optional[str]:
    """
    Pauses execution to get user input via the CLI. 
    Returns None if the user just presses ENTER (indicating approval).
    """
    print("\n" + "-"*60)
    
    print("👤 HUMAN FEEDBACK STEP")
    print("-" * 60)
    print("Review the plan above.")
    print("• To APPROVE: Press [ENTER] directly.")
    print("• To REQUEST CHANGES: Type your feedback/instructions and press [ENTER].")
    
    feedback = input("\n> Instruction: ").strip()
    
    if not feedback:
        return None # User accepted the plan
        
    return feedback