import asyncio
import json
from pathlib import Path
import sys
import re

# This line adds the main project folder to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# This specifically adds the 'backend' folder to the path as well
sys.path.append(str(project_root / "backend"))
# This adds the 'mcp' folder to the path
sys.path.append(str(project_root / "mcp"))

from backend.app.agents.math_agent import MathAgent
from backend.app.models.schemas import MathQuestionRequest, QuestionType

def extract_multiple_choice_answer(solution_text: str, steps_text: str = "") -> str:
    """
    Try to extract the multiple choice answer (A, B, C, D) from the solution.
    """
    # Combine all text to search
    full_text = f"{solution_text} {steps_text}".lower()
    
    # Pattern 1: Look for explicit answer statements
    patterns = [
        r'answer\s*:?\s*\(?([abcd])\)?',
        r'option\s*:?\s*\(?([abcd])\)?', 
        r'choice\s*:?\s*\(?([abcd])\)?',
        r'the\s+answer\s+is\s*:?\s*\(?([abcd])\)?',
        r'therefore\s*:?\s*\(?([abcd])\)?',
        r'\b([abcd])\s*is\s+correct',
        r'correct\s+answer\s*:?\s*\(?([abcd])\)?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text)
        if match:
            return match.group(1).upper()
    
    # Pattern 2: Look for isolated letters near end of text
    # Split by sentences and look in last few sentences
    sentences = full_text.split('.')
    for sentence in sentences[-3:]:  # Check last 3 sentences
        isolated_letter = re.search(r'\b([abcd])\b', sentence.strip())
        if isolated_letter:
            return isolated_letter.group(1).upper()
    
    return "UNKNOWN"

def is_answer_correct(generated_answer: str, ground_truth: str, question_text: str, full_solution = None) -> bool:
    """
    Enhanced answer checking with better debugging
    """
    gen_ans = generated_answer.strip().lower()
    truth = ground_truth.strip().lower()

    print(f"    Generated: '{generated_answer}'")
    print(f"    Expected: '{ground_truth}'")
    
    # Try to extract multiple choice answer if it's not clear
    if gen_ans not in ['a', 'b', 'c', 'd'] and full_solution:
        # Try to extract from the full solution
        steps_text = ""
        if hasattr(full_solution, 'steps') and full_solution.steps:
            steps_text = " ".join([step.description for step in full_solution.steps])
        
        extracted = extract_multiple_choice_answer(gen_ans, steps_text)
        print(f"    Extracted: '{extracted}'")
        
        if extracted != "UNKNOWN":
            gen_ans = extracted.lower()

    # Case 1: Direct match
    if gen_ans == truth:
        return True

    # Case 2: For multiple-choice, check if the ground truth letter is in the generated answer
    if len(truth) == 1 and truth.isalpha():
        if re.search(f'[^a-zA-Z0-9]{truth}[^a-zA-Z0-9]|^{truth}[^a-zA-Z0-9]|[^a-zA-Z0-9]{truth}$', f' {gen_ans} '):
            return True

    # Case 3: Try numerical answer matching (like before)
    try:
        options = re.findall(r'\(([A-D])\)\s*(.*?)(?=\s*\([A-D]\)|$)', question_text, re.DOTALL)
        option_dict = {opt[0].lower(): opt[1].strip().replace('\\', '') for opt in options}
        correct_option_text = option_dict.get(truth)

        if correct_option_text:
            correct_option_text = re.sub(r'[^a-zA-Z0-9.-]', '', correct_option_text.lower())
            gen_ans_cleaned = re.sub(r'[^a-zA-Z0-9.-]', '', gen_ans.lower())
            
            if gen_ans_cleaned in correct_option_text:
                return True
    except Exception:
        pass

    return False

async def run_benchmark():
    """
    Runs the Math Routing Agent against the JEEBench dataset and evaluates the results.
    """
    print("Starting JEEBench benchmark...")

    # Load the Dataset
    dataset_path = project_root / "evaluation" / "jee_bench" / "data" / "dataset.json"
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            math_questions = [q for q in all_data if q.get('subject') == 'math']
    except Exception as e:
        print(f"ERROR reading dataset: {e}")
        return

    print(f"Found {len(math_questions)} math questions to process.")

    # Initialize the Agent
    math_agent = MathAgent()
    results = []

    # Process Each Question (testing with first 3 for detailed debugging)
    for i, question_data in enumerate(math_questions[:3]): 
        question_text = question_data.get('question', '')
        ground_truth_answer = question_data.get('gold', '')

        if not question_text or not ground_truth_answer:
            continue

        print(f"\n--- Processing question {i+1}/3 ---")
        print(f"Question: {question_text[:100]}...")
        print(f"Expected Answer: {ground_truth_answer}")
        
        request = MathQuestionRequest(question=question_text, subject=QuestionType.OTHER)
        solution = await math_agent.solve_math_problem(request)
        
        # DEBUG: Print full solution details
        print(f"\n=== SOLUTION DEBUG ===")
        print(f"Final Answer: '{solution.final_answer}'")
        print(f"Confidence: {solution.confidence_score}")
        print(f"Source: {solution.source.value}")
        print(f"Steps Count: {len(solution.steps) if solution.steps else 0}")
        if solution.steps:
            print("Steps:")
            for j, step in enumerate(solution.steps[:3]):  # Show first 3 steps
                print(f"  {j+1}. {step.description[:100]}")
        print("======================\n")
        
        # Use enhanced checking function
        correct = is_answer_correct(solution.final_answer, ground_truth_answer, question_text, solution)

        results.append({
            "question": question_text[:200],  # Truncate for readability
            "ground_truth_answer": ground_truth_answer,
            "generated_answer": solution.final_answer,
            "is_correct": correct,
            "confidence": solution.confidence_score,
            "source": solution.source.value,
        })
        
        print(f"Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        print("-" * 60)

    # Save the Results
    results_dir = project_root / "evaluation" / "jee_bench" / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "debug_benchmark_results.json"
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark finished!")
    print(f"Results saved to {results_path}")

    # Print Summary
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    print("\n--- Benchmark Summary ---")
    print(f"Total Questions Processed: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-------------------------\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())