"""Test verification agent fix."""

import requests
import json

def test_question(question):
    """Test a single question."""
    print("\n" + "="*70)
    print(f"Question: {question}")
    print("="*70)
    
    response = requests.post(
        "http://localhost:8000/qa",
        json={"question": question},
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n📋 PLAN:")
        print(result.get('plan', 'No plan')[:200])
        
        print("\n🔍 SUB-QUESTIONS:")
        for i, sq in enumerate(result.get('sub_questions', []), 1):
            print(f"  {i}. {sq}")
        
        print("\n✅ FINAL ANSWER:")
        answer = result.get('answer', 'No answer')
        print(answer)
        
        # Check if answer looks correct
        print("\n🔍 ANSWER QUALITY CHECK:")
        if len(answer) < 50:
            print("  ⚠️  Answer seems too short")
        elif "return" in answer.lower() and "as-is" in answer.lower():
            print("  ❌ Answer contains verification meta-text!")
        elif "accurate" in answer.lower() and "supported" in answer.lower():
            print("  ❌ Answer is verification analysis, not actual answer!")
        else:
            print("  ✅ Answer looks good!")
        
        return True
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    print("🧪 Testing Verification Agent Fix")
    
    # Test with multiple questions
    questions = [
        "What are vector databases?",
        "How do HNSW indexes work?",
        "Compare vector databases to traditional databases"
    ]
    
    for q in questions:
        test_question(q)
        print("\n" + "="*70)
        input("Press Enter for next test...")