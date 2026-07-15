import math

def lda_pharmacokinetics_game():
    # 1. Scenario: Intensive Care Pharmacokinetic Target Verification
    print("--- 🧬 THE PEAK PREDICTOR: LDA PHARMACOKINETICS ENGINE 🧬 ---")
    print("Mission: Build a spatial discriminant model to intercept toxic antibiotic peaks.")
    print("Goal: Calculate class distributions to maximize the separation between patient outcomes.")

    # 2. Historical Training Reference Records (Continuous Feature to Binary Target)
    # Feature (X): Creatinine Clearance Rate (mL/min)
    # Target (y): 0 = Safe/Therapeutic Peak, 1 = Toxic/Overdose Peak
    history = [
        {"clearance": 45.0,  "toxic": 1},
        {"clearance": 55.0,  "toxic": 1},
        {"clearance": 85.0,  "toxic": 0},
        {"clearance": 105.0, "toxic": 0},
    ]

    print("\n--- 🖥️ HISTORICAL ICU METABOLIC PATIENT DATA ---")
    for idx, case in enumerate(history):
        status = "⚠️ TOXIC EXPOSURE" if case["toxic"] == 1 else "✅ SAFE/THERAPEUTIC"
        print(f"Patient {idx+1}: Renal Clearance = {case['clearance']} mL/min -> Outcome: {status}")

    # 3. Step 1: Calculate Prior Probabilities P(y) and Class-Specific Means (μ)
    total_records = len(history)
    toxic_cases = [c for c in history if c["toxic"] == 1]
    safe_cases = [c for c in history if c["toxic"] == 0]

    prior_toxic = len(toxic_cases) / total_records
    prior_safe = len(safe_cases) / total_records

    mean_toxic = sum(c["clearance"] for c in toxic_cases) / len(toxic_cases)
    mean_safe = sum(c["clearance"] for c in safe_cases) / len(safe_cases)

    # 4. Step 2: Calculate the Shared (Pooled) Variance (σ²)
    # LDA assumes all classes share the exact same variance matrix.
    squared_diffs = 0.0
    for c in history:
        if c["toxic"] == 1:
            squared_diffs += (c["clearance"] - mean_toxic) ** 2
        else:
            squared_diffs += (c["clearance"] - mean_safe) ** 2
    
    # Degrees of freedom: total samples minus number of classes (n - k)
    pooled_variance = squared_diffs / (total_records - 2)

    print("\n--- 📊 STEP 1: RESOLVING GAUSSIAN DISTRIBUTION METRICS ---")
    print(f"Mean Clearance for Toxic Class (μ₁): {mean_toxic:.2f} mL/min")
    print(f"Mean Clearance for Safe Class  (μ₀): {mean_safe:.2f} mL/min")
    print(f"Calculated Shared Pooled Variance (σ²): {pooled_variance:.2f}")

    # 5. Incoming Real-Time Intensive Care Query
    # A patient's lab results come back showing a baseline clearance rate of 63.0 mL/min
    test_patient_x = 63.0
    print(f"\n--- 🚨 SYSTEM ALERT: CRITICAL INFUSION INTAKE ---")
    print(f"Evaluating Incoming Patient Profile -> Renal Clearance Index (X): {test_patient_x} mL/min")

    # 6. Step 3: Compute Linear Discriminant Functions (δ)
    # Discriminant Equation: δ_k(x) = x * (μ_k / σ²) - (μ_k² / 2σ²) + ln(P(k))
    delta_toxic = (test_patient_x * (mean_toxic / pooled_variance)) - ((mean_toxic ** 2) / (2 * pooled_variance)) + math.log(prior_toxic)
    delta_safe  = (test_patient_x * (mean_safe / pooled_variance))  - ((mean_safe ** 2)  / (2 * pooled_variance)) + math.log(prior_safe)

    print("\n--- 🔄 COMPUTING LINEAR DISCRIMINANT FUNCTIONS ---")
    print(f"Score for Toxic Class (δ₁): {delta_toxic:.4f}")
    print(f"Score for Safe Class  (δ₀): {delta_safe:.4f}")

    # 7. Maximum Decision Rule Classifier Choice
    if delta_toxic > delta_safe:
        prediction = 1
        verdict = "⚠️ TOXIC PEAK PREDICTED: OVERRIDE INITIATED - REDUCE DOSAGE VOLUME BY 35%"
    else:
        prediction = 0
        verdict = "✅ SAFE PROFILE PREDICTED: CLEAR TO DISPENSE STANDARD LOADING DOSE"

    print(f"\nAutomated Clinical Decision: {verdict}")

    # 8. Ground Truth Quality Audit
    actual_truth = 1 
    if prediction == actual_truth:
        print("\n🏆 SUCCESS: Your LDA model successfully intercepted a toxic overdose path!")
        print("The customized safety adjustment was applied before the infusion pump was turned on.")
    else:
        print("\n💥 DOSING FAULT: Discriminant misclassification! An unsafe dosage peak was routed to the patient.")

if __name__ == "__main__":
    lda_pharmacokinetics_game()
