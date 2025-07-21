import subprocess
import sys

def run_pipeline():
    print("=== X-Ray Synchrotron Classification Pipeline ===\n")
    
    #  Linear Probe (baseline)
    print(" Linear Probe Training...")
    subprocess.run([sys.executable, "scripts/linear_probe_training.py"])
    


    #  Self-Supervised Learning (foundation model change component)
    print("Self-Supervised Learning...")
    subprocess.run([sys.executable, "scripts/self_supervised_training.py"])
    

    #PEFT Fine-tuning
    print("PEFT Fine-tuning...")
    subprocess.run([sys.executable, "scripts/peft_finetune.py"])

if __name__ == "__main__":
    run_pipeline()
