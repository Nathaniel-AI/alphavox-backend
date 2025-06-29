import os
import sys
import traceback

def safe_run(main_func, fix_func=None, max_attempts=3):
    """Run main_func, and if it crashes, call fix_func and retry."""
    attempts = 0
    while attempts < max_attempts:
        try:
            main_func()
            break
        except Exception as e:
            print(f"[AlphaVox] Error detected: {e}")
            traceback.print_exc()
            if fix_func:
                print("[AlphaVox] Attempting to fix the problem...")
                fix_func(e)
            else:
                print("[AlphaVox] No fix function provided.")
            attempts += 1
    else:
        print("[AlphaVox] Max attempts reached. Shutting down.")

# EXAMPLE FIXER: Replace a buggy line in code
def example_fix_func(error):
    # This is a naive example!
    bug_file = "app/main.py"
    backup_file = f"{bug_file}.bak"
    os.rename(bug_file, backup_file)
    with open(backup_file) as f, open(bug_file, "w") as out:
        for line in f:
            if "BAD_FUNCTION_CALL()" in line:
                out.write("# Auto-fixed: removed buggy line\n")
            else:
                out.write(line)
    print("[AlphaVox] Attempted to patch the code.")

# EXAMPLE USAGE
if __name__ == "__main__":
    

